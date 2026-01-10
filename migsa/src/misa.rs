//! Moment Independent Sensitivity Analysis methods from
//!
//! ```bibtex
//! @article{Wei2013,
//!    abstract = {The moment-independent sensitivity analysis (SA) is one of the most popular SA techniques. It aims at measuring the contribution of input variable(s) to the probability density function (PDF) of model output. However, compared with the variance-based one, robust and efficient methods are less available for computing the moment-independent SA indices (also called delta indices). In this paper, the Monte Carlo simulation (MCS) methods for moment-independent SA are investigated. A double-loop MCS method, which has the advantages of high accuracy and easy programming, is firstly developed. Then, to reduce the computational cost, a single-loop MCS method is proposed. The later method has several advantages. First, only a set of samples is needed for computing all the indices, thus it can overcome the problem of curse of dimensionality. Second, it is suitable for problems with dependent inputs. Third, it is purely based on model output evaluation and density estimation, thus can be used for model with high order (>2) interactions. At last, several numerical examples are introduced to demonstrate the advantages of the proposed methods. © 2012 Elsevier Ltd.},
//!    author = {Pengfei Wei and Zhenzhou Lu and Xiukai Yuan},
//!    doi = {10.1016/j.ress.2012.09.005},
//!    issn = {09518320},
//!    journal = {Reliability Engineering and System Safety},
//!    keywords = {Delta indices,Kernel density estimation,Moment-independent sensitivity analysis,Monte Carlo simulation},
//!    pages = {60-67},
//!    publisher = {Elsevier Ltd},
//!    title = {Monte Carlo simulation for moment-independent sensitivity analysis},
//!    volume = {110},
//!    year = {2013},
//! }
//! ```

use std::marker::PhantomData;
use std::num::NonZeroUsize;

use nalgebra::{DMatrix, DVector, dvector};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rv::dist::{Crp, Mixture, MvGaussian};
use rv::prelude::NormalInvWishart;
use rv::traits::{HasDensity, Sampleable};

use crate::mcmc::Sampler;
use crate::mcmc::samplers::partition::gibbs::PartitionGibbs;
use crate::models::Model;
use crate::models::mixture::ConjugateMixtureModel;
use crate::utils::{MeanAndVariance, Multivariate};

#[derive(Debug, Clone)]
struct InnerModels<D, M>
where
    M: Model<D>,
{
    joint_models: Vec<M>,
    _phantom_jd: PhantomData<D>,
}

#[derive(Debug, Clone)]
struct InnerData<D> {
    joints: Vec<D>,
}

struct InnerSampler<D, M, S>
where
    M: Model<D>,
    S: Sampler<M, D>,
{
    joint_samplers: Vec<S>,
    _phantom_jd: PhantomData<D>,
    _phantom_jm: PhantomData<M>,
}

impl<D, M> Model<InnerData<D>> for InnerModels<D, M>
where
    M: Model<D>,
{
    fn ln_score(&self, data: &InnerData<D>) -> f64 {
        self.joint_models
            .iter()
            .zip(data.joints.iter())
            .map(|(m, d)| m.ln_score(d))
            .sum::<f64>()
    }
}

impl<D, M, S> Sampler<InnerModels<D, M>, InnerData<D>> for InnerSampler<D, M, S>
where
    M: Model<D>,
    S: Sampler<M, D>,
{
    fn step<R: Rng>(
        &mut self,
        model: InnerModels<D, M>,
        data: &InnerData<D>,
        rng: &mut R,
    ) -> InnerModels<D, M> {
        let joint_models = self
            .joint_samplers
            .iter_mut()
            .zip(model.joint_models)
            .zip(data.joints.iter())
            .map(|((sampler, model), data)| sampler.step(model, data, rng))
            .collect();

        InnerModels {
            joint_models,
            _phantom_jd: PhantomData,
        }
    }
}

pub fn single_loop_deltas<F, G, R>(
    parameter_dist: &G,
    f: F,
    n_evals: usize,
    n_mcmc_iters: usize,
    n_mc_delta_iters: usize,
    warmup: usize,
    thinning: NonZeroUsize,
    rng: &mut R,
    alpha: f64,
) -> Vec<Vec<f64>>
where
    G: Sampleable<DVector<f64>>,
    F: Fn(&DVector<f64>) -> f64,
    R: Rng,
{
    let xs = parameter_dist.sample(n_evals, rng);
    let y: Vec<f64> = xs.iter().map(f).collect();

    single_loop_deltas_from_outputs(
        &xs,
        &y,
        n_mcmc_iters,
        n_mc_delta_iters,
        warmup,
        thinning,
        rng,
        alpha,
    )
}

/// Calculate the single loop deltas sensitivities
///
/// # Parameters
/// - `parameters`: Parameter sample.
/// - `responses`: model responses under parameter sample.
/// - `n_mcmc_iters`: Number of MCMC iterations for DPMM.
/// - `n_mc_delta_iters`: Number of samples from mixture distribition to estimate the delta
///   parameters.
/// - `warmup`: MCMC warmup iterations to discard.
/// - `thinning`: MCMC thinning to decorrelate samples.
/// - `rng`: The RNG source
/// - `alpha`: the Dirichlet concentration parameter.
///
/// # Panics
/// This function will panic if the number of parameter samples is zero.
pub fn single_loop_deltas_from_outputs<R>(
    parameters: &[DVector<f64>],
    responses: &[f64],
    n_mcmc_iters: usize,
    n_mc_delta_iters: usize,
    warmup: usize,
    thinning: NonZeroUsize,
    rng: &mut R,
    alpha: f64,
) -> Vec<Vec<f64>>
where
    R: Rng,
{
    let n_params = parameters.first().expect("At least one input").len();

    // A n_params x n x n
    let joints: Vec<Vec<DVector<f64>>> = (0..n_params)
        .map(|p| {
            parameters
                .iter()
                .zip(responses.iter())
                .map(|(x, y)| dvector![x[p], *y])
                .collect()
        })
        .collect();

    let data = InnerData { joints };

    let model = InnerModels {
        joint_models: data
            .joints
            .iter()
            .map(|xy| {
                ConjugateMixtureModel::new(
                    NormalInvWishart::new(DVector::zeros(2), 1.0, 2, DMatrix::identity(2, 2))
                        .expect("Should be valid"),
                    xy.iter(),
                    Crp::new_unchecked(alpha, xy.len()),
                    rng,
                )
            })
            .collect(),
        _phantom_jd: PhantomData,
    };

    let mut sampler = InnerSampler {
        joint_samplers: data.joints.iter().map(|_| PartitionGibbs::new()).collect(),
        _phantom_jd: PhantomData,
        _phantom_jm: PhantomData,
    };

    // TODO: Don't actually do the delta calculations before warmup and thinning.
    sampler
        .iter(model, &data, rng)
        .skip(warmup)
        .step_by(thinning.into())
        .take(n_mcmc_iters)
        .map(|model| {
            model
                .joint_models
                .iter()
                .map(|jmodel| {
                    let mut rng = SmallRng::from_os_rng();
                    let mixture: Mixture<MvGaussian> = jmodel.draw(&mut rng);
                    let marginals = mixture.univariate_marginals();
                    let mixture_x = &marginals[0];
                    let mixture_y = &marginals[1];

                    // Monte Carlo integration
                    mixture
                        .sample_stream(&mut rng)
                        .map(|xy| {
                            let ln_f_x = mixture_x.ln_f(&xy[0]);
                            let ln_f_y = mixture_y.ln_f(&xy[1]);
                            let ln_f_joint = mixture.ln_f(&xy);

                            (ln_f_y + ln_f_x - ln_f_joint).exp_m1().abs() / 2.0
                        })
                        .take(n_mc_delta_iters)
                        .fold(
                            MeanAndVariance::<f64>::default(),
                            MeanAndVariance::<f64>::update,
                        )
                        .mean()
                })
                .collect()
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
pub fn dblquad<F>(f: F, x_bounds: (f64, f64), y_bounds: (f64, f64), nx: usize, ny: usize) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let (x_left, x_right) = x_bounds;
    let (y_left, y_right) = y_bounds;

    let dx = (x_right - x_left) / (nx as f64);
    let dy = (y_right - y_left) / (ny as f64);

    let mut int = f(x_left, y_left) + f(x_right, y_left) + f(x_left, y_right) + f(x_right, y_right);

    for i in 0..nx {
        let xi = dx.mul_add((i + 1) as f64, x_left);
        int += 2.0f64.mul_add(f(xi, y_left), 2.0 * f(xi, y_right));
    }

    for j in 0..ny {
        let yj = dy.mul_add((j + 1) as f64, y_left);
        int += 2.0f64.mul_add(f(x_left, yj), 2.0 * f(x_right, yj));
    }

    for i in 0..nx {
        let xi = dx.mul_add((i + 1) as f64, x_left);
        for j in 0..ny {
            let yj = dy.mul_add((j + 1) as f64, y_left);
            int += 4.0 * f(xi, yj);
        }
    }

    dx * dy / 4.0 * int
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use nalgebra::{DVector, dmatrix, dvector};
    use rand::SeedableRng;
    use rv::dist::MvGaussian;

    use super::*;

    #[test]
    fn wei_example_1() {
        let w = dvector![1.5, 1.6, 1.7, 1.8, 1.9, 2.0];
        let f = |x: &DVector<f64>| -> f64 { w.dot(x) };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);

        let delta_is = single_loop_deltas(
            &MvGaussian::new_unchecked(DVector::zeros(6), DMatrix::identity(6, 6)),
            f,
            1_000,
            1_000,
            1_000,
            15,
            NonZeroUsize::new(1).expect("1 is not zero"),
            &mut rng,
            10.0,
        );

        let n = delta_is.len();

        let mut means = [0.0_f64; 6];

        for delta_sample in delta_is {
            for i in 0..6 {
                means[i] += delta_sample[i];
            }
        }
        for mean in &mut means {
            *mean /= n as f64;
        }

        println!("deltas = {means:#?}");
        assert::close(&means, &[0.119, 0.129, 0.138, 0.148, 0.158, 0.168], 1E-2);
    }

    #[test]
    fn wei_example_2() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);

        let rho = dmatrix![
            1.0, 0.2, 0.0, 0.0, 0.0, 0.0;
            0.2, 1.0, 0.0, 0.0, 0.0, 0.0;
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
            0.0, 0.0, 0.0, 1.0, 0.1, 0.1;
            0.0, 0.0, 0.0, 0.1, 1.0, 0.1;
            0.0, 0.0, 0.0, 0.1, 0.1, 1.0;
        ];

        let means = dvector![500.0, 1000.0, 2.9, 2.4487, 3.8884, 100.0];
        let cv: f64 = 0.08;
        let stds = cv * &means;
        let cov = DMatrix::from_diagonal(&stds) * rho * DMatrix::from_diagonal(&stds);
        let parameter_dist = MvGaussian::new_unchecked(means, cov);

        #[allow(clippy::many_single_char_names, clippy::items_after_statements)]
        fn d(p: &DVector<f64>) -> f64 {
            let x = p[0];
            let y = p[1];
            let e = p[2];
            let w = p[3];
            let t = p[4];
            let l = p[5];

            (4.0 * l.powi(3) / (e * w * t)) * (x / w.powi(2)).hypot(y / t.powi(2))
        }

        let delta_is = single_loop_deltas(
            &parameter_dist,
            d,
            1_000,
            1_000,
            1_000,
            2_000,
            NonZeroUsize::new(10).expect("10 is not zero"),
            &mut rng,
            10.0,
        );

        let f = std::fs::File::create("./deltas.json").expect("to read file");
        serde_json::to_writer(f, &delta_is).expect("to write file");

        let n = delta_is.len();

        let mut means = [0.0_f64; 6];

        for delta_sample in delta_is {
            for i in 0..6 {
                means[i] += delta_sample[i];
            }
        }
        for mean in &mut means {
            *mean /= n as f64;
        }

        println!("deltas = {means:#?}");
        assert::close(&means, &[0.056, 0.030, 0.081, 0.179, 0.146, 0.235], 1E-2);
    }

    #[test]
    fn dblquad_constant() {
        let int = dblquad(|_x, _y| 1.0, (-1.0, 1.0), (-1.0, 1.0), 1000, 1000);
        assert::close(int, 4.0, 1e-2);
    }
}
