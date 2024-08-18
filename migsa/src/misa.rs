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

use nalgebra::{dvector, DMatrix, DVector};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rv::dist::{Mixture, MvGaussian};
use rv::prelude::NormalInvWishart;
use rv::traits::{Mean, MultivariateRv, Rv};

use crate::mcmc::samplers::partition::gibbs::PartitionGibbs;
use crate::mcmc::Sampler;
use crate::models::mixture::ConjugateMixtureModel;
use crate::models::Model;

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
    parameter_dist: G,
    f: F,
    n_evals: usize,
    n: usize,
    warmup: usize,
    thinning: NonZeroUsize,
    rng: &mut R,
) -> Vec<Vec<f64>>
where
    G: MultivariateRv<DVector<f64>, f64>,
    G::Atom: std::fmt::Debug,
    F: Fn(&DVector<f64>) -> f64,
    R: Rng,
{
    let xs = parameter_dist.sample(n_evals, rng);
    let y: Vec<f64> = xs.iter().map(|x| f(x)).collect();
    let n_params = parameter_dist.dimensions();

    // A n_params x n x n
    let joints: Vec<Vec<DVector<f64>>> = (0..n_params)
        .map(|p| {
            xs.iter()
                .zip(y.iter())
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
                    1.0,
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
        .iter_sample(model, &data, rng, |model| {
            model
                .joint_models
                .iter()
                .map(|jmodel| {
                    let mut rng = SmallRng::from_entropy();
                    let mixture: Mixture<MvGaussian> = jmodel.draw(&mut rng);
                    //let mixture_mean: DVector<f64> = mixture
                    //    .weights()
                    //    .iter()
                    //    .zip(mixture.components().iter())
                    //    .map(|(&w, c)| c.mean().unwrap() * w)
                    //    .sum();

                    let mixture_y = mixture.marginal(1).unwrap();
                    let mixture_x = mixture.marginal(0).unwrap();
                    // let scale_x = mixture
                    //     .weights()
                    //     .iter()
                    //     .zip(mixture.components().iter())
                    //     .map(|(&w, c)| c.cov().get((0, 0)).unwrap() * w)
                    //     .sum::<f64>();
                    // let scale_y = mixture
                    //     .weights()
                    //     .iter()
                    //     .zip(mixture.components().iter())
                    //     .map(|(&w, c)| c.cov().get((1, 1)).unwrap() * w)
                    //     .sum::<f64>();

                    // Monte Carlo integration
                    //mixture.sample_stream(&mut rng)

                    let xys: Vec<DVector<f64>> = mixture.sample(3_000, &mut rng);
                    xys.iter()
                        .map(|xy| {
                            let ln_f_x = mixture_x.ln_f(&xy[0]);
                            let ln_f_y = mixture_y.ln_f(&xy[1]);
                            let ln_f_xy = mixture.ln_f(&xy);

                            ((ln_f_y + ln_f_x - ln_f_xy).exp() - 1.0).abs() / 2.0
                        })
                        .sum::<f64>()
                        / (xys.len() as f64)

                    // Quadrature Version
                    // let delta_fn = |mut x, mut y| {
                    //     x = x * scale_x + mixture_mean[0];
                    //     y = y * scale_y + mixture_mean[1];

                    //     let ln_f_x = mixture_x.ln_f(&x);
                    //     let ln_f_y = mixture_y.ln_f(&y);
                    //     let ln_f_xy = mixture.ln_f(&dvector![x, y]);

                    //     ((ln_f_y + ln_f_x - ln_f_xy).exp() - 1.0).abs() / 2.0 * ln_f_xy.exp()
                    // };

                    // dblquad(delta_fn, (-5.0, 5.0), (-5.0, 5.0), 1000, 1000)
                })
                .collect()
        })
        .skip(warmup)
        .step_by(thinning.into())
        //.filter(|deltas: &Vec<f64>| deltas.iter().all(|&d| d <= 1.0))
        .take(n)
        .collect()
}

pub fn dblquad<F>(f: F, x_bounds: (f64, f64), y_bounds: (f64, f64), nx: usize, ny: usize) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let (a, b) = x_bounds;
    let (c, d) = y_bounds;

    let dx = (b - a) / (nx as f64);
    let dy = (d - c) / (ny as f64);

    let mut int = f(a, c) + f(b, c) + f(a, d) + f(b, d);

    for i in 0..nx {
        let xi = a + dx * ((i + 1) as f64);
        int += 2.0 * f(xi, c) + 2.0 * f(xi, d);
    }

    for j in 0..ny {
        let yj = c + dy * ((j + 1) as f64);
        int += 2.0 * f(a, yj) + 2.0 * f(b, yj);
    }

    for i in 0..nx {
        let xi = a + dx * ((i + 1) as f64);
        for j in 0..ny {
            let yj = c + dy * ((j + 1) as f64);
            int += 4.0 * f(xi, yj);
        }
    }

    dx * dy / 4.0 * int
}

#[cfg(test)]
mod tests {
    use nalgebra::{dmatrix, dvector, DVector};
    use rand::SeedableRng;
    use rv::dist::MvGaussian;

    use super::*;

    #[test]
    fn wei_example_1() {
        let w = dvector![1.5, 1.6, 1.7, 1.8, 1.9, 2.0];
        let f = |x: &DVector<f64>| -> f64 { w.dot(&x) };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);

        let delta_is = single_loop_deltas(
            MvGaussian::new_unchecked(DVector::zeros(6), DMatrix::identity(6, 6)),
            f,
            1_000,
            1_000,
            15,
            NonZeroUsize::new(1).unwrap(),
            &mut rng,
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

        println!("deltas = {:#?}", means);
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
        let parameter_dist = MvGaussian::new(means, cov).unwrap();

        fn d(p: &DVector<f64>) -> f64 {
            let x = p[0];
            let y = p[1];
            let e = p[2];
            let w = p[3];
            let t = p[4];
            let l = p[5];

            (4.0 * l.powi(3) / (e * w * t))
                * ((x / w.powi(2)).powi(2) + (y / t.powi(2)).powi(2)).sqrt()
        }

        let delta_is = single_loop_deltas(
            parameter_dist,
            d,
            1_000,
            1_000,
            2_000,
            NonZeroUsize::new(10).unwrap(),
            &mut rng,
        );

        let f = std::fs::File::create("./deltas.json").unwrap();
        serde_json::to_writer(f, &delta_is).unwrap();

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

        println!("deltas = {:#?}", means);
        assert::close(&means, &[0.056, 0.030, 0.081, 0.179, 0.146, 0.235], 1E-2);
    }

    #[test]
    fn dblquad_constant() {
        let int = dblquad(|_x, _y| 1.0, (-1.0, 1.0), (-1.0, 1.0), 1000, 1000);
        assert::close(int, 4.0, 1e-2);
    }
}
