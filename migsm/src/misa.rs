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
use rand::Rng;
use rv::prelude::{NormalGamma, NormalInvWishart};
use rv::traits::Rv;

use crate::mcmc::samplers::partition::gibbs::PartitionGibbs;
use crate::mcmc::Sampler;
use crate::models::mixture::ConjugateMixtureModel;
use crate::models::Model;

#[derive(Debug, Clone)]
struct InnerModels<YD, JD, YM, JM>
where
    YM: Model<YD>,
    JM: Model<JD>,
{
    y_model: YM,
    joint_models: Vec<JM>,
    _phantom_yd: PhantomData<YD>,
    _phantom_jd: PhantomData<JD>,
}

#[derive(Debug, Clone)]
struct InnerData<YD, JD> {
    y: YD,
    joints: Vec<JD>,
}

struct InnerSampler<YD, JD, YM, JM, YS, JS>
where
    YM: Model<YD>,
    JM: Model<JD>,
    YS: Sampler<YM, YD>,
    JS: Sampler<JM, JD>,
{
    y_sampler: YS,
    joint_samplers: Vec<JS>,
    _phantom_yd: PhantomData<YD>,
    _phantom_jd: PhantomData<JD>,
    _phantom_ym: PhantomData<YM>,
    _phantom_jm: PhantomData<JM>,
}

impl<YD, JD, YM, JM> Model<InnerData<YD, JD>> for InnerModels<YD, JD, YM, JM>
where
    YM: Model<YD>,
    JM: Model<JD>,
{
    fn ln_score(&self, data: &InnerData<YD, JD>) -> f64 {
        self.y_model.ln_score(&data.y)
            + self
                .joint_models
                .iter()
                .zip(data.joints.iter())
                .map(|(m, d)| m.ln_score(d))
                .sum::<f64>()
    }
}

impl<YD, JD, YM, JM, YS, JS> Sampler<InnerModels<YD, JD, YM, JM>, InnerData<YD, JD>>
    for InnerSampler<YD, JD, YM, JM, YS, JS>
where
    YM: Model<YD>,
    JM: Model<JD>,
    YS: Sampler<YM, YD>,
    JS: Sampler<JM, JD>,
{
    fn step<R: Rng>(
        &mut self,
        model: InnerModels<YD, JD, YM, JM>,
        data: &InnerData<YD, JD>,
        rng: &mut R,
    ) -> InnerModels<YD, JD, YM, JM> {
        let y_model = self.y_sampler.step(model.y_model, &data.y, rng);

        let joint_models = self
            .joint_samplers
            .iter_mut()
            .zip(model.joint_models.into_iter())
            .zip(data.joints.iter())
            .map(|((sampler, model), data)| sampler.step(model, data, rng))
            .collect();

        InnerModels {
            y_model,
            joint_models,
            _phantom_yd: PhantomData,
            _phantom_jd: PhantomData,
        }
    }
}

pub fn deltas<F, R>(
    f: F,
    n_params: usize,
    n: usize,
    warmup: usize,
    thinning: NonZeroUsize,
    rng: &mut R,
) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
    R: Rng,
{
    let g = rv::prelude::Gaussian::new_unchecked(0.0, 1.0);
    let xs: Vec<Vec<f64>> = (0..n).map(|_| g.sample(n_params, rng)).collect();
    let y: Vec<f64> = xs.iter().map(|x| f(&x)).collect();

    // A n_params x n x n
    let joints: Vec<Vec<DVector<f64>>> = (0..n_params)
        .map(|p| {
            xs.iter()
                .zip(y.iter())
                .map(|(x, y)| dvector![x[p], *y])
                .collect()
        })
        .collect();

    let data = InnerData { y, joints };

    let model = InnerModels {
        y_model: ConjugateMixtureModel::new(
            NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap(),
            data.y.iter(),
            1.0,
            rng,
        ),
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
        _phantom_yd: PhantomData,
        _phantom_jd: PhantomData,
    };

    let mut sampler = InnerSampler {
        y_sampler: PartitionGibbs::new(),
        joint_samplers: data.joints.iter().map(|_| PartitionGibbs::new()).collect(),
        _phantom_yd: PhantomData,
        _phantom_jd: PhantomData,
        _phantom_ym: PhantomData,
        _phantom_jm: PhantomData,
    };

    sampler
        .iter_sample(model, &data, rng, |model| {
            let pys: Vec<f64> = data.y.iter().map(|y| model.y_model.f(y)).collect();

            data.joints
                .iter()
                .zip(model.joint_models.iter())
                .map(|(xys, m)| {
                    assert_eq!(pys.len(), xys.len());
                    xys.iter()
                        .zip(pys.iter())
                        .map(|(xy, py)| {
                            let pxy = m.f(xy);
                            ((py * g.f(&xy[0]) / pxy) - 1.0).abs()
                        })
                        .sum::<f64>()
                        / 2.0
                        / (xys.len() as f64)
                })
                .collect()
        })
        .skip(warmup)
        .step_by(thinning.into())
        .take(n)
        .collect()
}

#[cfg(test)]
mod tests {
    use nalgebra::{dvector, DVector};
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn wei_example_1() {
        let w = dvector![1.5, 1.6, 1.7, 1.8, 1.9, 2.0];
        let f = |x: &[f64]| -> f64 {
            let x = DVector::from_column_slice(x);
            w.dot(&x)
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);

        let delta_is = deltas(f, 6, 1000, 10, NonZeroUsize::new(1).unwrap(), &mut rng);

        let n = delta_is.len();

        let mut means = [0.0_f64; 6];

        for delta_sample in delta_is {
            for i in 0..6 {
                means[i] += delta_sample[i];
            }
        }
        for i in 0..6 {
            means[i] /= n as f64;
        }

        println!("deltas = {:#?}", means);
        assert::close(&means, &[0.119, 0.129, 0.138, 0.148, 0.158, 0.168], 1E-2);
    }
}
