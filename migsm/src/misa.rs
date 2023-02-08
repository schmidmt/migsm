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

use std::fs::File;

use nalgebra::{dvector, DMatrix, DVector};
use rand::Rng;
use rv::misc::linspace;
use rv::prelude::{NormalGamma, NormalInvWishart};
use rv::traits::Rv;

use crate::mcmc::samplers::partition::gibbs::PartitionGibbs;
use crate::mcmc::Sampler;
use crate::models::mixture::ConjugateMixtureModel;
use crate::models::Model;

#[derive(Debug, Clone)]
struct InnerMDS<YM, YD, YS, JM, JD, JS>
where
    YM: Model<YD>,
    JM: Model<JD>,
    YS: Sampler<YM, YD>,
    JS: Sampler<JM, JD>,
{
    y: YD,
    y_model: YM,
    y_model_sampler: YS,
    joints: Vec<JD>,
    joint_models: Vec<JM>,
    joint_model_samplers: Vec<JS>,
}

impl<YM, YD, YS, JM, JD, JS> InnerMDS<YM, YD, YS, JM, JD, JS>
where
    YM: Model<YD>,
    JM: Model<JD>,
    YS: Sampler<YM, YD>,
    JS: Sampler<JM, JD>,
{
    pub fn step<R: Rng>(mut self, rng: &mut R) -> Self {
        let y_model = self.y_model_sampler.step(self.y_model, &self.y, rng);
        let joint_models = self
            .joint_models
            .into_iter()
            .zip(self.joints.iter())
            .zip(self.joint_model_samplers.iter_mut())
            .map(|((m, joint), sampler)| sampler.step(m, &joint, rng))
            .collect();

        Self {
            y_model,
            joint_models,
            ..self
        }
    }
}

pub fn deltas<F, R>(
    f: F,
    n_params: usize,
    n: usize,
    warmup: usize,
    thinning: usize,
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

    let y_model = ConjugateMixtureModel::new(
        NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap(),
        y.iter(),
        1.0,
        rng,
    );

    // A model of joint samples for each parameter
    let joint_models = joints
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
        .collect();

    let joint_model_samplers = joints.iter().map(|_| PartitionGibbs::new()).collect();

    let mds = InnerMDS {
        y,
        y_model,
        y_model_sampler: PartitionGibbs::new(),
        joints,
        joint_models,
        joint_model_samplers,
    };

    // Warm-up stage
    let mds = (0..warmup).fold(mds, |mds, _| mds.step(rng));

    // Sample Deltas
    let res = (0..(thinning * n))
        .scan(mds, |mds, _| {
            let new_mds = mds.clone().step(rng);

            *mds = new_mds;
            Some(mds.clone())
        })
        .step_by(thinning)
        .map(|mds| {
            let pys: Vec<f64> = mds.y.iter().map(|y| mds.y_model.f(y)).collect();

            mds.joints
                .iter()
                .zip(mds.joint_models.iter())
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
        .collect();
    res
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

        let delta_is = deltas(f, 6, 1000, 100, 1, &mut rng);

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
