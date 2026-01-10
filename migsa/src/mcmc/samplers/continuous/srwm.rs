use std::fmt::Debug;
use std::marker::PhantomData;

use rand::Rng;
use rv::prelude::Gaussian;
use rv::traits::Sampleable;

use crate::mcmc::{Model, Sampler};
use crate::Lens;

pub struct SRWM<X, M, L>
where
    L: Lens<M, X>,
{
    proposal_dist: Gaussian,
    lens: L,
    _phantom_x: PhantomData<X>,
    _phantom_m: PhantomData<M>,
}

impl<X, M, L> SRWM<X, M, L>
where
    L: Lens<M, X>,
{
    pub const fn new(proposal_dist: Gaussian, lens: L) -> Self {
        Self {
            lens,
            proposal_dist,
            _phantom_x: PhantomData,
            _phantom_m: PhantomData,
        }
    }
}

impl<M, L, D> Sampler<M, D> for SRWM<f64, M, L>
where
    M: Model<D> + Clone + Debug,
    L: Lens<M, f64>,
{
    fn step<R: Rng>(&mut self, model: M, data: &D, rng: &mut R) -> M {
        let delta: f64 = self.proposal_dist.draw(rng);
        let proposed_model = self.lens.set(model.clone(), self.lens.get(&model) + delta);

        let orig_ln_f = model.ln_score(data);
        let new_ln_f = proposed_model.ln_score(data);

        let log_accpt = new_ln_f - orig_ln_f;
        let accpt = log_accpt.exp();

        if rng.random::<f64>() < accpt {
            proposed_model
        } else {
            model
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use rv::prelude::*;

    use super::SRWM;
    use crate::mcmc::{GewekeTest, Model, PriorModel, ResampleModel};

    #[test]
    fn geweke() {
        #[derive(Debug, Default, Clone)]
        struct GaussainUnknownMean {
            mean: f64,
        }

        impl GaussainUnknownMean {
            fn get(&self) -> &f64 {
                &self.mean
            }
            #[allow(clippy::needless_update)]
            fn set(self, mean: f64) -> Self {
                Self { mean, ..self }
            }
        }

        impl Model<Vec<f64>> for GaussainUnknownMean {
            fn ln_score(&self, data: &Vec<f64>) -> f64 {
                let prior = Gaussian::new_unchecked(0.0, 1.0);
                let m = Gaussian::new_unchecked(self.mean, 1.0);
                data.iter().map(|x| m.ln_f(x)).sum::<f64>() + prior.ln_f(&self.mean)
            }
        }

        impl PriorModel<Vec<f64>> for GaussainUnknownMean {
            fn draw_from_prior<R: Rng>(rng: &mut R) -> Self {
                let mean = Gaussian::new_unchecked(0.0, 1.0).draw(rng);
                Self { mean }
            }
        }

        impl ResampleModel<Vec<f64>> for GaussainUnknownMean {
            fn resample_data<R: Rng>(&mut self, _data: Option<&Vec<f64>>, rng: &mut R) -> Vec<f64> {
                Gaussian::new_unchecked(self.mean, 1.0).sample(10, rng)
            }
        }

        let mut rng = SmallRng::seed_from_u64(0xF00D);

        let sampler: SRWM<f64, GaussainUnknownMean, _> = SRWM::new(
            Gaussian::new_unchecked(0.0, 0.4),
            (GaussainUnknownMean::get, GaussainUnknownMean::set),
        );

        sampler.assert_geweke(
            crate::mcmc::GewekeTestOptions {
                thinning: 100,
                n_samples: 1000,
                burn_in: 1000,
                min_p_value: 0.05,
                stat_map: |(GaussainUnknownMean { mean }, data): (
                    GaussainUnknownMean,
                    Vec<f64>,
                )| {
                    #[allow(clippy::cast_precision_loss)]
                    let n = data.len() as f64;
                    vec![mean, (data.into_iter().sum::<f64>() / n)]
                },
            },
            &mut rng,
        );
    }
}
