

/*
use std::fmt::Debug;

use crate::{
    data::Partition,
    mcmc::{
        samplers::partition::{gibbs::PartitionGibbs, split_merge::PartitionSplitMerge},
        McmcSampler, McmcSamplerStep, NestedMcmcSampler,
    },
};

use once_cell::sync::OnceCell;
use rand::Rng;
use rv::{
    misc::logsumexp,
    prelude::{Crp, CrpError},
    traits::{ConjugatePrior, HasSuffStat, Rv},
};


pub struct Dpmm<X, Fx, Pr, R>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
    R: Rng + ?Sized,
{
    sampler: NestedMcmcSampler<Partition<Fx::Stat, X>, R>,
    crp: Crp,
    prior: Pr,
}

impl<X, Fx, Pr, R> Dpmm<X, Fx, Pr, R>
where
    X: Clone + 'static,
    Fx: Rv<X> + HasSuffStat<X> + 'static,
    Pr: ConjugatePrior<X, Fx> + Clone + 'static,
    Fx::Stat: Clone + Debug,
    R: Rng,
{
    pub fn new(
        data: impl Into<Vec<X>>,
        concentration: f64,
        prior: Pr,
        rng: &mut R,
    ) -> Result<Self, CrpError> {
        let data = data.into();
        let crp = Crp::new(concentration, data.len())?;

        let fx = prior.draw(rng);

        let mut partition = Partition::new_stat(move || fx.empty_suffstat());
        partition.append(data);

        let raw_partition: rv::data::Partition = crp.draw(rng);

        for (i, p) in raw_partition.z().iter().enumerate() {
            partition.assign(i, *p);
        }

        let sampler = NestedMcmcSampler::new(vec![
            Box::new(PartitionGibbs::new(
                partition.clone(),
                prior.clone(),
                crp.clone(),
            )) as Box<dyn McmcSampler<R, State = Partition<Fx::Stat, X>>>,
            Box::new(PartitionSplitMerge::new(
                partition,
                prior.clone(),
                crp.clone(),
            )) as Box<dyn McmcSampler<R, State = Partition<Fx::Stat, X>>>,
        ]);

        Ok(Self {
            crp,
            prior,
            sampler,
        })
    }

    pub fn sample(
        &mut self,
        warmup: usize,
        thinning: usize,
        size: usize,
        rng: &mut R,
    ) -> DpmmSample<X, Fx, Pr> {
        self.warmup(warmup, rng);
        let samples = self.iter_step(thinning, rng).take(size).collect();
        DpmmSample {
            alpha: self.crp.alpha(),
            prior: &self.prior,
            samples,
            ln_m_cache: OnceCell::new(),
        }
    }
}

impl<X, Fx, Pr, R> McmcSampler<R> for Dpmm<X, Fx, Pr, R>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X> + 'static,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
    R: Rng + ?Sized,
{
    type State = Partition<Fx::Stat, X>;

    fn step(&mut self, rng: &mut R) {
        self.sampler.step(rng);
    }

    fn state(&self) -> &Self::State {
        self.sampler.state()
    }

    fn set_state(&mut self, state: Self::State) {
        self.sampler.set_state(state);
    }

    fn ln_score(&self) -> f64 {
        self.sampler.ln_score()
    }
}

#[derive(Clone)]
pub struct DpmmSample<'a, X, Fx, Pr>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
{
    pub alpha: f64,
    pub prior: &'a Pr,
    pub samples: Vec<McmcSamplerStep<Partition<Fx::Stat, X>>>,
    ln_m_cache: OnceCell<Vec<f64>>,
}

impl<'a, X, Fx, Pr> DpmmSample<'a, X, Fx, Pr>
where
    X: Clone + Send + Sync,
    Fx: Rv<X> + HasSuffStat<X> + Send + Sync,
    Pr: ConjugatePrior<X, Fx> + Send + Sync,
    Fx::Stat: Clone + Debug + Send + Sync,
{
    pub fn ln_pp(&self, x: &X) -> f64 {
        let ln_weights = self.ln_m_cache.get_or_init(|| {
            let weights: Vec<f64> = self
                .samples
                .iter()
                .map(|s| s.state.ln_m(self.prior))
                .collect();

            let total_weight = logsumexp(&weights);

            weights.into_iter().map(|x| x - total_weight).collect()
        });

        let alpha = self.alpha;
        let states: Vec<&Partition<Fx::Stat, X>> =
            self.samples.iter().map(|sample| &sample.state).collect();

        logsumexp(
            &states
                .iter()
                .zip(ln_weights.into_iter())
                .map(|(partition, ln_w)| ln_w + partition.ln_pp(self.prior, alpha, x))
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;
    use rand::rngs::SmallRng;
    use rv::{
        data::MvGaussianSuffStat,
        prelude::{MvGaussian, NormalInvWishart},
    };

    use crate::{
        dpmm::{Dpmm, Partition},
        mcmc::GewekeTest,
    };

    #[test]
    fn geweke() {
        struct DpmmGeweke {
            sampler: Dpmm<DVector<f64>, MvGaussian, NormalInvWishart, SmallRng>,
        }

        impl GewekeTest<Dpmm<DVector<f64>, MvGaussian, NormalInvWishart, SmallRng>, SmallRng>
            for DpmmGeweke
        {
            type Model = Partition<MvGaussianSuffStat, DVector<f64>>;

            type X = DVector<f64>;

            fn sampler(
                &mut self,
            ) -> &mut Dpmm<DVector<f64>, MvGaussian, NormalInvWishart, SmallRng> {
                &mut self.sampler
            }

            fn draw_from_prior(&mut self, rng: &mut SmallRng) -> Self::Model {
                todo!()
            }

            fn stat_map(
                data_and_state: crate::mcmc::DataAndState<Self::X, Self::Model>,
            ) -> Vec<f64> {
                todo!()
            }

            fn resample_data(
                &mut self,
                data_and_state: crate::mcmc::DataAndState<Self::X, Self::Model>,
                rng: &mut SmallRng,
            ) -> crate::mcmc::DataAndState<Self::X, Self::Model> {
                todo!()
            }
        }

        todo!()
        //DpmmGeweke {
        //    sampler: Dpmm::new(),
        //}
    }
}

*/
