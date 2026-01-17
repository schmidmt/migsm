use crate::models::Model;
use rand::Rng;
use rv::misc::{KsAlternative, KsMode, ks_two_sample};
use serde::Serialize;

pub mod samplers;

/// Trait for Markov Chain Monte Carlo Samplers.
pub trait Sampler<M, D>: Sized
where
    M: Model<D> + Clone,
{
    /// Step the Sampler.
    fn step<R: Rng>(&mut self, model: M, data: &D, rng: &mut R) -> M;

    /// Warm-up the sampler
    fn multi_step<R: Rng>(&mut self, model: M, data: &D, steps: usize, rng: &mut R) -> M {
        (0..steps).fold(model, |acc, _| self.step(acc, data, rng))
    }

    fn iter<'a, R: Rng>(
        &'a mut self,
        model: M,
        data: &'a D,
        rng: &'a mut R,
    ) -> SamplerIter<'a, M, D, Self, R>
    where
        M: Model<D> + Clone + 'a,
    {
        SamplerIter {
            sampler: self,
            model: Some(model),
            data,
            rng,
        }
    }

    fn iter_sample<'a, T, F: Fn(&M) -> T + 'a, R: Rng>(
        &mut self,
        model: M,
        data: &'a D,
        rng: &'a mut R,
        f: F,
    ) -> impl Iterator<Item = T> {
        (0..).scan(model, move |model, _| {
            //*model = self.step(model.clone(), data, rng);
            take(model, |m| self.step(m, data, rng));
            Some(f(model))
        })
    }
}

pub struct SamplerIter<'a, M, D, S, R>
where
    S: Sampler<M, D>,
    M: Model<D> + Clone,
    R: Rng,
{
    sampler: &'a mut S,
    model: Option<M>,
    data: &'a D,
    rng: &'a mut R,
}

impl<'a, M, D, S, R> Iterator for SamplerIter<'a, M, D, S, R>
where
    S: Sampler<M, D>,
    M: Model<D> + Clone + 'a,
    R: Rng,
{
    type Item = M;

    fn next(&mut self) -> Option<Self::Item> {
        let model = self.model.take().expect("Ought to have a value here");
        let model = self.sampler.step(model, self.data, self.rng);
        self.model = Some(model);
        // TODO: This is terribly slow... maybe look at streaming iterators.
        self.model.clone()
    }
}

fn take<T, F>(mut_ref: &mut T, closure: F)
where
    F: FnOnce(T) -> T,
{
    use std::ptr;

    unsafe {
        let old_t = ptr::read(mut_ref);
        let new_t = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| closure(old_t)))
            .unwrap_or_else(|_| ::std::process::abort());

        ptr::write(mut_ref, new_t);
    }
}

#[derive(Clone, Debug)]
pub struct McmcSamplerStep<S> {
    pub model: S,
    pub ln_score: f64,
}

#[derive(Clone, Debug, Default)]
pub struct GewekeTestOptions<F> {
    pub thinning: usize,
    pub n_samples: usize,
    pub burn_in: usize,
    pub min_p_value: f64,
    pub stat_map: F,
}

pub trait ResampleModel<D>: Model<D> {
    fn resample_data<R: Rng>(&mut self, data: Option<&D>, rng: &mut R) -> D;
}

pub trait PriorModel<D>: Model<D> {
    fn draw_from_prior<R: Rng>(rng: &mut R) -> Self;
}

pub trait GewekeTest<M, D>: Sampler<M, D>
where
    M: ResampleModel<D> + PriorModel<D> + Model<D> + Clone + std::fmt::Debug,
    D: std::fmt::Debug + Clone,
{
    /// Draw samples from the Marginal Conditional Simulator
    ///
    /// Samples are drawn by
    /// 1. Drawing from the prior
    /// 2. Resampling the data from the model.
    fn marginal_conditional_sampler<R: Rng>(
        &mut self,
        iterations: usize,
        rng: &mut R,
    ) -> Vec<(M, D)> {
        (0..iterations)
            .map(|_| {
                let mut model = M::draw_from_prior(rng);
                let data = model.resample_data(None, rng);
                (model, data)
            })
            .collect()
    }

    /// Draw samples from the Successive Conditional Simulator
    ///
    /// Samples are drawn by
    /// 1. Resampling the data
    /// 2. Stepping the model with the sampler under test
    fn successive_conditional_simulator<R: Rng>(
        &mut self,
        iterations: usize,
        rng: &mut R,
    ) -> Vec<(M, D)> {
        let mut init_model = M::draw_from_prior(rng);
        let init_data = init_model.resample_data(None, rng);
        (0..iterations)
            .scan((init_model, init_data), |(model, data), _i| {
                let new_data = model.resample_data(Some(data), rng);
                let new_model = self.step(model.clone(), &new_data, rng);
                *model = new_model.clone();
                *data = new_data.clone();
                Some((new_model, new_data))
            })
            .collect()
    }

    fn assert_geweke<R: Rng, F: Fn((M, D)) -> Vec<f64>>(
        mut self,
        options: GewekeTestOptions<F>,
        rng: &mut R,
    ) where
        Self: Sized,
    {
        let mcs = self
            .marginal_conditional_sampler(
                options.n_samples * options.thinning + options.burn_in,
                rng,
            )
            .into_iter()
            .skip(options.burn_in)
            .step_by(options.thinning)
            .map(&options.stat_map)
            .collect();

        let scs = self
            .successive_conditional_simulator(
                options.n_samples * options.thinning + options.burn_in,
                rng,
            )
            .into_iter()
            .skip(options.burn_in)
            .step_by(options.thinning)
            .map(&options.stat_map)
            .collect();

        let mcs = transpose2(mcs);
        let scs = transpose2(scs);

        mcs.into_iter().zip(scs).for_each(|(mc, sc)| {
            let (stat, alpha) = ks_two_sample(&mc, &sc, KsMode::Auto, KsAlternative::TwoSided)
                .expect("KS Two sample should be valid");

            if alpha < options.min_p_value {
                #[derive(Serialize)]
                struct Samples {
                    mcs: Vec<f64>,
                    scs: Vec<f64>,
                }

                let mut file = tempfile::NamedTempFile::new().expect("to be able to create a temp file");

                serde_json::to_writer(&mut file, &Samples { mcs: mc, scs: sc }).expect("to write to file");

                let (_, path) = file.keep().expect("to be able to make a persistent temp file");
                let path = path.display();

                panic!(
                    "KS alpha is lower than bound: {alpha:5.3} < {:5.3} (ks stat = {stat}) (Debug file: {path})",
                    options.min_p_value
                );
            }
        });
    }
}

impl<M, D, S> GewekeTest<M, D> for S
where
    M: ResampleModel<D> + PriorModel<D> + Clone + std::fmt::Debug,
    S: Sampler<M, D>,
    D: std::fmt::Debug + Clone,
{
}

fn transpose2<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v
        .into_iter()
        .map(std::iter::IntoIterator::into_iter)
        .collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| {
                    n.next()
                        .expect("the 'Matrix' formed by the double vec should be square")
                })
                .collect::<Vec<T>>()
        })
        .collect()
}
