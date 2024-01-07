use crate::models::Model;
use rand::Rng;
use rv::misc::{ks_two_sample, KsAlternative, KsMode};
use serde::Serialize;

pub mod samplers;

pub struct Link<M> {
    pub model: M,
    pub log_likelihood: f64,
    pub log_prior: f64,
}

/// Trait for Markov Chain Monte Carlo Samplers.
pub trait Sampler<M, D>: Sized
where
    M: Model<D>,
{
    /// Step the Sampler.
    fn step<R: Rng>(&mut self, model: M, data: &D, rng: &mut R) -> M;

    /// Warm-up the sampler
    fn multi_step<R: Rng>(&mut self, model: M, data: &D, steps: usize, rng: &mut R) -> M {
        (0..steps).fold(model, |acc, _| self.step(acc, data, rng))
    }

    fn iter_sample<'a, T, F: Fn(&M) -> T + 'a, R: Rng>(
        &mut self,
        model: M,
        data: &'a D,
        rng: &'a mut R,
        f: F,
    ) -> impl Iterator<Item = T> {
        (0..).scan(model, move |model, _| {
            take(model, |model| self.step(model, data, rng));

            Some(f(model))
        })
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

// Lensed Samplers
/*
/// Sampler for  value within a model, accessed via a lens and updated with the given sampler.
pub struct LensedSampler<S, D, OuterModel, InnerModel, L>
where
    L: Lens<OuterModel, InnerModel>,
    S: Sampler<InnerModel, D>,
    OuterModel: Model<D>,
    InnerModel: Model<D>,
{
    lens: L,
    inner: S,
    _phantom_outer_model: PhantomData<OuterModel>,
    _phantom_inner_model: PhantomData<InnerModel>,
    _phantom_d: PhantomData<D>,
}

impl<S, D, OuterModel, InnerModel, L> LensedSampler<S, D, OuterModel, InnerModel, L>
where
    L: Lens<OuterModel, InnerModel>,
    S: Sampler<InnerModel, D>,
    InnerModel: Model<D>,
    OuterModel: Model<D>,
{
    pub fn new(lens: L, inner: S) -> Self {
        Self {
            lens,
            inner,
            _phantom_outer_model: PhantomData,
            _phantom_inner_model: PhantomData,
            _phantom_d: PhantomData,
        }
    }
}

impl<S, D, OuterModel, InnerModel, L> Sampler<OuterModel, D>
    for LensedSampler<S, D, OuterModel, InnerModel, L>
where
    S: Sampler<InnerModel, D>,
    L: Lens<OuterModel, InnerModel>,
    InnerModel: Model<D> + Clone,
    OuterModel: Model<D> + Clone,
{
    fn step<R: Rng>(&mut self, model: OuterModel, data: &D, rng: &mut R) -> OuterModel {
        let x = self.lens.get(&model);
        let new_x = self.inner.step(x.clone(), data, rng);
        self.lens.set(model, new_x.clone())
    }
}

pub trait Proposer {
    type State;

    fn propose<R: Rng + ?Sized>(&mut self, state: &Self::State, rng: &mut R) -> Self::State;
}

pub struct MetropolisProposalSampler<S, P, F>
where
    S: Model,
    P: Proposer<State = S>,
    F: Fn(&S) -> f64,
{
    proposer: P,
    ln_score: F,
    current_state: S,
    current_ln_score: f64,
}

impl<S, P, F, R> McmcSampler<R> for MetropolisProposalSampler<S, P, F>
where
    S: Model,
    P: Proposer<State = S>,
    F: Fn(&S) -> f64,
    R: Rng + ?Sized,
{
    type Model = S;

    fn step(&mut self, rng: &mut R) {
        let proposed_state = self.proposer.propose(&self.current_state, rng);
        let proposed_ln_score = proposed_state.ln_score();
        let ln_alpha = proposed_ln_score - self.current_ln_score;

        if rng.gen::<f64>() < ln_alpha.exp() {
            self.current_state = proposed_state;
            self.current_ln_score = proposed_ln_score;
        }
    }
}

pub struct NestedSampler<M, D>
where
    M: Model<D>,
{
    samplers: Vec<Box<dyn Sampler<M, D>>>,
}

impl<M, D> NestedSampler<M, D>
where
    M: Model<D>,
{
    pub fn new<I: IntoIterator<Item = Box<dyn Sampler<M, D>>>>(samplers: I) -> Self {
        Self {
            samplers: samplers.into_iter().collect(),
        }
    }
}

impl<M, D> Sampler<M, D> for NestedSampler<M, D>
where
    M: Model<D>,
{
    fn step<R: Rng>(&mut self, model: M, data: &D, rng: &mut R) {
        self.samplers
            .iter_mut()
            .fold(model, |sampler, x| sampler.step(x, rng))
    }
}
*/

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
    fn marginal_conditional_sampler<R: Rng>(
        &mut self,
        iterations: usize,
        rng: &mut R,
    ) -> Vec<(M, D)> {
        //eprintln!("MCS!!!!!!!!!!!!!!\n\n");
        (0..iterations)
            .map(|_| {
                let mut model = M::draw_from_prior(rng);
                let data = model.resample_data(None, rng);
                (model, data)
            })
            .collect()
    }

    fn successive_conditional_simulator<R: Rng>(
        &mut self,
        iterations: usize,
        rng: &mut R,
    ) -> Vec<(M, D)> {
        let mut init_model = M::draw_from_prior(rng);
        let init_data = init_model.resample_data(None, rng);
        //eprintln!("__GEWEKE_SCS__");
        (0..iterations)
            .scan((init_model, init_data), |(model, data), _i| {
                //eprintln!("__GEWEKE_SCS__ step i = {i}");
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

                let mut file = tempfile::NamedTempFile::new().unwrap();

                serde_json::to_writer(&mut file, &Samples { mcs: mc, scs: sc }).unwrap();

                let (_, path) = file.keep().unwrap();
                let path = path.display();

                panic!(
                    "KS alpha is lower than bound: {alpha:5.3} < {:5.3} (ks stat = {stat}) (Debug file: {path})",
                    options.min_p_value
                );
            }
        })
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
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}
