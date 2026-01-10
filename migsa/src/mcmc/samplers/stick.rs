use itertools::{Either, EitherOrBoth, Itertools};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256Plus;
use rv::{
    data::DataOrSuffStat,
    dist::{Beta, BetaError, UnitPowerLaw, UnitPowerLawError},
    misc::{ConvergentSequence, sorted_uniforms},
    traits::{
        Cdf, ConjugatePrior, DiscreteDistr, Entropy, HasDensity, HasSuffStat, InverseCdf, Mode, Rv,
        Sampleable, SuffStat, Support,
    },
};
use serde::{Deserialize, Serialize};
use special::Beta as _;
use std::sync::{Arc, RwLock};

///! Shamelessly copied from the rv crate

// Represents a stick-breaking process.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct StickBreaking {
    break_prefix: Vec<Beta>,
    break_tail: UnitPowerLaw,
}

/// Implementation of the `StickBreaking` struct.
impl StickBreaking {
    /// Creates a new instance of `StickBreaking` with the given `breaker`.
    ///
    /// # Arguments
    /// * `breaker` - The `UnitPowerLaw` used for stick breaking.
    ///
    /// # Returns
    /// A new instance of `StickBreaking`.
    ///
    /// # Example
    /// ```
    /// use rv::prelude::*;
    /// use rv::experimental::stick_breaking_process::StickBreaking;
    ///
    /// let alpha = 5.0;
    /// let stick_breaking = StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());
    /// ```
    pub const fn new(breaker: UnitPowerLaw) -> Self {
        let break_prefix = Vec::new();
        Self {
            break_prefix,
            break_tail: breaker,
        }
    }

    pub const fn new_with_prefix(break_prefix: Vec<Beta>, break_tail: UnitPowerLaw) -> Self {
        Self {
            break_prefix,
            break_tail,
        }
    }

    pub fn from_alpha(alpha: f64) -> Result<Self, UnitPowerLawError> {
        let breaker = UnitPowerLaw::new(alpha)?;
        Ok(Self::new(breaker))
    }

    /// Sets the alpha parameter for both the `break_tail` and all Beta distributions in `break_prefix`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The new alpha value to set.
    ///
    /// # Returns
    ///
    /// A result indicating success or containing a `UnitPowerLawError` if setting alpha on `break_tail` fails,
    /// or a `BetaError` if setting alpha on any `Beta` distribution in `break_prefix` fails.
    pub fn set_alpha(&mut self, alpha: f64) -> Result<(), BetaError> {
        let old_alpha = self.alpha();
        self.break_tail.set_alpha(alpha).map_err(|e| match e {
            UnitPowerLawError::AlphaNotFinite { alpha } => BetaError::AlphaNotFinite { alpha },
            UnitPowerLawError::AlphaTooLow { alpha } => BetaError::AlphaTooLow { alpha },
        })?;
        let d_alpha = alpha - old_alpha;
        for b in &mut self.break_prefix {
            b.set_alpha(b.alpha() + d_alpha)?;
        }
        Ok(())
    }

    pub const fn break_prefix(&self) -> &Vec<Beta> {
        &self.break_prefix
    }

    pub const fn break_tail(&self) -> &UnitPowerLaw {
        &self.break_tail
    }

    pub fn break_dists(&self) -> impl Iterator<Item = Either<&Beta, &UnitPowerLaw>> {
        self.break_prefix
            .iter()
            .map(Either::Left)
            .chain(std::iter::repeat(Either::Right(&self.break_tail)))
    }

    pub fn alpha(&self) -> f64 {
        self.break_tail.alpha()
    }
}

pub struct PartialWeights(pub Vec<f64>);
pub struct BreakSequence(pub Vec<f64>);

impl From<&BreakSequence> for PartialWeights {
    fn from(bs: &BreakSequence) -> Self {
        let mut remaining = 1.0;
        let ws =
            bs.0.iter()
                .map(|b| {
                    debug_assert!((0.0..=1.0).contains(b));
                    let w = (1.0 - b) * remaining;
                    debug_assert!((0.0..=1.0).contains(&w));
                    remaining -= w;
                    debug_assert!((0.0..=1.0).contains(&remaining));
                    w
                })
                .collect();
        Self(ws)
    }
}

impl From<&PartialWeights> for BreakSequence {
    fn from(ws: &PartialWeights) -> Self {
        let mut r_new = 1.0;
        let mut r_old = 1.0;
        let mut b = f64::NAN;
        let bs: Vec<f64> =
            ws.0.iter()
                .map(|w| {
                    debug_assert!((0.0..=1.0).contains(w));
                    r_new = r_old - w;
                    debug_assert!((0.0..=1.0).contains(&r_new));
                    b = r_new / r_old;
                    debug_assert!((0.0..=1.0).contains(&b));
                    r_old = r_new;
                    b
                })
                .collect();
        assert!(
            (0.0..=1.0).contains(bs.last().unwrap()),
            "Weights cannot sum to more than one."
        );
        Self(bs)
    }
}

/// Implements the `HasDensity` trait for `StickBreaking`.
impl HasDensity<PartialWeights> for StickBreaking {
    /// Calculates the natural logarithm of the density function for the given input `x`.
    ///
    /// # Arguments
    /// * `x` - A reference to a slice of `f64` values.
    ///
    /// # Returns
    /// The natural logarithm of the density function.
    fn ln_f(&self, w: &PartialWeights) -> f64 {
        self.break_dists()
            .zip(BreakSequence::from(w).0.iter())
            .map(|(b, p)| match b {
                Either::Left(beta) => beta.ln_f(p),
                Either::Right(unit_powlaw) => unit_powlaw.ln_f(p),
            })
            .sum()
    }
}

impl Sampleable<StickSequence> for StickBreaking {
    /// Draws a sample from the `StickBreaking` distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    ///
    /// A `StickSequence` representing the drawn sample.
    fn draw<R: Rng>(&self, rng: &mut R) -> StickSequence {
        let seed: u64 = rng.random();

        let seq = StickSequence::new(self.break_tail.clone(), Some(seed));
        for beta in &self.break_prefix {
            let p = beta.draw(rng);
            seq.push_break(p);
        }
        seq
    }
}

fn rising_beta_prod(x: f64, a: usize, y: f64, b: usize) -> f64 {
    let x_y = x + y;
    let mut r = 1.0;
    for k in 0..a {
        let k = k as f64;
        r *= x + k;
        r /= x_y + k;
    }
    let x_y_a = x_y + a as f64;
    for k in 0..b {
        let k = k as f64;
        r *= y + k;
        r /= x_y_a + k;
    }
    r
}

/// Implements the `Sampleable` trait for `StickBreaking`.
impl Sampleable<StickBreakingDiscrete> for StickBreaking {
    /// Draws a sample from the `StickBreaking` distribution.
    ///
    /// # Arguments
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    /// A sample from the `StickBreaking` distribution.
    fn draw<R: Rng>(&self, rng: &mut R) -> StickBreakingDiscrete {
        StickBreakingDiscrete::new(self.draw(rng))
    }
}

/// Implementation of the `ConjugatePrior` trait for the `StickBreaking` struct.
impl ConjugatePrior<usize, StickBreakingDiscrete> for StickBreaking {
    type Posterior = Self;
    type MCache = ();
    type PpCache = Self::Posterior;

    fn empty_stat(&self) -> <StickBreakingDiscrete as HasSuffStat<usize>>::Stat {
        StickBreakingDiscreteSuffStat::new()
    }

    /// Computes the logarithm of the marginal likelihood cache.
    fn ln_m_cache(&self) -> Self::MCache {}

    /// Computes the logarithm of the predictive probability cache.
    fn ln_pp_cache(&self, x: &DataOrSuffStat<usize, StickBreakingDiscrete>) -> Self::PpCache {
        self.posterior(x)
    }

    /// Computes the posterior distribution from the sufficient statistic.
    fn posterior_from_suffstat(&self, stat: &StickBreakingDiscreteSuffStat) -> Self::Posterior {
        let pairs = stat.break_pairs();
        let new_prefix = self
            .break_prefix
            .iter()
            .zip_longest(pairs)
            .map(|pair| match pair {
                EitherOrBoth::Left(beta) => beta.clone(),
                EitherOrBoth::Right((a, b)) => {
                    Beta::new(self.break_tail.alpha() + a as f64, 1.0 + b as f64).unwrap()
                }
                EitherOrBoth::Both(beta, (a, b)) => {
                    Beta::new(beta.alpha() + a as f64, beta.beta() + b as f64).unwrap()
                }
            })
            .collect();
        Self {
            break_prefix: new_prefix,
            break_tail: self.break_tail.clone(),
        }
    }

    /// Computes the logarithm of the marginal likelihood.
    fn ln_m(&self, x: &DataOrSuffStat<usize, StickBreakingDiscrete>) -> f64 {
        let count_pairs = match x {
            DataOrSuffStat::Data(xs) => {
                let mut stat = StickBreakingDiscreteSuffStat::new();
                stat.observe_many(xs);
                stat.break_pairs()
            }
            DataOrSuffStat::SuffStat(stat) => stat.break_pairs(),
        };
        let alpha = self.break_tail.alpha();
        let params = self.break_prefix.iter().map(|b| (b.alpha(), b.beta()));
        count_pairs
            .iter()
            .zip_longest(params)
            .map(|pair| match pair {
                EitherOrBoth::Left((num_pass, num_fail)) => {
                    let (num_pass, num_fail) = (*num_pass as f64, *num_fail as f64);

                    // TODO: Simplify this after everything is working
                    (num_pass + alpha).ln_beta(num_fail + 1.0) - alpha.ln_beta(1.0)
                    // num_pass * alpha.ln() - (num_pass + num_fail) * (alpha + 1.0).ln()
                }
                EitherOrBoth::Right((_a, _b)) => 0.0,
                EitherOrBoth::Both((num_pass, num_fail), (a, b)) => {
                    // let (num_pass, num_fail) = (*num_pass as f64, *num_fail as f64);
                    // num_pass * a.ln() + num_fail * b.ln() - (num_pass + num_fail) * (a + b).ln()
                    // (num_pass + a).ln_beta(num_fail + b) - a.ln_beta(b)

                    // rising_pow(a, *num_pass).ln() + rising_pow(b, *num_fail).ln()
                    // - rising_pow(a + b, num_pass + num_fail).ln()

                    rising_beta_prod(a, *num_pass, b, *num_fail).ln()
                }
            })
            .sum()
    }

    /// Computes the logarithm of the marginal likelihood with cache.
    fn ln_m_with_cache(
        &self,
        _cache: &Self::MCache,
        x: &DataOrSuffStat<usize, StickBreakingDiscrete>,
    ) -> f64 {
        self.ln_m(x)
    }

    /// Computes the logarithm of the predictive probability with cache.
    fn ln_pp_with_cache(&self, cache: &Self::PpCache, y: &usize) -> f64 {
        cache.ln_m(&DataOrSuffStat::Data(&[*y]))
    }

    /// Computes the predictive probability.
    fn pp(&self, y: &usize, x: &DataOrSuffStat<usize, StickBreakingDiscrete>) -> f64 {
        let post = self.posterior(x);
        post.m(&DataOrSuffStat::Data(&[*y]))
    }
}

// We'd like to be able to serialize and deserialize StickSequence, but serde can't handle
// `Arc` or `RwLock`. So we use `StickSequenceFmt` as an intermediate type.
#[cfg(feature = "serde")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
struct StickSequenceFmt {
    breaker: UnitPowerLaw,
    inner: _Inner,
}

#[cfg(feature = "serde")]
impl From<StickSequenceFmt> for StickSequence {
    fn from(fmt: StickSequenceFmt) -> Self {
        Self {
            breaker: fmt.breaker,
            inner: Arc::new(RwLock::new(fmt.inner)),
        }
    }
}

#[cfg(feature = "serde")]
impl From<StickSequence> for StickSequenceFmt {
    fn from(sticks: StickSequence) -> Self {
        Self {
            breaker: sticks.breaker,
            inner: sticks.inner.read().map(|inner| inner.clone()).unwrap(),
        }
    }
}

// Add this function to provide a default RNG
#[allow(dead_code)]
fn default_rng() -> Xoshiro256Plus {
    Xoshiro256Plus::from_os_rng()
}

// NOTE: We currently derive PartialEq, but this (we think) compares the
// internal state of the RNGs, which is probably not what we want.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq)]
pub struct _Inner {
    #[cfg_attr(feature = "serde", serde(skip, default = "default_rng"))]
    rng: Xoshiro256Plus,
    ccdf: Vec<f64>,
}

impl _Inner {
    fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Xoshiro256Plus::from_os_rng, Xoshiro256Plus::seed_from_u64),
            ccdf: vec![1.0],
        }
    }

    #[must_use]
    pub fn ccdf(&self) -> &[f64] {
        &self.ccdf
    }

    fn extend<B: Rv<f64> + Clone>(&mut self, breaker: &B) -> f64 {
        let p: f64 = breaker.draw(&mut self.rng);
        let remaining_mass = self.ccdf.last().unwrap();
        let new_remaining_mass = remaining_mass * p;
        self.ccdf.push(new_remaining_mass);
        new_remaining_mass
    }

    fn extend_until<B, F>(&mut self, breaker: &B, p: F)
    where
        B: Rv<f64> + Clone,
        F: Fn(&Self) -> bool,
    {
        while !p(self) {
            self.extend(breaker);
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        rename_all = "snake_case",
        from = "StickSequenceFmt",
        into = "StickSequenceFmt"
    )
)]
#[derive(Clone, Debug)]
pub struct StickSequence {
    breaker: UnitPowerLaw,
    inner: Arc<RwLock<_Inner>>,
}

impl PartialEq<Self> for StickSequence {
    fn eq(&self, other: &Self) -> bool {
        self.ensure_breaks(other.num_weights_unstable());
        other.ensure_breaks(self.num_weights_unstable());
        self.breaker == other.breaker
            && self.with_inner(|self_inner| {
                other.with_inner(|other_inner| {
                    self_inner.ccdf == other_inner.ccdf && self_inner.rng == other_inner.rng
                })
            })
    }
}

impl StickSequence {
    /// Creates a new `StickSequence` with the given breaker and optional seed.
    ///
    /// # Arguments
    ///
    /// * `breaker` - A `UnitPowerLaw` instance used as the breaker.
    /// * `seed` - An optional seed for the random number generator.
    ///
    /// # Returns
    ///
    /// A new instance of `StickSequence`.
    pub fn new(breaker: UnitPowerLaw, seed: Option<u64>) -> Self {
        Self {
            breaker,
            inner: Arc::new(RwLock::new(_Inner::new(seed))),
        }
    }

    /// Pushes a new break to the stick sequence using a given probability `p`.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability used to calculate the new remaining mass.
    pub fn push_break(&self, p: f64) {
        self.with_inner_mut(|inner| {
            let remaining_mass = *inner.ccdf.last().unwrap();
            let new_remaining_mass = remaining_mass * p;
            inner.ccdf.push(new_remaining_mass);
        });
    }

    /// Pushes a new value `p` directly to the ccdf vector if `p` is less than the last element.
    ///
    /// # Arguments
    ///
    /// * `p` - The value to be pushed to the ccdf vector.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not less than the last element of the ccdf vector.
    pub fn push_to_ccdf(&self, p: f64) {
        self.with_inner_mut(|inner| {
            assert!(p < *inner.ccdf.last().unwrap());
            inner.ccdf.push(p);
        });
    }

    /// Extends the ccdf vector until a condition defined by `pred` is met, then applies function `f`.
    ///
    /// # Type Parameters
    ///
    /// * `P` - A predicate function type that takes a reference to a vector of f64 and returns a bool.
    /// * `F` - A function type that takes a reference to a vector of f64 and returns a value of type `Ans`.
    /// * `Ans` - The return type of the function `f`.
    ///
    /// # Arguments
    ///
    /// * `pred` - A predicate function that determines when to stop extending the ccdf vector.
    /// * `f` - A function to apply to the ccdf vector once the condition is met.
    ///
    /// # Returns
    ///
    /// The result of applying function `f` to the ccdf vector.
    pub fn extendmap_ccdf<P, F, Ans>(&self, pred: P, f: F) -> Ans
    where
        P: Fn(&Vec<f64>) -> bool,
        F: Fn(&Vec<f64>) -> Ans,
    {
        self.extend_until(|inner| pred(&inner.ccdf));
        self.with_inner(|inner| f(&inner.ccdf))
    }

    /// Provides read access to the inner `_Inner` structure.
    ///
    /// # Type Parameters
    ///
    /// * `F` - A function type that takes a reference to `_Inner` and returns a value of type `Ans`.
    /// * `Ans` - The return type of the function `f`.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that is applied to the inner `_Inner` structure.
    ///
    /// # Returns
    ///
    /// The result of applying function `f` to the inner `_Inner` structure.
    pub fn with_inner<F, Ans>(&self, f: F) -> Ans
    where
        F: FnOnce(&_Inner) -> Ans,
    {
        self.inner.read().map(|inner| f(&inner)).unwrap()
    }

    /// Provides write access to the inner `_Inner` structure.
    ///
    /// # Type Parameters
    ///
    /// * `F` - A function type that takes a mutable reference to `_Inner` and returns a value of type `Ans`.
    /// * `Ans` - The return type of the function `f`.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that is applied to the inner `_Inner` structure.
    ///
    /// # Returns
    ///
    /// The result of applying function `f` to the inner `_Inner` structure.
    pub fn with_inner_mut<F, Ans>(&self, f: F) -> Ans
    where
        F: FnOnce(&mut _Inner) -> Ans,
    {
        self.inner.write().map(|mut inner| f(&mut inner)).unwrap()
    }

    /// Ensures that the ccdf vector is extended to at least `n + 1` elements.
    ///
    /// # Arguments
    ///
    /// * `n` - The minimum number of elements the ccdf vector should have.
    pub fn ensure_breaks(&self, n: usize) {
        self.extend_until(|inner| inner.ccdf.len() > n);
    }

    /// Returns the `n`th element of the ccdf vector, ensuring the vector is long enough.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the element to retrieve from the ccdf vector.
    ///
    /// # Returns
    ///
    /// The `n`th element of the ccdf vector.
    pub fn ccdf(&self, n: usize) -> f64 {
        self.ensure_breaks(n);
        self.with_inner(|inner| {
            let ccdf = &inner.ccdf;
            ccdf[n]
        })
    }

    /// Returns the number of weights instantiated so far.
    ///
    /// # Returns
    ///
    /// The number of weights. This is "unstable" because it's a detail of the
    /// implementation that should not be depended on.
    pub fn num_weights_unstable(&self) -> usize {
        self.with_inner(|inner| inner.ccdf.len() - 1)
    }

    /// Returns the weight of the `n`th stick.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the stick whose weight is to be returned.
    ///
    /// # Returns
    ///
    /// The weight of the `n`th stick.
    pub fn weight(&self, n: usize) -> f64 {
        self.ensure_breaks(n + 1);
        self.with_inner(|inner| {
            let ccdf = &inner.ccdf;
            ccdf[n] - ccdf[n + 1]
        })
    }

    /// Returns the weights of the first `n` sticks.
    ///
    /// Note that this includes sticks `0..n-1`, but not `n`.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of sticks for which to return the weights.
    ///
    /// # Returns
    ///
    /// A `PartialWeights` instance containing the weights of the first `n` sticks.
    pub fn weights(&self, n: usize) -> PartialWeights {
        self.ensure_breaks(n);
        let w = self.with_inner(|inner| {
            let mut last_p = 1.0;
            inner
                .ccdf
                .iter()
                .skip(1)
                .map(|&p| {
                    let w = last_p - p;
                    last_p = p;
                    w
                })
                .collect()
        });
        PartialWeights(w)
    }

    /// Returns a clone of the breaker used in this `StickSequence`.
    ///
    /// # Returns
    ///
    /// A clone of the `UnitPowerLaw` instance used as the breaker.
    pub fn breaker(&self) -> UnitPowerLaw {
        self.breaker.clone()
    }

    /// Extends the ccdf vector until a condition defined by `p` is met.
    ///
    /// # Type Parameters
    ///
    /// * `F` - A function type that takes a reference to `_Inner` and returns a bool.
    ///
    /// # Arguments
    ///
    /// * `p` - A predicate function that determines when to stop extending the ccdf vector.
    pub fn extend_until<F>(&self, p: F)
    where
        F: Fn(&_Inner) -> bool,
    {
        self.with_inner_mut(|inner| inner.extend_until(&self.breaker, p));
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
/// A "Stick-breaking discrete" distribution parameterized by a `StickSequence`.
pub struct StickBreakingDiscrete {
    sticks: StickSequence,
}

impl StickBreakingDiscrete {
    /// Creates a new instance of `StickBreakingDiscrete` with the specified `StickSequence`.
    ///
    /// # Arguments
    ///
    /// * `sticks` - The `StickSequence` used for generating random numbers.
    ///
    /// # Returns
    ///
    /// A new instance of `StickBreakingDiscrete`.
    pub const fn new(sticks: StickSequence) -> Self {
        Self { sticks }
    }

    /// Calculates the inverse complementary cumulative distribution function
    /// (invccdf) for the `StickBreakingDiscrete` distribution. This method is preferred over the
    /// traditional cumulative distribution function (cdf) as it provides higher precision in the
    /// tail regions of the distribution.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability value for which to calculate the invccdf.
    ///
    /// # Returns
    ///
    /// The index of the first element in the `StickSequence` whose cumulative probability is less
    /// than `p`.
    pub fn invccdf(&self, p: f64) -> usize {
        debug_assert!(p > 0.0 && p < 1.0);
        self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &p,
            |ccdf| ccdf.iter().position(|q| *q < p).unwrap() - 1,
        )
    }

    /// Provides a reference to the `StickSequence` used by the `StickBreakingDiscrete` distribution.
    ///
    /// # Returns
    ///
    /// A reference to the `StickSequence`.
    pub const fn stick_sequence(&self) -> &StickSequence {
        &self.sticks
    }

    /// Calculates the inverse complementary cumulative distribution function (invccdf) for
    /// multiple sorted values. This method is useful for efficiently computing the invccdf for a
    /// sequence of values that are already sorted in ascending order. The returned vector contains
    /// the indices of the `StickSequence` elements whose cumulative probabilities are less than the
    /// corresponding values in `ps`.
    ///
    /// # Arguments
    ///
    /// * `ps` - A slice of probability values for which to calculate the invccdf. The values must
    ///   be sorted in ascending order.
    ///
    /// # Returns
    ///
    /// A vector containing the indices of the `StickSequence` elements whose cumulative probabilities
    /// are less than the corresponding values in `ps`.
    pub fn multi_invccdf_sorted(&self, ps: &[f64]) -> Vec<usize> {
        let n = ps.len();
        self.sticks.extendmap_ccdf(
            // Note that ccdf is decreasing, but ps is increasing
            |ccdf| ccdf.last().unwrap() < ps.first().unwrap(),
            |ccdf| {
                let mut result: Vec<usize> = Vec::with_capacity(n);

                // Start at the end of the sorted probability values (the largest value)
                let mut i: usize = n - 1;
                for q in ccdf.iter().skip(1).enumerate() {
                    while ps[i] > *q.1 {
                        result.push(q.0);
                        if i == 0 {
                            break;
                        }
                        i -= 1;
                    }
                }
                result
            },
        )
    }
}

/// Implementation of the `Support` trait for `StickBreakingDiscrete`.
impl Support<usize> for StickBreakingDiscrete {
    /// Checks if the given value is supported by `StickBreakingDiscrete`.
    ///
    /// # Arguments
    ///
    /// * `x` - The value to be checked.
    ///
    /// # Returns
    ///
    /// Returns `true` for all values as `StickBreakingDiscrete` supports all `usize` values, `false` otherwise.
    fn supports(&self, _: &usize) -> bool {
        true
    }
}

/// Implementation of the `Cdf` trait for `StickBreakingDiscrete`.
impl Cdf<usize> for StickBreakingDiscrete {
    /// Calculates the survival function (SF) for a given value `x`.
    ///
    /// The survival function is defined as 1 minus the cumulative distribution function (CDF).
    /// It represents the probability that a random variable is greater than `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - The value for which to calculate the survival function.
    ///
    /// # Returns
    ///
    /// The calculated survival function value as a `f64`.
    fn sf(&self, x: &usize) -> f64 {
        self.sticks.ccdf(*x + 1)
    }

    /// Calculates the cumulative distribution function (CDF) for a given value `x`.
    ///
    /// The cumulative distribution function (CDF) represents the probability that a random variable
    /// is less than or equal to `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - The value for which to calculate the cumulative distribution function.
    ///
    /// # Returns
    ///
    /// The calculated cumulative distribution function value as a `f64`.
    fn cdf(&self, x: &usize) -> f64 {
        1.0 - self.sf(x)
    }
}

impl InverseCdf<usize> for StickBreakingDiscrete {
    /// Calculates the inverse cumulative distribution function (invcdf) for a given probability `p`.
    ///
    /// The inverse cumulative distribution function (invcdf) represents the value below which a random variable
    /// falls with probability `p`.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability value for which to calculate the invcdf.
    ///
    /// # Returns
    ///
    /// The calculated invcdf value as a `usize`.
    fn invcdf(&self, p: f64) -> usize {
        self.invccdf(1.0 - p)
    }
}

impl DiscreteDistr<usize> for StickBreakingDiscrete {}

impl Mode<usize> for StickBreakingDiscrete {
    /// Calculates the mode of the `StickBreakingDiscrete` distribution.
    ///
    /// The mode is the value that appears most frequently in a data set or probability distribution.
    ///
    /// # Returns
    ///
    /// The mode of the distribution as an `Option<usize>`. Returns `None` if the mode cannot be determined.
    fn mode(&self) -> Option<usize> {
        let w0 = self.sticks.weight(0);
        // Once the unallocated mass is less than that of the first stick, the
        // allocated mass is guaranteed to contain the mode.
        self.sticks.extendmap_ccdf(
            |ccdf| ccdf.last().unwrap() < &w0,
            |ccdf| {
                let weights: Vec<f64> = ccdf.windows(2).map(|qs| qs[0] - qs[1]).collect();
                weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
            },
        )
    }
}

/// Provides density and log-density functions for `StickBreakingDiscrete`.
impl HasDensity<usize> for StickBreakingDiscrete {
    /// Computes the density of a given stick index.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the stick.
    ///
    /// # Returns
    ///
    /// The density of the stick at index `n`.
    fn f(&self, n: &usize) -> f64 {
        let sticks = &self.sticks;
        sticks.weight(*n)
    }

    /// Computes the natural logarithm of the density of a given stick index.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the stick.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the density of the stick at index `n`.
    fn ln_f(&self, n: &usize) -> f64 {
        self.f(n).ln()
    }
}

/// Enables sampling from `StickBreakingDiscrete`.
impl Sampleable<usize> for StickBreakingDiscrete {
    /// Draws a single sample from the distribution.
    ///
    /// # Type Parameters
    ///
    /// * `R` - The random number generator type.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    ///
    /// A single sample as a usize.
    fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        let u: f64 = rng.random();
        self.invccdf(u)
    }

    /// Draws multiple samples from the distribution and shuffles them.
    ///
    /// # Type Parameters
    ///
    /// * `R` - The random number generator type.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of samples to draw.
    /// * `rng` - A mutable reference to the random number generator.
    ///
    /// # Returns
    ///
    /// A vector of usize samples, shuffled.
    fn sample<R: Rng>(&self, n: usize, mut rng: &mut R) -> Vec<usize> {
        let ps = sorted_uniforms(n, &mut rng);
        let mut result = self.multi_invccdf_sorted(&ps);

        // At this point `result` is sorted, so we need to shuffle it.
        // Note that shuffling is O(n) but sorting is O(n log n)
        result.shuffle(&mut rng);
        result
    }
}

impl Entropy for StickBreakingDiscrete {
    fn entropy(&self) -> f64 {
        let probs = (0..).map(|n| self.f(&n));
        probs
            .map(|p| p * p.ln())
            .scan(0.0, |state, x| {
                *state -= x;
                Some(*state)
            })
            .limit(1e-10)
    }
}

/*
impl Entropy for &Mixture<StickBreakingDiscrete> {
    fn entropy(&self) -> f64 {
        let probs = (0..).map(|n| self.f(&n));
        probs
            .map(|p| p * p.ln())
            .scan(0.0, |state, x| {
                *state -= x;
                Some(*state)
            })
            .limit(1e-10)
    }
}
*/

/// Represents the sufficient statistics for a Stick-Breaking Discrete distribution.
///
/// This struct encapsulates the sufficient statistics for a Stick-Breaking Discrete distribution,
/// primarily involving a vector of counts representing the observed data.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StickBreakingDiscreteSuffStat {
    /// A vector of counts for observed data.
    ///
    /// Each element represents the count of observations for a given category.
    counts: Vec<usize>,
}

impl StickBreakingDiscreteSuffStat {
    /// Constructs a new instance.
    ///
    /// Initializes a new `StickBreakingDiscreteSuffStat` with an empty vector of counts.
    ///
    /// # Returns
    ///
    /// A new `StickBreakingDiscreteSuffStat` instance.
    #[must_use]
    pub const fn new() -> Self {
        Self { counts: Vec::new() }
    }

    #[must_use]
    pub const fn from_counts(counts: Vec<usize>) -> Self {
        Self { counts }
    }

    /// Calculates break pairs for probabilities.
    ///
    /// Returns a vector of pairs where each pair consists of the sum of all counts after the current index and the count at the current index.
    ///
    /// # Returns
    ///
    /// A vector of `(usize, usize)` pairs for calculating probabilities.
    #[must_use]
    pub fn break_pairs(&self) -> Vec<(usize, usize)> {
        let mut s = self.counts.iter().sum();
        self.counts
            .iter()
            .map(|&x| {
                s -= x;
                (s, x)
            })
            .collect()
    }

    /// Provides read-only access to counts.
    ///
    /// # Returns
    ///
    /// A reference to the vector of counts.
    #[must_use]
    pub const fn counts(&self) -> &Vec<usize> {
        &self.counts
    }
}

impl From<&[usize]> for StickBreakingDiscreteSuffStat {
    /// Constructs from a slice of counts.
    ///
    /// Allows creation from a slice of counts, converting raw observation data into a sufficient statistic.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of counts.
    ///
    /// # Returns
    ///
    /// A new `StickBreakingDiscreteSuffStat` instance.
    fn from(data: &[usize]) -> Self {
        let mut stat = Self::new();
        stat.observe_many(data);
        stat
    }
}

impl Default for StickBreakingDiscreteSuffStat {
    /// Returns a default instance.
    ///
    /// Equivalent to `new()`, for APIs requiring a default constructor.
    ///
    /// # Returns
    ///
    /// A default `StickBreakingDiscreteSuffStat` instance.
    fn default() -> Self {
        Self::new()
    }
}

impl HasSuffStat<usize> for StickBreakingDiscrete {
    type Stat = StickBreakingDiscreteSuffStat;

    /// Initializes an empty sufficient statistic.
    ///
    /// # Returns
    ///
    /// An empty `StickBreakingDiscreteSuffStat`.
    fn empty_suffstat(&self) -> Self::Stat {
        Self::Stat::new()
    }

    /// Calculates the log probability density of observed data.
    ///
    /// # Arguments
    /// * `stat` - A reference to the sufficient statistic.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the probability of the observed data.
    fn ln_f_stat(&self, stat: &Self::Stat) -> f64 {
        self.stick_sequence()
            .weights(stat.counts.len())
            .0
            .iter()
            .zip(stat.counts.iter())
            .map(|(w, c)| (*c as f64) * w.ln())
            .sum()
    }
}

impl SuffStat<usize> for StickBreakingDiscreteSuffStat {
    /// Returns the total count of observations.
    ///
    /// # Returns
    ///
    /// The total count of all observed data.
    fn n(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Updates the statistic with a new observation.
    ///
    /// # Arguments
    ///
    /// * `i` - The index at which to increment the count.
    fn observe(&mut self, i: &usize) {
        if self.counts.len() < *i + 1 {
            self.counts.resize(*i + 1, 0);
        }
        self.counts[*i] += 1;
    }

    /// Removes a previously observed data point.
    ///
    /// # Arguments
    ///
    /// * `i` - The index at which to decrement the count.
    ///
    /// # Panics
    ///
    /// Panics if there are no observations of the specified category to forget.
    fn forget(&mut self, i: &usize) {
        assert!(self.counts[*i] > 0, "No observations of {i} to forget.");
        self.counts[*i] -= 1;
    }

    fn merge(&mut self, other: Self) {
        if other.counts.len() > self.counts.len() {
            self.counts.resize(other.counts.len(), 0);
        }
        self.counts
            .iter_mut()
            .zip(other.counts.iter())
            .for_each(|(ct_a, &ct_b)| *ct_a += ct_b);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::rng;
    use rv::{dist::ChiSquared, misc::LogSumExp, prelude::Bernoulli, prelude::HasDensity};

    use super::*;

    proptest! {
        #[test]
        fn partial_weights_to_break_sequence(v in prop::collection::vec(0.0..=1.0, 1..100), m in 0.0..=1.0) {
            // we want the sum of ws to be in the range [0, 1]
            let multiplier: f64 = m / v.iter().sum::<f64>();
            let ws = PartialWeights(v.iter().map(|w| w * multiplier).collect());
            let bs = BreakSequence::from(&ws);
            assert::close(ws.0, PartialWeights::from(&bs).0, 1e-10);
        }
    }

    proptest! {
        #[test]
        fn break_sequence_to_partial_weights(v in prop::collection::vec(0.0..=1.0, 1..100)) {
            let bs = BreakSequence(v);
            let ws = PartialWeights::from(&bs);
            let bs2 = BreakSequence::from(&ws);
            assert::close(bs.0, bs2.0, 1e-10);
        }
    }

    #[test]
    fn sb_ln_m_vs_monte_carlo() {
        let n_samples = 1_000_000;
        let xs: Vec<usize> = vec![1, 2, 3];

        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let obs = DataOrSuffStat::Data(&xs);
        let ln_m = sb.ln_m(&obs);

        let mc_est = {
            sb.sample_stream(&mut rand::rng())
                .take(n_samples)
                .map(|sbd: StickBreakingDiscrete| xs.iter().map(|x| sbd.ln_f(x)).sum::<f64>())
                .logsumexp()
                - (n_samples as f64).ln()
        };
        // high error tolerance. MC estimation is not the most accurate...
        assert::close(ln_m, mc_est, 1e-2);
    }

    #[test]
    fn sb_pp_posterior() {
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&3, &DataOrSuffStat::Data(&[1, 2]));
        let post = sb.posterior(&DataOrSuffStat::Data(&[1, 2]));
        let post_f = post.pp(
            &3,
            &DataOrSuffStat::SuffStat(&StickBreakingDiscreteSuffStat::new()),
        );
        assert::close(sb_pp, post_f, 1e-10);
    }

    #[test]
    fn sb_repeated_obs_more_likely() {
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_m = sb.ln_m(&DataOrSuffStat::Data(&[10]));
        let post = sb.posterior(&DataOrSuffStat::Data(&[10]));
        let post_m = post.ln_m(&DataOrSuffStat::Data(&[10]));
        assert!(post_m > sb_m);
    }

    #[test]
    fn sb_bayes_law() {
        let mut rng = rand::rng();

        // Prior
        let prior = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let par: StickSequence = prior.draw(&mut rng);
        let par_data = par.weights(7);
        let prior_lnf = prior.ln_f(&par_data);

        // Likelihood
        let lik = StickBreakingDiscrete::new(par);
        let lik_data: &usize = &5;
        let lik_lnf = lik.ln_f(lik_data);

        // Evidence
        let ln_ev = prior.ln_m(&DataOrSuffStat::Data(&[*lik_data]));

        // Posterior
        let post = prior.posterior(&DataOrSuffStat::Data(&[*lik_data]));
        let post_lnf = post.ln_f(&par_data);

        // Bayes' law
        assert::close(post_lnf, prior_lnf + lik_lnf - ln_ev, 1e-12);
    }

    #[test]
    fn sb_pp_is_quotient_of_marginals() {
        // pp(x|y) = m({x, y})/m(x)
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&1, &DataOrSuffStat::Data(&[0]));

        let m_1 = sb.m(&DataOrSuffStat::Data(&[0]));
        let m_1_2 = sb.m(&DataOrSuffStat::Data(&[0, 1]));

        assert::close(sb_pp, m_1_2 / m_1, 1e-12);
    }

    #[test]
    fn sb_big_alpha_heavy_tails() {
        let sb_5 = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_2 = StickBreaking::new(UnitPowerLaw::new(2.0).unwrap());
        let sb_pt5 = StickBreaking::new(UnitPowerLaw::new(0.5).unwrap());

        let m_pt5_10 = sb_pt5.m(&DataOrSuffStat::Data(&[10]));
        let m_2_10 = sb_2.m(&DataOrSuffStat::Data(&[10]));
        let m_5_10 = sb_5.m(&DataOrSuffStat::Data(&[10]));

        assert!(m_pt5_10 < m_2_10);
        assert!(m_2_10 < m_5_10);
    }

    #[test]
    fn sb_marginal_zero() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let m_0 = sb.m(&DataOrSuffStat::Data(&[0]));
        let bern = Bernoulli::new(3.0 / 4.0).unwrap();
        assert::close(m_0, bern.f(&0), 1e-12);
    }

    #[test]
    fn sb_postpred_zero() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let pp_0 = sb.pp(&0, &DataOrSuffStat::Data(&[0]));
        let bern = Bernoulli::new(3.0 / 5.0).unwrap();
        assert::close(pp_0, bern.f(&0), 1e-12);
    }

    #[test]
    fn sb_pp_zero_marginals() {
        // pp(x|y) = m({x, y})/m(x)
        let sb = StickBreaking::new(UnitPowerLaw::new(5.0).unwrap());
        let sb_pp = sb.pp(&0, &DataOrSuffStat::Data(&[0]));

        let m_1 = sb.m(&DataOrSuffStat::Data(&[0]));
        let m_1_2 = sb.m(&DataOrSuffStat::Data(&[0, 0]));

        assert::close(sb_pp, m_1_2 / m_1, 1e-12);
    }

    #[test]
    fn sb_posterior_obs_one() {
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let post = sb.posterior(&DataOrSuffStat::Data(&[2]));

        assert_eq!(post.break_prefix[0], Beta::new(4.0, 1.0).unwrap());
        assert_eq!(post.break_prefix[1], Beta::new(4.0, 1.0).unwrap());
        assert_eq!(post.break_prefix[2], Beta::new(3.0, 2.0).unwrap());
    }

    #[test]
    fn sb_logposterior_diff() {
        // Like Bayes Law, but takes a quotient to cancel evidence

        let mut rng = rand::rng();
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());
        let seq1: StickSequence = sb.draw(&mut rng);
        let seq2: StickSequence = sb.draw(&mut rng);

        let w1 = seq1.weights(3);
        let w2 = seq2.weights(3);

        let logprior_diff = sb.ln_f(&w1) - sb.ln_f(&w2);

        let data = [1, 2];
        let stat = StickBreakingDiscreteSuffStat::from(&data[..]);
        let post = sb.posterior(&DataOrSuffStat::SuffStat(&stat));
        let logpost_diff = post.ln_f(&w1) - post.ln_f(&w2);

        let sbd1 = StickBreakingDiscrete::new(seq1);
        let sbd2 = StickBreakingDiscrete::new(seq2);
        let loglik_diff = sbd1.ln_f_stat(&stat) - sbd2.ln_f_stat(&stat);

        assert::close(logpost_diff, loglik_diff + logprior_diff, 1e-12);
    }

    #[test]
    fn sb_posterior_rejection_sampling() {
        let mut rng = rand::rng();
        let sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());

        let num_samples = 1000;

        // Our computed posterior
        let data = [10];
        let post = sb.posterior(&DataOrSuffStat::Data(&data[..]));

        // An approximation using rejection sampling
        let mut stat = StickBreakingDiscreteSuffStat::new();
        let mut n = 0;
        while n < num_samples {
            let seq: StickSequence = sb.draw(&mut rng);
            let sbd = StickBreakingDiscrete::new(seq.clone());
            if sbd.draw(&mut rng) == 10 {
                stat.observe(&sbd.draw(&mut rng));
                n += 1;
            }
        }

        let counts = stat.counts();

        // This would be counts.len() - 1, but the current implementation has a
        // trailing zero we need to ignore
        let dof = (counts.len() - 2) as f64;

        // Chi-square test is not exact, so we'll trim to only consider cases
        // where expected count is at least 5.
        let expected_counts = (0..)
            .map(|j| post.m(&DataOrSuffStat::Data(&[j])) * f64::from(num_samples))
            .take_while(|x| *x > 5.0);

        let ts = counts
            .iter()
            .zip(expected_counts)
            .map(|(o, e)| ((*o as f64) - e).powi(2) / e);

        let t: &f64 = &ts.clone().sum();
        let p = ChiSquared::new(dof).unwrap().sf(t);

        assert!(p > 0.001, "p-value = {p}");
    }

    #[test]
    fn test_set_alpha() {
        // Step 1: Generate a new StickBreaking instance with alpha=3
        let mut sb = StickBreaking::new(UnitPowerLaw::new(3.0).unwrap());

        // Step 2: Set the prefix to [Beta(4, 3), Beta(3, 2), Beta(2, 1)]
        sb.break_prefix = vec![
            Beta::new(4.0, 2.0).unwrap(),
            Beta::new(3.0, 2.0).unwrap(),
            Beta::new(2.0, 2.0).unwrap(),
        ];

        // Step 3: Call set_alpha(2.0)
        sb.set_alpha(2.0).unwrap();

        // Step 4: Check that the prefix is now [Beta(3, 3), Beta(2, 2), Beta(1, 1)]
        assert_eq!(sb.break_prefix[0], Beta::new(3.0, 2.0).unwrap());
        assert_eq!(sb.break_prefix[1], Beta::new(2.0, 2.0).unwrap());
        assert_eq!(sb.break_prefix[2], Beta::new(1.0, 2.0).unwrap());
        assert_eq!(sb.break_tail, UnitPowerLaw::new(2.0).unwrap());
    }

    #[test]
    fn test_stickseq_weights() {
        // test that `weights` gives the same as `weight` for all n
        let breaker = UnitPowerLaw::new(10.0).unwrap();
        let sticks = StickSequence::new(breaker, None);
        let weights = sticks.weights(100);
        assert_eq!(weights.0.len(), 100);
        for (n, w) in weights.0.iter().enumerate() {
            assert_eq!(sticks.weight(n), *w);
        }
    }

    #[test]
    fn test_push_to_ccdf() {
        let breaker = UnitPowerLaw::new(10.0).unwrap();
        let sticks = StickSequence::new(breaker, None);
        sticks.push_to_ccdf(0.9);
        sticks.push_to_ccdf(0.8);
        assert_eq!(sticks.ccdf(1), 0.9);
        assert_eq!(sticks.ccdf(2), 0.8);
    }

    #[test]
    fn test_push_break() {
        let breaker = UnitPowerLaw::new(10.0).unwrap();
        let sticks = StickSequence::new(breaker, None);
        sticks.push_break(0.9);
        sticks.push_break(0.8);
        assert::close(sticks.weights(2).0, vec![0.1, 0.18], 1e-10);
    }

    #[test]
    fn test_multi_invccdf_sorted() {
        let sticks = StickSequence::new(UnitPowerLaw::new(10.0).unwrap(), None);
        let sbd = StickBreakingDiscrete::new(sticks);
        let ps = sorted_uniforms(5, &mut rng());
        assert_eq!(
            sbd.multi_invccdf_sorted(&ps),
            ps.iter().rev().map(|p| sbd.invccdf(*p)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_break_pairs() {
        let suff_stat = StickBreakingDiscreteSuffStat {
            counts: vec![1, 2, 3],
        };

        let pairs = suff_stat.break_pairs();
        assert_eq!(pairs, vec![(5, 1), (3, 2), (0, 3)]);
    }

    // #[test]
    // fn test_ln_f_stat() {
    //     let sbd = StickBreakingDiscrete::new();
    //     let suff_stat = StickBreakingDiscreteSuffStat {
    //         counts: vec![1, 2, 3],
    //     };

    //     let ln_f_stat = sbd.ln_f_stat(&suff_stat);
    //     assert_eq!(ln_f_stat, 2.1972245773362196); // Replace with the expected value
    // }

    #[test]
    fn test_observe_and_forget() {
        let mut suff_stat = StickBreakingDiscreteSuffStat::new();

        suff_stat.observe(&1);
        suff_stat.observe(&2);
        suff_stat.observe(&2);
        suff_stat.forget(&2);

        assert_eq!(suff_stat.counts, vec![0, 1, 1]);
        assert_eq!(suff_stat.n(), 2);
    }

    #[test]
    fn test_new_is_default() {
        assert!(StickBreakingDiscreteSuffStat::new() == StickBreakingDiscreteSuffStat::default());
    }
}
