use rand::Rng;
use rv::prelude::{Beta, BetaError, Uniform};
use rv::traits::Rv;

use crate::mcmc::Sampler;
use crate::models::Model;

#[derive(Debug, Clone, Copy, Default)]
pub struct Slice;

impl<M, D> Sampler<M, D> for Slice
where
    M: Model<D>,
{
    fn step<R: rand::Rng>(&mut self, model: M, data: &D, rng: &mut R) -> M {
        todo!()
    }
}

struct StickBreakingProcess<'a, R>
where
    R: Rng,
{
    beta: Beta,
    remaining: f64,
    rng: &'a mut R,
}

impl<'a, R> StickBreakingProcess<'a, R>
where
    R: Rng,
{
    pub fn truncated(self, n: usize) -> Vec<f64> {
        self.take(n).collect()
    }
}

impl<'a, R: Rng> StickBreakingProcess<'a, R> {
    pub fn new(alpha: f64, rng: &'a mut R) -> Result<Self, BetaError> {
        Ok(Self {
            beta: Beta::new(alpha, 1.0)?,
            remaining: 1.0,
            rng,
        })
    }
}

impl<'a, R> Iterator for StickBreakingProcess<'a, R>
where
    R: Rng,
{
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let nu: f64 = self.beta.draw(self.rng);

        let beta = nu * self.remaining;
        self.remaining *= 1.0 - nu;
        Some(nu)
    }
}
