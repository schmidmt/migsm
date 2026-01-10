use rv::dist::UnitPowerLaw;

use crate::mcmc::Sampler;
use crate::mcmc::samplers::stick::StickBreaking;
use crate::models::Model;

#[derive(Debug, Clone, Copy, Default)]
pub struct Slice;

impl<M, D> Sampler<M, D> for Slice
where
    M: Model<D>,
{
    fn step<R: rand::Rng>(&mut self, _model: M, _data: &D, _rng: &mut R) -> M {
        let alpha = 10.0;
        let _stick = StickBreaking::new(UnitPowerLaw::new(alpha).unwrap());

        todo!()
    }
}
