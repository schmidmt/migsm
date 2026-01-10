use crate::mcmc::samplers::stick::StickBreaking;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub trait DirichletProcess {}

impl DirichletProcess for StickBreaking {}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, PartialOrd)]
pub struct Crp {
    at_table: Vec<usize>,
    counts: Vec<usize>,
    alpha: f64,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, PartialOrd)]
pub struct CrpRealization {
    seats: Vec<usize>,
    counts: Vec<usize>,
    alpha: f64,
}

impl CrpRealization {
    pub fn weights(&self) -> Vec<f64> {
        let total: f64 = (self.counts.iter().sum::<usize>() as f64) + self.alpha;
        self.counts.iter().map(|x| (*x as f64) / total).collect()
    }

    pub fn extend(&mut self)
}
