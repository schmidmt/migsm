/// A generic trait for models which have some scoring function.
pub trait Model<D> {
    /// Log posterior probability or any generic log score.
    ///
    /// Arguments
    /// =========
    ///
    /// * `data` - The data to which the model is being applied.
    fn ln_score(&self, data: &D) -> f64;
}

pub mod mixture;
pub mod partition;
