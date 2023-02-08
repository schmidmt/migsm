use super::Model;

/// A model which support a partition based hierarchy.
pub trait PartitionModel<X, D>: Model<D>
where
    D: std::ops::Index<usize, Output = X>,
{
    /// Assign a datum at `idx` to the partition `partition_index`.
    fn assign(&mut self, idx: usize, partition_index: usize, data: &D);
    /// Unassign a datum at `idx`.
    fn unassign(&mut self, idx: usize, data: &D);
    /// The number of partitions.
    fn n_partitions(&self) -> usize;
    /// Log Posterior Predictive for t particular datum `x` in partition `partition_index`.
    fn ln_pp_partition(&self, x: &X, partition_index: usize) -> f64;
    /// Log Posterior Predictive for datum `x` to appear in an empty partition.
    fn ln_pp_empty(&self, x: &X) -> f64;

    /// Sized of each partition.
    fn counts(&self) -> &[usize];
    /// Partition assignments for each datum.
    fn assignments(&self) -> &[Option<usize>];

    /// The number of data.
    fn n_data(&self) -> usize {
        self.assignments().len()
    }
}
