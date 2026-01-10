use std::sync::Arc;

use rand::Rng;
use rv::{
    misc::LogSumExp,
    prelude::{Crp, DataOrSuffStat},
    traits::{ConjugatePrior, HasSuffStat, Rv, Sampleable, SuffStat},
};

#[derive(Clone)]
pub struct Partition<T, X>
where
    T: Clone,
{
    data: Vec<X>,
    assignments: Vec<Option<usize>>,
    component_data: Vec<T>,
    n_partitions: usize,
    counts: Vec<usize>,
    init: Arc<dyn Fn() -> T + 'static>,
    on_assign: fn(&mut T, &X),
    on_unassign: fn(&mut T, &X),
}

impl<T, X> Partition<T, X>
where
    T: Clone,
{
    /// Create a new empty `Partition`
    ///
    /// # Arguments
    /// * `init` - Function to create new, empty partition data.
    /// * `assign` - Function to run on a partition data when a value is assigned.
    /// * `unassign` - Function to run on a partition data when a value is removed.
    pub fn new<I: Fn() -> T + 'static>(
        init: I,
        assign: fn(&mut T, &X),
        unassign: fn(&mut T, &X),
    ) -> Self {
        Self {
            data: Vec::new(),
            assignments: Vec::new(),
            component_data: Vec::new(),
            n_partitions: 0,
            counts: Vec::new(),
            init: Arc::new(init),
            on_assign: assign,
            on_unassign: unassign,
        }
    }

    /// Append data to this partition
    pub fn push(&mut self, value: X) {
        self.data.push(value);
        self.assignments.push(None);
    }

    /// Append an iterator of values to this partition's data.
    pub fn append<I: IntoIterator<Item = X>>(&mut self, iter: I) {
        let mut new_data: Vec<X> = iter.into_iter().collect();
        self.assignments
            .append(&mut (0..new_data.len()).map(|_| None).collect::<Vec<_>>());
        self.data.append(&mut new_data);
    }

    /// Pop the last value out of the data of this partition
    pub fn pop(&mut self) -> Option<X> {
        self.assignments.pop();
        self.data.pop()
    }

    /// Insert an element into the partition's data.
    pub fn insert(&mut self, index: usize, element: X) {
        self.data.insert(index, element);
        self.assignments.insert(index, None);
    }

    /// Delete an element at index in the partition's data.
    pub fn remove(&mut self, index: usize) -> X {
        self.assignments.remove(index);
        self.data.remove(index)
    }

    /// Get a reference to the data in this partition.
    #[must_use]
    pub fn data(&self) -> &[X] {
        &self.data
    }

    /// Return the number data in this partition.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the partition is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[must_use]
    pub const fn n_partitions(&self) -> usize {
        self.n_partitions
    }

    #[must_use]
    pub fn partition_sizes(&self) -> &[usize] {
        &self.counts
    }

    /// Get a reference to the partition data in this partition.
    #[must_use]
    pub fn partition_data(&self) -> &[T] {
        &self.component_data
    }

    /// Get the assignment for an index.
    #[must_use]
    pub fn assignment(&self, index: usize) -> Option<usize> {
        self.assignments.get(index).copied().flatten()
    }

    #[must_use]
    pub fn assignments(&self) -> &[Option<usize>] {
        &self.assignments
    }

    /// Assign datum at `index` to partition `assignment`.
    pub fn assign(&mut self, index: usize, assignment: usize) {
        let x = &self.data[index];

        // forget data from the last assignment
        if let Some(old_assignment) = self.assignments.get(index).copied().flatten() {
            self.counts[old_assignment] -= 1;
            (self.on_unassign)(&mut self.component_data[old_assignment], x);
        }

        // Update the assignment
        self.assignments[index] = Some(assignment);

        // Ensure the assignment data and count data is present
        if assignment >= self.n_partitions {
            let to_add = assignment + 1 - self.n_partitions;
            self.component_data
                .extend((0..to_add).map(|_| (self.init)()));
            self.counts.extend((0..to_add).map(|_| 0));
            self.n_partitions += to_add;
        }

        // Update the partition info
        (self.on_assign)(&mut self.component_data[assignment], x);
        self.counts[assignment] += 1;
    }

    /// Unassign the value at index from its partition
    pub fn unassign(&mut self, index: usize) {
        if let Some(assignment) = self.assignments[index] {
            self.assignments[index] = None;
            (self.on_unassign)(&mut self.component_data[assignment], &self.data[index]);
            self.counts[assignment] -= 1;

            if self.counts[assignment] == 0 {
                self.remove_empty_partition(assignment);
            }
        }
    }

    /// Remove empty partitions and ensure partition indices start from 0 and increment by one.
    pub fn compact(&mut self) {
        #[allow(clippy::needless_collect)] // We need this collect to we can manipulate the
        // underlying partition...
        let to_simplify: Vec<usize> = self
            .counts
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (*s == 0).then_some(i))
            .rev()
            .collect();

        for i in to_simplify {
            self.remove_empty_partition(i);
        }
    }

    fn remove_empty_partition(&mut self, partition: usize) {
        debug_assert_eq!(self.counts[partition], 0);
        self.component_data.remove(partition);
        self.counts.remove(partition);
        self.n_partitions -= 1;

        self.assignments
            .iter_mut()
            .filter_map(|assn| {
                assn.as_mut()
                    .and_then(|assn| (*assn >= partition).then_some(assn))
            })
            .for_each(|i| *i -= 1);
    }

    /// Resample the partition from a `CRP`
    pub fn resample_crp<R: Rng>(&mut self, crp: &Crp, rng: &mut R) {
        let mut crp_draw: rv::data::Partition = crp.draw(rng);

        self.n_partitions = 0;
        self.counts = Vec::with_capacity(crp_draw.k());
        self.component_data = Vec::with_capacity(crp_draw.k());
        self.assignments = (0..self.data.len()).map(|_| None).collect();

        crp_draw
            .z_mut()
            .drain(..)
            .enumerate()
            .for_each(|(i, z)| self.assign(i, z));
    }
}

impl<X, T> Partition<T, X>
where
    T: SuffStat<X> + Clone,
{
    /// Create a new empty partition with empty `SuffStat` generator.
    pub fn new_stat<F: Fn() -> T + 'static>(f: F) -> Self {
        Self::new(
            f,
            rv::prelude::SuffStat::observe,
            rv::prelude::SuffStat::forget,
        )
    }

    /// Log marginal likelihood.
    pub fn ln_m<Fx, Pr>(&self, prior: &Pr) -> f64
    where
        Pr: ConjugatePrior<X, Fx>,
        Fx: Rv<X> + HasSuffStat<X, Stat = T>,
    {
        self.component_data
            .iter()
            .map(|stat| prior.ln_m(&rv::prelude::DataOrSuffStat::SuffStat(stat)))
            .sum()
    }

    /// Log Posterior Probability.
    #[allow(clippy::cast_precision_loss)]
    pub fn ln_pp<Fx, Pr>(&self, prior: &Pr, alpha: f64, x: &X) -> f64
    where
        Pr: ConjugatePrior<X, Fx>,
        Fx: Rv<X> + HasSuffStat<X, Stat = T>,
    {
        let ln_total_weight: f64 = ((self.len() as f64) + alpha).ln();
        let mut ln_weights: Vec<f64> = self
            .partition_sizes()
            .iter()
            .map(|s| (*s as f64).ln() - ln_total_weight)
            .collect();

        ln_weights.push(alpha.ln() - ln_total_weight);

        self.component_data
            .iter()
            .chain(std::iter::once(&(self.init)()))
            .zip(ln_weights)
            .map(|(p, ln_w)| ln_w + prior.ln_pp(x, &DataOrSuffStat::SuffStat(p)))
            .logsumexp()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rv::{
        data::{GaussianSuffStat, PoissonSuffStat},
        prelude::{Gaussian, NormalGamma},
        traits::{HasSuffStat, Sampleable},
    };

    use super::Partition;

    #[test]
    fn ln_pp() {
        let dist = Gaussian::standard();
        let empty_suffstat: GaussianSuffStat =
            <Gaussian as HasSuffStat<f64>>::empty_suffstat(&dist);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1234);
        let data: Vec<f64> = dist.sample(1000, &mut rng);
        let mut part: Partition<GaussianSuffStat, f64> =
            Partition::new_stat(move || empty_suffstat);
        part.append(data);

        for i in 0..1000 {
            part.assign(i, 0);
        }
        let prior = NormalGamma::new(0.0, 1.0, 1.0, 1.0).expect("given parameters to be valid");

        let a = -10.0;
        let b = 10.0;
        let n = 1000;
        let delta = (b - a) / f64::from(n);

        let mut sum = 0.0;
        for i in 0..n {
            let x = f64::from(i).mul_add(delta, a);
            sum += part.ln_pp(&prior, 1.0, &x).exp() * delta;
        }

        assert::close(sum, 1.0, 1E-4);
    }

    #[test]
    fn partition_assign_unassign() {
        let data = [1_usize, 2, 3];

        let mut part: Partition<PoissonSuffStat, usize> = Partition::new(
            PoissonSuffStat::new,
            rv::prelude::SuffStat::observe,
            rv::prelude::SuffStat::forget,
        );
        part.append(data);

        assert_eq!(part.assignment(0), None);

        part.assign(0, 0);
        part.assign(1, 1);
        part.assign(2, 1);
        assert_eq!(part.assignment(0), Some(0));
        assert_eq!(part.assignment(1), Some(1));
        assert_eq!(part.assignment(2), Some(1));
        assert_eq!(part.n_partitions, 2);

        part.unassign(0);
        assert_eq!(part.assignment(0), None);
        assert_eq!(part.assignment(1), Some(0));
        assert_eq!(part.assignment(2), Some(0));
        assert_eq!(part.n_partitions, 1);

        assert!(part
            .partition_sizes()
            .iter()
            .zip(part.component_data.iter())
            .all(|(s, d)| d.n() == *s));
    }

    #[test]
    fn removes_empty() {
        let data = [1_usize, 2, 3];

        let mut part: Partition<(), usize> = Partition::new(|| (), |_stat, _x| (), |_stat, _x| ());
        part.append(data);

        part.assign(0, 0);
        part.assign(1, 0);
        part.assign(2, 0);

        part.assign(0, 0);
        part.assign(1, 0);
        part.assign(2, 0);
    }
}
