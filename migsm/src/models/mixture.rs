use std::cell::OnceCell;
use std::fmt::Debug;
use std::marker::PhantomData;

use rand::{Rng, SeedableRng};
use rv::misc::{logsumexp, pflip};
use rv::prelude::{Crp, DataOrSuffStat};
use rv::traits::{ConjugatePrior, HasSuffStat, Rv, SuffStat};

use crate::utils::NoPrettyPrint;

use super::partition::PartitionModel;
use super::Model;

#[derive(Clone)]
pub struct ConjugateMixtureModel<X, Fx, Pr>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    pub(crate) prior: Pr,
    pub(crate) crp: Crp,
    pub(crate) assignments: Vec<Option<usize>>,
    pub(crate) counts: Vec<usize>,
    pub(crate) partition_stats: Vec<Fx::Stat>,
    pub(crate) empty_stat: Fx::Stat,
    pub(crate) _phantom_x: PhantomData<X>,
    pub(crate) _phantom_fx: PhantomData<Fx>,
    pub(crate) component_weights: OnceCell<Vec<f64>>,
}

impl<X, Fx, Pr> std::fmt::Debug for ConjugateMixtureModel<X, Fx, Pr>
where
    X: Clone + Debug,
    Fx: Rv<X> + HasSuffStat<X> + Debug,
    Pr: ConjugatePrior<X, Fx> + Debug,
    Fx::Stat: Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let assignments: Vec<_> = self
            .assignments
            .iter()
            .map(|a| {
                a.as_ref()
                    .map(ToString::to_string)
                    .unwrap_or(String::from("-"))
            })
            .map(NoPrettyPrint::new)
            .collect();

        f.debug_struct("MixtureModel")
            .field("prior", &self.prior)
            .field("crp.alpha", &self.crp.alpha())
            .field("assignments", &NoPrettyPrint::new(assignments))
            .field("counts", &NoPrettyPrint::new(&self.counts))
            .field(
                "partition_stats",
                &NoPrettyPrint::new(&self.partition_stats),
            )
            .finish()
    }
}

impl<X, Fx, Pr> ConjugateMixtureModel<X, Fx, Pr>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
{
    pub fn new<'a, R, ID>(prior: Pr, data: ID, alpha: f64, rng: &mut R) -> Self
    where
        R: Rng,
        ID: ExactSizeIterator + Iterator<Item = &'a X>,
        X: 'a,
    {
        let crp = Crp::new(alpha, data.len()).unwrap();
        let fx = prior.draw(&mut rand::rngs::SmallRng::seed_from_u64(0x1234));
        let init_partition = crp.draw(rng);

        let mut partition_stats: Vec<Fx::Stat> = (0..(init_partition.k()))
            .map(|_| fx.empty_suffstat())
            .collect();

        let assignments: Vec<Option<usize>> =
            init_partition.z().iter().cloned().map(Some).collect();

        let mut counts: Vec<usize> = vec![0; init_partition.k()];
        init_partition.z().iter().zip(data).for_each(|(&asgn, x)| {
            counts[asgn] += 1;
            partition_stats[asgn].observe(x);
        });

        Self {
            prior,
            crp,
            partition_stats,
            empty_stat: fx.empty_suffstat(),
            assignments,
            counts,
            _phantom_x: PhantomData,
            _phantom_fx: PhantomData,
            component_weights: OnceCell::new(),
        }
    }

    pub fn with_assignment<'a, ID>(
        prior: Pr,
        data: ID,
        alpha: f64,
        assignments: &[Option<usize>],
    ) -> Self
    where
        ID: ExactSizeIterator + Iterator<Item = &'a X>,
        X: 'a,
    {
        assert!(assignments.len() == data.len());

        let crp = Crp::new(alpha, data.len()).unwrap();
        let fx = prior.draw(&mut rand::rngs::SmallRng::seed_from_u64(0x1234));

        let n_partitions: usize = assignments
            .iter()
            .cloned()
            .flatten()
            .map(|x| x + 1)
            .max()
            .unwrap_or(0);

        let mut partition_stats: Vec<Fx::Stat> =
            (0..n_partitions).map(|_| fx.empty_suffstat()).collect();

        let mut counts: Vec<usize> = vec![0; n_partitions];
        assignments.iter().zip(data).for_each(|(&asgn, x)| {
            if let Some(asgn) = asgn {
                counts[asgn] += 1;
                partition_stats[asgn].observe(x);
            }
        });

        Self {
            prior,
            crp,
            partition_stats,
            empty_stat: fx.empty_suffstat(),
            assignments: assignments.to_vec(),
            counts,
            _phantom_x: PhantomData,
            _phantom_fx: PhantomData,
            component_weights: OnceCell::new(),
        }
    }

    fn remove_partition(&mut self, partition_index: usize) {
        // remove stats, counts
        self.partition_stats.swap_remove(partition_index);
        self.counts.swap_remove(partition_index);

        // swap self.partition_stats.len() with assigned_to
        let swap_from = self.partition_stats.len();
        self.assignments.iter_mut().for_each(|assign| {
            if *assign == Some(swap_from) {
                *assign = Some(partition_index);
            }
        });

        // reset weight cache
        self.component_weights.take();
    }
}

impl<X, Fx, Pr> ConjugateMixtureModel<X, Fx, Pr>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
{
    /// Portion of the `log_score` from the inner distributions marginal probability.
    pub fn log_m(&self) -> f64 {
        self.partition_stats
            .iter()
            .map(|ps| self.prior.ln_m(&DataOrSuffStat::SuffStat(ps)))
            .sum()
    }

    /// Portion of `log_score` from the CRP prior
    pub fn crp_log_f(&self) -> f64 {
        let part = rv::data::Partition::new_unchecked(
            self.assignments.iter().flatten().cloned().collect(),
            self.counts.clone(),
        );
        self.crp.ln_f(&part)
    }

    fn component_weights(&self) -> &[f64] {
        self.component_weights.get_or_init(|| {
            let weights: Vec<f64> = self
                .counts
                .iter()
                .map(|x| *x as f64)
                .chain(std::iter::once(self.crp.alpha()))
                .collect();

            // Normalize the weights
            let weight_sum: f64 = weights.iter().sum();
            weights.into_iter().map(|w| w / weight_sum).collect()
        })
    }
}

impl<X, Fx, Pr> rv::traits::Rv<X> for ConjugateMixtureModel<X, Fx, Pr>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
{
    fn ln_f(&self, x: &X) -> f64 {
        // The ln poster predictive probabilities for each component
        let component_posterior_ln_ps: Vec<f64> = self
            .partition_stats
            .iter()
            .map(|stat| self.prior.ln_pp(x, &DataOrSuffStat::SuffStat(stat)))
            .chain(std::iter::once(
                self.prior
                    .ln_pp(x, &DataOrSuffStat::SuffStat(&self.empty_stat)),
            ))
            .collect();

        // TODO: Add empty component weight and empty posterior predictive

        // Component weights
        let weights: &[f64] = self.component_weights();

        let ln_ps: Vec<f64> = weights
            .into_iter()
            .zip(component_posterior_ln_ps.into_iter())
            .map(|(w, component_ln_pp)| w.ln() + component_ln_pp)
            .collect();

        logsumexp(&ln_ps)
    }

    fn draw<R: Rng>(&self, rng: &mut R) -> X {
        let weights: &[f64] = self.component_weights();
        let component_idx = pflip(&weights, 1, rng)[0];

        if component_idx < self.partition_stats.len() {
            // Draw from an existing component
            self.prior
                .posterior(&DataOrSuffStat::SuffStat(
                    &self.partition_stats[component_idx],
                ))
                .draw(rng)
                .draw(rng)
        } else {
            // Draw from a new component
            self.prior.draw(rng).draw(rng)
        }
    }
}

impl<X, Fx, Pr, D> Model<D> for ConjugateMixtureModel<X, Fx, Pr>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
    D: std::ops::Index<usize, Output = X>,
{
    fn ln_score(&self, _data: &D) -> f64 {
        self.crp_log_f() + self.log_m()
    }
}

impl<X, Fx, Pr, D> PartitionModel<X, D> for ConjugateMixtureModel<X, Fx, Pr>
where
    X: Clone + std::fmt::Debug,
    Fx: Rv<X> + HasSuffStat<X> + std::fmt::Debug,
    Pr: ConjugatePrior<X, Fx> + std::fmt::Debug,
    Fx::Stat: Clone + Debug + PartialEq,
    D: std::ops::Index<usize, Output = X>,
{
    fn assignments(&self) -> &[Option<usize>] {
        &self.assignments
    }

    fn assign(&mut self, idx: usize, partition_index: usize, data: &D) {
        if self.assignments[idx].is_some() {
            // Remove from the source part if this datum is assigned to one
            self.unassign(idx, data);
        }

        // Add additional partitions if the `partition_index` is larger
        // than the current number
        if partition_index >= self.partition_stats.len() {
            let to_add = (partition_index + 1).saturating_sub(self.partition_stats.len());
            (0..to_add).for_each(|_| {
                self.counts.push(0);
                self.partition_stats.push(self.empty_stat.clone());
            });
        }

        // Add to the stat and counts
        self.assignments[idx] = Some(partition_index);
        self.partition_stats[partition_index].observe(&data[idx]);
        self.counts[partition_index] += 1;

        // reset weight cache
        self.component_weights.take();
    }

    fn unassign(&mut self, idx: usize, data: &D) {
        if let Some(assigned_to) = self.assignments[idx].take() {
            self.counts[assigned_to] -= 1;
            self.partition_stats[assigned_to].forget(&data[idx]);

            if self.counts[assigned_to] == 0 {
                for cur in (0..=assigned_to).rev() {
                    if self.counts[cur] != 0 {
                        break;
                    }
                    self.remove_partition(cur);
                }
            }
        }

        // reset weight cache
        self.component_weights.take();
    }

    fn ln_pp_partition(&self, x: &X, partition_index: usize) -> f64 {
        let stat = &self.partition_stats[partition_index];

        // TODO: This is unnormalized, we should have a ln_pp_partition and
        // ln_pp_partition_unnormed as seperate functions
        (self.counts[partition_index] as f64).ln()
            + self
                .prior
                .ln_pp(x, &rv::prelude::DataOrSuffStat::SuffStat(stat))
    }

    fn ln_pp_empty(&self, x: &X) -> f64 {
        let suff_stat = rv::prelude::DataOrSuffStat::SuffStat(&self.empty_stat);
        self.prior.ln_pp(x, &suff_stat) + self.crp.alpha().ln()
    }

    fn n_partitions(&self) -> usize {
        self.counts.len()
    }

    fn counts(&self) -> &[usize] {
        &self.counts
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rv::misc::linspace;
    use rv::prelude::NormalGamma;

    use crate::utils::trapz;

    use super::*;

    #[test]
    fn normalized_density() {
        //let mut rng = SmallRng::seed_from_u64(0x1234);
        let mut rng = SmallRng::from_entropy();

        let data = vec![
            -4.5544936,
            -4.52564779,
            -4.1853936,
            -4.15343902,
            -3.86482039,
            -3.76028548,
            -3.56250333,
            -3.5287607,
            -3.31191143,
            -3.28149122,
            -3.27924403,
            -3.26417644,
            -3.26063013,
            -3.25927447,
            -3.2383868,
            -3.23130454,
            -3.01295357,
            -2.883912,
            -2.87986565,
            -2.85244248,
            -2.84868816,
            -2.79047781,
            -2.7677571,
            -2.76688933,
            -2.67850587,
            -2.65911541,
            -2.57474524,
            -2.57165177,
            -2.51851718,
            -2.45340006,
            -2.42083854,
            -2.41261285,
            -2.40158247,
            -2.31398857,
            -2.30898471,
            -2.30152007,
            -2.28106932,
            -2.27324462,
            -2.27137457,
            -2.25736296,
            -2.21289833,
            -2.21170942,
            -2.1199172,
            -2.10996804,
            -2.07670077,
            -2.02585074,
            -1.94293939,
            -1.90689499,
            -1.89953579,
            -1.8680245,
            -1.79222364,
            -1.68467249,
            -1.66716631,
            -1.66077255,
            -1.65975321,
            -1.6304449,
            -1.61056136,
            -1.5946174,
            -1.51460548,
            -1.48392865,
            -1.45421367,
            -1.38979104,
            -1.34829009,
            -1.34489445,
            -1.31401553,
            -1.31302229,
            -1.30624839,
            -1.25798209,
            -1.24206144,
            -1.15707343,
            -1.0226428,
            -1.01540607,
            -0.90029856,
            -0.87135452,
            -0.60457677,
            -0.56471383,
            -0.54827582,
            -0.19723344,
            -0.12971491,
            -0.09642296,
            0.02389986,
            0.03903373,
            0.13105941,
            0.16712943,
            0.24527941,
            0.30720275,
            0.32804499,
            0.39937038,
            0.45222418,
            0.46402806,
            0.50839666,
            0.52354158,
            0.71850454,
            0.73391244,
            0.75283399,
            0.76823031,
            0.83693656,
            0.85190246,
            0.87090987,
            0.87526806,
            0.87705185,
            1.02148868,
            1.02919284,
            1.05728064,
            1.084751,
            1.08523818,
            1.14328822,
            1.1709311,
            1.24131843,
            1.25068549,
            1.33721364,
            1.39069194,
            1.39749686,
            1.39923658,
            1.40903594,
            1.41755939,
            1.42423715,
            1.43150528,
            1.45171105,
            1.49697615,
            1.50704937,
            1.51518269,
            1.51817142,
            1.56013578,
            1.59679919,
            1.64821253,
            1.67553846,
            1.6838957,
            1.728994,
            1.72901348,
            1.83222399,
            1.83440032,
            1.88056726,
            1.90071539,
            1.91845868,
            1.92505068,
            1.97639987,
            2.01563257,
            2.04810074,
            2.08306057,
            2.0869686,
            2.1089411,
            2.11086443,
            2.23452142,
            2.24980082,
            2.25115166,
            2.27138148,
            2.2929052,
            2.33206326,
            2.35904414,
            2.36249945,
            2.3634702,
            2.37388054,
            2.37510924,
            2.37584074,
            2.39758125,
            2.44379782,
            2.45804199,
            2.4882009,
            2.51291248,
            2.52804,
            2.53888581,
            2.67518785,
            2.69374961,
            2.69681828,
            2.71947076,
            2.72233874,
            2.7515825,
            2.78897303,
            2.81511294,
            2.81940018,
            2.82363339,
            2.87784744,
            2.8830207,
            2.88597894,
            2.89626649,
            2.94799445,
            2.96770859,
            3.05648029,
            3.07250878,
            3.1643126,
            3.18982778,
            3.24839924,
            3.26002893,
            3.2606978,
            3.34193817,
            3.39407902,
            3.42099927,
            3.42112114,
            3.45967162,
            3.53421301,
            3.58515201,
            3.68629867,
            3.74953675,
            3.8596315,
            3.9434699,
            3.9487921,
            4.12092763,
            4.20821025,
            4.2749019,
        ];

        let mm = ConjugateMixtureModel::new(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            1.0,
            &mut rng,
        );

        let xs: Vec<f64> = linspace(-100.0, 100.0, 10_000);

        for (i, stat) in mm.partition_stats.iter().enumerate() {
            let part_f: Vec<f64> = xs
                .iter()
                .map(|x| mm.prior.ln_pp(x, &DataOrSuffStat::SuffStat(stat)).exp())
                .collect();

            let part_int: f64 = trapz(&part_f, &xs);
            if (part_int - 1.0).abs() > 1e-3 {
                panic!("{part_int} != 1.0 for component {i}");
            }
        }

        let ps: Vec<f64> = xs.iter().map(|x| mm.ln_f(x).exp()).collect();

        let int: f64 = trapz(&ps, &xs);

        assert::close(int, 1.0, 1e-4);
    }
}
