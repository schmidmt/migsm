use std::fmt::Debug;
use std::marker::PhantomData;

use nalgebra::DVector;
use rand::{Rng, SeedableRng};
use rv::data::Partition;
use rv::dist::Mixture;
use rv::misc::{LogSumExp, ln_pflip};
use rv::prelude::{Crp, DataOrSuffStat};
use rv::traits::{ConjugatePrior, HasDensity, HasSuffStat, Rv, Sampleable, SuffStat};

use crate::mcmc::samplers::stick::StickBreaking;
use crate::utils::NoPrettyPrint;

use super::Model;
use super::partition::PartitionModel;

pub trait DirichletProcessComponentWeights {
    fn component_probabilities(&self, partition: &rv::data::Partition) -> Vec<f64>;

    fn empty_component_p(&self, n: usize) -> f64;
}

impl DirichletProcessComponentWeights for Crp {
    fn component_probabilities(&self, partition: &rv::data::Partition) -> Vec<f64> {
        #[allow(clippy::cast_precision_loss)]
        let weights: Vec<f64> = partition
            .z()
            .iter()
            .map(|x| *x as f64)
            .chain(std::iter::once(self.alpha()))
            .collect();

        // Normalize the weights
        let weight_sum: f64 = weights.iter().sum();
        weights.into_iter().map(|w| w / weight_sum).collect()
    }

    fn empty_component_p(&self, n: usize) -> f64 {
        self.alpha() / (n as f64)
    }
}

impl DirichletProcessComponentWeights for StickBreaking {
    fn component_probabilities(&self, partition: &rv::data::Partition) -> Vec<f64> {
        todo!()
    }

    fn empty_component_p(&self, n: usize) -> f64 {
        todo!()
    }
}

#[derive(Clone)]
pub struct ConjugateMixtureModel<X, Fx, Pr, Dp>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone,
{
    prior: Pr,
    dirichlet_process: Dp,
    assignments: Vec<Option<usize>>,
    counts: Vec<usize>,
    partition_stats: Vec<Fx::Stat>,
    empty_stat: Fx::Stat,
    _phantom_x: PhantomData<X>,
    _phantom_fx: PhantomData<Fx>,
}

impl<X, Fx, Pr, Dp> std::fmt::Debug for ConjugateMixtureModel<X, Fx, Pr, Dp>
where
    X: Clone + Debug,
    Fx: Rv<X> + HasSuffStat<X> + Debug,
    Pr: ConjugatePrior<X, Fx> + Debug,
    Fx::Stat: Clone + Debug,
    Dp: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let assignments: Vec<_> = self
            .assignments
            .iter()
            .map(|a| {
                a.as_ref()
                    .map_or_else(|| String::from("-"), ToString::to_string)
            })
            .map(NoPrettyPrint::new)
            .collect();

        f.debug_struct("MixtureModel")
            .field("prior", &self.prior)
            .field("dirichlet_process", &self.dirichlet_process)
            .field("assignments", &NoPrettyPrint::new(assignments))
            .field("counts", &NoPrettyPrint::new(&self.counts))
            .field(
                "partition_stats",
                &NoPrettyPrint::new(&self.partition_stats),
            )
            .finish_non_exhaustive()
    }
}

impl<X, Fx, Pr, Dp> ConjugateMixtureModel<X, Fx, Pr, Dp>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
    Dp: Sampleable<rv::data::Partition>,
{
    /// Create a new `ConjugateMixtureModel` from a prior, data, alpha
    ///
    /// # Panics
    /// If the number of data is zero or the alpha is non strictly positive, then this will panic.
    pub fn new<'a, R, ID>(prior: Pr, data: ID, prior_process: Dp, rng: &mut R) -> Self
    where
        R: Rng,
        ID: ExactSizeIterator + Iterator<Item = &'a X>,
        X: 'a,
    {
        let fx = prior.draw(&mut rand::rngs::SmallRng::seed_from_u64(0x1234));
        let init_partition = prior_process.draw(rng);

        let mut partition_stats: Vec<Fx::Stat> = (0..(init_partition.k()))
            .map(|_| fx.empty_suffstat())
            .collect();

        let assignments: Vec<Option<usize>> =
            init_partition.z().iter().copied().map(Some).collect();

        let mut counts: Vec<usize> = vec![0; init_partition.k()];
        init_partition.z().iter().zip(data).for_each(|(&asgn, x)| {
            counts[asgn] += 1;
            partition_stats[asgn].observe(x);
        });

        Self {
            prior,
            dirichlet_process: prior_process,
            partition_stats,
            empty_stat: fx.empty_suffstat(),
            assignments,
            counts,
            _phantom_x: PhantomData,
            _phantom_fx: PhantomData,
        }
    }

    pub(crate) fn with_inner_values(
        prior: Pr,
        dirichlet_process: Dp,
        partition_stats: Vec<Fx::Stat>,
        assignments: Vec<Option<usize>>,
        counts: Vec<usize>,
    ) -> Self {
        let fx = prior.draw(&mut rand::rngs::SmallRng::seed_from_u64(0x1234));
        Self {
            prior,
            dirichlet_process,
            assignments,
            counts,
            partition_stats,
            empty_stat: fx.empty_suffstat(),
            _phantom_x: PhantomData,
            _phantom_fx: PhantomData,
        }
    }

    /// Create a new `ConjugateMixtureModel` from a set of given assignments
    ///
    /// # Panics
    /// If the number of data is zero or the alpha is non strictly positive, then this will panic.
    pub fn with_assignment<'a, ID>(
        prior: Pr,
        data: ID,
        dirichlet_process: Dp,
        assignments: &[Option<usize>],
    ) -> Self
    where
        ID: ExactSizeIterator + Iterator<Item = &'a X>,
        X: 'a,
    {
        assert_eq!(
            assignments.len(),
            data.len(),
            "Assignment doesn't match data size: {} != {}",
            assignments.len(),
            data.len()
        );

        let fx = prior.draw(&mut rand::rngs::SmallRng::seed_from_u64(0x1234));

        let n_partitions: usize = assignments
            .iter()
            .copied()
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
            dirichlet_process,
            partition_stats,
            empty_stat: fx.empty_suffstat(),
            assignments: assignments.to_vec(),
            counts,
            _phantom_x: PhantomData,
            _phantom_fx: PhantomData,
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
    }

    pub fn partition_stats(&self) -> &[Fx::Stat] {
        &self.partition_stats
    }

    pub fn partition_stats_mut(&mut self) -> &mut [Fx::Stat] {
        &mut self.partition_stats
    }

    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    pub fn assignments(&self) -> &[Option<usize>] {
        &self.assignments
    }

    pub fn dirichlet_process(&self) -> &Dp {
        &self.dirichlet_process
    }

    pub fn prior(&self) -> &Pr {
        &self.prior
    }

    pub fn empty_stat(&self) -> &Fx::Stat {
        &self.empty_stat
    }
}

impl<X, Fx, Pr, DP> ConjugateMixtureModel<X, Fx, Pr, DP>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
    DP: DirichletProcessComponentWeights + HasDensity<rv::data::Partition>,
{
    /// Portion of the `log_score` from the inner distributions marginal probability.
    pub fn log_m(&self) -> f64 {
        self.partition_stats
            .iter()
            .map(|ps| self.prior.ln_m(&DataOrSuffStat::SuffStat(ps)))
            .sum()
    }

    /// Portion of `log_score` from the CRP prior
    pub fn dp_log_f(&self) -> f64 {
        let part = rv::data::Partition::new_unchecked(
            self.assignments.iter().flatten().copied().collect(),
            self.counts.clone(),
        );
        self.dirichlet_process.ln_f(&part)
    }

    fn component_weights(&self) -> Vec<f64> {
        self.dirichlet_process
            .component_probabilities(&Partition::new_unchecked(
                self.assignments.iter().flatten().copied().collect(),
                self.counts.clone(),
            ))
    }
}

/*
impl<X, Y, Fx, Pr> ConjugateMixtureModel<X, Fx, Pr>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X> + MultivariateRv<X, Y>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
{
    pub fn marginal_ln_f(&self, y: &Y) -> f64 {
        // The ln poster predictive probabilities for each component
        let component_posterior_ln_ps = self
            .partition_stats
            .iter()
            .map(|stat| self.prior.ln_pp(x, &DataOrSuffStat::SuffStat(stat)))
            .chain(std::iter::once(
                self.prior
                    .ln_pp(x, &DataOrSuffStat::SuffStat(&self.empty_stat)),
            ));

        // Component weights
        let weights: &[f64] = self.component_weights();

        let ln_ps: Vec<f64> = weights
            .iter()
            .zip(component_posterior_ln_ps)
            .map(|(w, component_ln_pp)| w.ln() + component_ln_pp)
            .collect();

        logsumexp(&ln_ps)
    }
}
*/

impl<X, Fx, Pr, DP> rv::traits::Sampleable<Mixture<Fx>> for ConjugateMixtureModel<X, Fx, Pr, DP>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
    DP: DirichletProcessComponentWeights + HasDensity<rv::data::Partition>,
{
    fn draw<R: Rng>(&self, rng: &mut R) -> Mixture<Fx> {
        let weights: Vec<f64> = self.component_weights();

        let components = self
            .partition_stats
            .iter()
            .chain(std::iter::once(&self.empty_stat))
            .map(|s| self.prior.posterior(&DataOrSuffStat::SuffStat(s)).draw(rng))
            .collect();

        Mixture::new_unchecked(weights, components)
    }
}

impl<Fx, Pr, DP> rv::traits::HasDensity<DVector<f64>>
    for ConjugateMixtureModel<DVector<f64>, Fx, Pr, DP>
where
    Fx: Rv<DVector<f64>> + HasSuffStat<DVector<f64>>,
    Pr: ConjugatePrior<DVector<f64>, Fx>,
    Fx::Stat: Clone + Debug,
    DP: DirichletProcessComponentWeights + HasDensity<rv::data::Partition>,
{
    fn ln_f(&self, x: &DVector<f64>) -> f64 {
        self.partition_stats
            .iter()
            .chain(std::iter::once(&self.empty_stat))
            .zip(self.component_weights())
            .map(|(stat, weight)| {
                self.prior.ln_pp(x, &DataOrSuffStat::SuffStat(stat)) + weight.ln()
            })
            .sum()
    }
}

macro_rules! cmm_float {
    ($kind: ty) => {
        impl<Fx, Pr, DP> rv::traits::HasDensity<$kind> for ConjugateMixtureModel<$kind, Fx, Pr, DP>
        where
            Fx: Rv<$kind> + HasSuffStat<$kind>,
            Pr: ConjugatePrior<$kind, Fx>,
            Fx::Stat: Clone + Debug,
            DP: DirichletProcessComponentWeights + HasDensity<rv::data::Partition>,
        {
            fn ln_f(&self, x: &$kind) -> f64 {
                // The ln poster predictive probabilities for each component
                let component_posterior_ln_ps = self
                    .partition_stats
                    .iter()
                    .map(|stat| self.prior.ln_pp(x, &DataOrSuffStat::SuffStat(stat)))
                    .chain(std::iter::once(
                        self.prior
                            .ln_pp(x, &DataOrSuffStat::SuffStat(&self.empty_stat)),
                    ));

                // Component weights
                let weights = self.component_weights();

                weights
                    .iter()
                    .zip(component_posterior_ln_ps)
                    .map(|(w, component_ln_pp)| w.ln() + component_ln_pp)
                    .logsumexp()
            }
        }

        impl<Fx, Pr, DP> rv::traits::Sampleable<$kind> for ConjugateMixtureModel<$kind, Fx, Pr, DP>
        where
            Fx: Rv<$kind> + HasSuffStat<$kind>,
            Pr: ConjugatePrior<$kind, Fx>,
            Fx::Stat: Clone + Debug,
            DP: DirichletProcessComponentWeights + HasDensity<rv::data::Partition>,
        {
            fn draw<R: Rng>(&self, rng: &mut R) -> $kind {
                let weights = self.component_weights();
                let component_idx = ln_pflip(weights, false, rng);

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
    };
}

cmm_float!(f64);
cmm_float!(f32);

impl<X, Fx, Pr, D, DP> Model<D> for ConjugateMixtureModel<X, Fx, Pr, DP>
where
    X: Clone,
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
    Fx::Stat: Clone + Debug,
    D: std::ops::Index<usize, Output = X>,
    DP: DirichletProcessComponentWeights + HasDensity<rv::data::Partition>,
{
    fn ln_score(&self, _data: &D) -> f64 {
        self.dp_log_f() + self.log_m()
    }
}

impl<X, Fx, Pr, D, DP> PartitionModel<X, D> for ConjugateMixtureModel<X, Fx, Pr, DP>
where
    X: Clone + std::fmt::Debug,
    Fx: Rv<X> + HasSuffStat<X> + std::fmt::Debug,
    Pr: ConjugatePrior<X, Fx> + std::fmt::Debug,
    Fx::Stat: Clone + Debug + PartialEq,
    D: std::ops::Index<usize, Output = X>,
    DP: DirichletProcessComponentWeights
        + HasDensity<rv::data::Partition>
        + Sampleable<rv::data::Partition>,
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
    }

    #[allow(clippy::cast_precision_loss)]
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
        self.prior.ln_pp(x, &suff_stat)
            + self
                .dirichlet_process
                .empty_component_p(self.counts.len())
                .ln()
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

    #[allow(clippy::too_many_lines)]
    #[test]
    fn normalized_density() {
        //let mut rng = SmallRng::seed_from_u64(0x1234);
        let mut rng = SmallRng::from_os_rng();

        let data = vec![
            -4.554_493_6,
            -4.525_647_79,
            -4.185_393_6,
            -4.153_439_02,
            -3.864_820_39,
            -3.760_285_48,
            -3.562_503_33,
            -3.528_760_7,
            -3.311_911_43,
            -3.281_491_22,
            -3.279_244_03,
            -3.264_176_44,
            -3.260_630_13,
            -3.259_274_47,
            -3.238_386_8,
            -3.231_304_54,
            -3.012_953_57,
            -2.883_912,
            -2.879_865_65,
            -2.852_442_48,
            -2.848_688_16,
            -2.790_477_81,
            -2.767_757_1,
            -2.766_889_33,
            -2.678_505_87,
            -2.659_115_41,
            -2.574_745_24,
            -2.571_651_77,
            -2.518_517_18,
            -2.453_400_06,
            -2.420_838_54,
            -2.412_612_85,
            -2.401_582_47,
            -2.313_988_57,
            -2.308_984_71,
            -2.301_520_07,
            -2.281_069_32,
            -2.273_244_62,
            -2.271_374_57,
            -2.257_362_96,
            -2.212_898_33,
            -2.211_709_42,
            -2.119_917_2,
            -2.109_968_04,
            -2.076_700_77,
            -2.025_850_74,
            -1.942_939_39,
            -1.906_894_99,
            -1.899_535_79,
            -1.868_024_5,
            -1.792_223_64,
            -1.684_672_49,
            -1.667_166_31,
            -1.660_772_55,
            -1.659_753_21,
            -1.630_444_9,
            -1.610_561_36,
            -1.594_617_4,
            -1.514_605_48,
            -1.483_928_65,
            -1.454_213_67,
            -1.389_791_04,
            -1.348_290_09,
            -1.344_894_45,
            -1.314_015_53,
            -1.313_022_29,
            -1.306_248_39,
            -1.257_982_09,
            -1.242_061_44,
            -1.157_073_43,
            -1.022_642_8,
            -1.015_406_07,
            -0.900_298_56,
            -0.871_354_52,
            -0.604_576_77,
            -0.564_713_83,
            -0.548_275_82,
            -0.197_233_44,
            -0.129_714_91,
            -0.096_422_96,
            0.023_899_86,
            0.039_033_73,
            0.131_059_41,
            0.167_129_43,
            0.245_279_41,
            0.307_202_75,
            0.328_044_99,
            0.399_370_38,
            0.452_224_18,
            0.464_028_06,
            0.508_396_66,
            0.523_541_58,
            0.718_504_54,
            0.733_912_44,
            0.752_833_99,
            0.768_230_31,
            0.836_936_56,
            0.851_902_46,
            0.870_909_87,
            0.875_268_06,
            0.877_051_85,
            1.021_488_68,
            1.029_192_84,
            1.057_280_64,
            1.084_751_,
            1.085_238_18,
            1.143_288_22,
            1.170_931_1,
            1.241_318_43,
            1.250_685_49,
            1.337_213_64,
            1.390_691_94,
            1.397_496_86,
            1.399_236_58,
            1.409_035_94,
            1.417_559_39,
            1.424_237_15,
            1.431_505_28,
            1.451_711_05,
            1.496_976_15,
            1.507_049_37,
            1.515_182_69,
            1.518_171_42,
            1.560_135_78,
            1.596_799_19,
            1.648_212_53,
            1.675_538_46,
            1.683_895_7,
            1.728_994,
            1.729_013_48,
            1.832_223_99,
            1.834_400_32,
            1.880_567_26,
            1.900_715_39,
            1.918_458_68,
            1.925_050_68,
            1.976_399_87,
            2.015_632_57,
            2.048_100_74,
            2.083_060_57,
            2.086_968_6,
            2.108_941_1,
            2.110_864_43,
            2.234_521_42,
            2.249_800_82,
            2.251_151_66,
            2.271_381_48,
            2.292_905_2,
            2.332_063_26,
            2.359_044_14,
            2.362_499_45,
            2.363_470_2,
            2.373_880_54,
            2.375_109_24,
            2.375_840_74,
            2.397_581_25,
            2.443_797_82,
            2.458_041_99,
            2.488_200_9,
            2.512_912_48,
            2.528_04,
            2.538_885_81,
            2.675_187_85,
            2.693_749_61,
            2.696_818_28,
            2.719_470_76,
            2.722_338_74,
            2.751_582_5,
            2.788_973_03,
            2.815_112_94,
            2.819_400_18,
            2.823_633_39,
            2.877_847_44,
            2.883_020_7,
            2.885_978_94,
            2.896_266_49,
            2.947_994_45,
            2.967_708_59,
            3.056_480_29,
            3.072_508_78,
            3.164_312_6,
            3.189_827_78,
            3.248_399_24,
            3.260_028_93,
            3.260_697_8,
            3.341_938_17,
            3.394_079_02,
            3.420_999_27,
            3.421_121_14,
            3.459_671_62,
            3.534_213_01,
            3.585_152_01,
            3.686_298_67,
            3.749_536_75,
            3.859_631_5,
            3.943_469_9,
            3.948_792_1,
            4.120_927_63,
            4.208_210_25,
            4.274_901_9,
        ];

        let mm: ConjugateMixtureModel<f64, rv::prelude::Gaussian, NormalGamma, Crp> =
            ConjugateMixtureModel::new(
                NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
                data.iter(),
                Crp::new(1.0, data.len()).unwrap(),
                &mut rng,
            );

        let xs: Vec<f64> = linspace(-100.0, 100.0, 10_000);

        for (i, stat) in mm.partition_stats.iter().enumerate() {
            let part_f: Vec<f64> = xs
                .iter()
                .map(|x| mm.prior.ln_pp(x, &DataOrSuffStat::SuffStat(stat)).exp())
                .collect();

            let part_int: f64 = trapz(&part_f, &xs);
            assert!(
                ((part_int - 1.0).abs() <= 1e-3),
                "{part_int} != 1.0 for component {i}"
            );
        }

        let ps: Vec<f64> = xs.iter().map(|x| mm.ln_f(x).exp()).collect();

        let int: f64 = trapz(&ps, &xs);

        assert::close(int, 1.0, 1e-4);
    }
}
