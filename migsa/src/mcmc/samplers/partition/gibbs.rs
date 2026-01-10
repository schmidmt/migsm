use std::fmt::Debug;

use rand::Rng;
use rand::seq::SliceRandom;
use rv::misc::ln_pflip;

use crate::mcmc::Sampler;
use crate::models::partition::PartitionModel;

/// Gibbs based sampling on the space of partitions.
#[derive(Default, Clone, Copy, Debug)]
pub struct PartitionGibbs {}

impl PartitionGibbs {
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

impl<X, M, D> Sampler<M, D> for PartitionGibbs
where
    M: PartitionModel<X, D>,
    D: std::ops::Index<usize, Output = X>,
{
    fn step<R: Rng>(&mut self, mut model: M, data: &D, rng: &mut R) -> M {
        let mut indicies: Vec<usize> = (0..(model.assignments().len())).collect();
        indicies.shuffle(rng);

        for index in indicies {
            model.unassign(index, data);
            let x = &data[index];

            // TODO: Unchanged log weights could be cached.
            let mut log_weights: Vec<f64> = (0..model.n_partitions())
                .map(|i| model.ln_pp_partition(x, i))
                .collect();

            let new_component_pp = model.ln_pp_empty(x);
            log_weights.push(new_component_pp);

            let new_assignment = ln_pflip(&log_weights, false, rng);
            model.assign(index, new_assignment, data);
        }

        model
    }
}

#[cfg(test)]
mod tests {
    use super::PartitionGibbs;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rv::data::{GaussianSuffStat, Partition};
    use rv::prelude::{Crp, DataOrSuffStat, Gaussian, NormalGamma};
    use rv::traits::{ConjugatePrior, HasDensity, Sampleable, SuffStat};

    use crate::mcmc::samplers::partition::gibbs::PartitionModel;
    use crate::mcmc::{GewekeTest, PriorModel, ResampleModel, Sampler};
    use crate::models::mixture::{ConjugateMixtureModel, DirichletProcessComponentWeights};
    use crate::utils::{convert_to_unicode, total_variation_distance};

    const N: usize = 50;

    impl<DP> ResampleModel<Vec<f64>> for ConjugateMixtureModel<f64, Gaussian, NormalGamma, DP>
    where
        DP: DirichletProcessComponentWeights + Sampleable<Partition> + HasDensity<Partition>,
    {
        fn resample_data<R: rand::Rng>(
            &mut self,
            data: Option<&Vec<f64>>,
            rng: &mut R,
        ) -> Vec<f64> {
            let mut new_data: Vec<f64> = Vec::with_capacity(N);

            // Create new data by sampling from the posterior for each
            for i in 0..N {
                // remove from stat
                if let Some(old_data) = data {
                    let assignment = self.assignments()[i].expect("Should be assigned");
                    self.partition_stats_mut()[assignment].forget(&old_data[i]);
                }

                // Generate new point with the prior and data from the partition
                let posterior: NormalGamma =
                    self.prior()
                        .posterior(&DataOrSuffStat::<'_, f64, Gaussian>::SuffStat(
                            &self.partition_stats()
                                [self.assignments()[i].expect("Should be assigned")],
                        ));

                let new_dist: Gaussian = posterior.draw(rng);
                let new_data_point: f64 = new_dist.draw(rng);
                // Add new data to the new_data accumulator
                new_data.push(new_data_point);

                // Update the statistics to account for this new point
                let assignment = self.assignments()[i].expect("Should be assigned");
                self.partition_stats_mut()[assignment].observe(&new_data_point);
            }

            new_data
        }
    }

    impl<DP> PriorModel<Vec<f64>> for ConjugateMixtureModel<f64, Gaussian, NormalGamma, DP>
    where
        DP: DirichletProcessComponentWeights + Sampleable<Partition> + HasDensity<Partition>,
    {
        fn draw_from_prior<R: rand::Rng>(n: usize, alpha: f64, rng: &mut R) -> Self {
            let prior = NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0);
            let crp = Crp::new_unchecked(10.0, N);

            // Draw assignments
            let partition = crp.draw(rng);
            let assignments: Vec<Option<usize>> = partition.z().iter().map(|x| Some(*x)).collect();

            // Initialize stats
            let counts: Vec<usize> = partition.counts().clone();
            let empty_stat: GaussianSuffStat = GaussianSuffStat::new();

            // Accumulate the stats for each partition
            let partition_stats = counts
                .iter()
                .map(|&count| {
                    let mut stat = GaussianSuffStat::new();
                    let f = prior.draw(rng);
                    let sample: Vec<f64> = f.sample(count, rng);
                    stat.observe_many(&sample);
                    stat
                })
                .collect();

            Self::with_inner_values(
                prior,
                crp,
                assignments,
                counts.into_iter().map(|x| Some(x)).collect(),
                partition_stats,
            )
        }
    }

    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn geweke() {
        let sampler = PartitionGibbs::default();
        let mut rng = SmallRng::seed_from_u64(0x1234);

        sampler.assert_geweke(
            crate::mcmc::GewekeTestOptions {
                thinning: 100,
                n_samples: 500,
                burn_in: 100,
                min_p_value: 0.05,
                stat_map: |(m, _data): (ConjugateMixtureModel<f64, Gaussian, NormalGamma, Crp>, Vec<f64>)| {
                    let counts =
                        <ConjugateMixtureModel<f64, rv::dist::Gaussian, NormalGamma, Crp> as PartitionModel<
                            f64,
                            Vec<f64>,
                        >>::counts(&m);
                    let max_cluster_size = counts.iter().max().copied().unwrap_or(0_usize) as f64;
                    let n_clusters = counts.len() as f64;
                    let mean_cluster_size = (counts.iter().sum::<usize>() as f64) / n_clusters;

                    vec![n_clusters, mean_cluster_size, max_cluster_size]
                },
            },
            &mut rng,
        );
    }

    #[test]
    fn assign_unassign() {
        let data = vec![
            14.666_697_610_436_463,
            48.605_169_635_783_69,
            -2.361_739_052_978_315_5,
            1.783_119_534_569_353,
            0.958_454_269_711_143_3,
            0.604_745_319_478_709_4,
            31.214_703_868_606_932,
            -0.168_372_709_448_957_33,
            4.494_618_426_196_251,
            -1.473_907_929_899_958,
        ];

        let mut model = ConjugateMixtureModel::with_assignment(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            1.0,
            &[
                Some(0),
                Some(0),
                Some(3),
                Some(2),
                Some(3),
                Some(3),
                Some(0),
                Some(1),
                Some(0),
                Some(3),
            ],
        );

        // Check the initial config is what we expect.

        assert_eq!(model.counts, &[4, 1, 1, 4]);
        let stat: GaussianSuffStat = model.partition_stats[0];
        assert::close(stat.mean(), 24.745_297_385_255_835, 1E-10);
        assert::close(stat.sum_x_sq(), 3_572.133_866_521_219, 1E-10);
        assert::close(stat.sum_x(), 98.981_189_541_023_34, 1E-10);

        model.unassign(1, &data);

        assert_eq!(
            model.assignments,
            &[
                Some(0),
                None,
                Some(3),
                Some(2),
                Some(3),
                Some(3),
                Some(0),
                Some(1),
                Some(0),
                Some(3),
            ],
        );
        let expected_mean = (data[0] + data[6] + data[8]) / 3.0;

        let stat: &GaussianSuffStat = &model.partition_stats[0];

        assert::close(expected_mean, stat.mean(), 1E-9);

        let expected_var = (data[8] - expected_mean).mul_add(
            data[8] - expected_mean,
            (data[6] - expected_mean)
                .mul_add(data[6] - expected_mean, (data[0] - expected_mean).powi(2)),
        ) / 3.0;

        assert::close(
            expected_var,
            3.0f64.mul_add(
                stat.mean().powi(2),
                (2.0 * stat.sum_x()).mul_add(-stat.mean(), stat.sum_x_sq()),
            ) / 3.0,
            1E-9,
        );

        dbg!(&model);

        // Assign stats
        model.assign(1, 4, &data);

        dbg!(&model);
    }

    #[test]
    fn mixture_model_add_remove() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut model = ConjugateMixtureModel::with_assignment(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            1.0,
            &[None, None, None, None, None],
        );

        assert!(model.counts.is_empty());

        model.assign(0, 3, &data);
        assert_eq!(model.counts, &[0, 0, 0, 1]);

        model.assign(3, 1, &data);
        assert_eq!(model.counts, &[0, 1, 0, 1]);

        model.unassign(0, &data);
        assert_eq!(model.counts, &[0, 1]);
    }

    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn two_gaussian_modes() {
        let mut rng = SmallRng::from_os_rng();

        let g1 = Gaussian::new_unchecked(-20.0, 1.0);
        let g2 = Gaussian::new_unchecked(20.0, 1.0);

        let mut data: Vec<f64> = g1.sample(40, &mut rng);
        data.append(&mut g2.sample(40, &mut rng));

        let model = ConjugateMixtureModel::new(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            1.0,
            &mut rng,
        );

        let mut sampler = PartitionGibbs::new();
        let assoc_mat: Vec<Vec<f64>> = vec![vec![0.0; data.len()]; data.len()];

        let n_samples: usize = 1000;

        let (model, assoc) = (0..n_samples).fold((model, assoc_mat), |(m, mut a), _i| {
            let next_model = sampler.step(m, &data, &mut rng);

            for p in 0..data.len() {
                for q in (p + 1)..data.len() {
                    if next_model.assignments[p] == next_model.assignments[q] {
                        a[p][q] += 1.0 / (n_samples as f64);
                        a[q][p] += 1.0 / (n_samples as f64);
                    }
                }
                a[p][p] += 1.0 / (n_samples as f64);
            }

            (next_model, a)
        });

        let assoc_glyphs = convert_to_unicode(assoc);

        for row in assoc_glyphs {
            for c in row {
                print!("{c}");
            }
            println!();
        }

        println!(
            "{:?}",
            model
                .assignments
                .into_iter()
                .map(|x| x.expect("to be some"))
                .collect::<Vec<usize>>()
        );

        assert_eq!(model.counts.len(), 2);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[test]
    fn density_estimate() {
        let mut rng = SmallRng::seed_from_u64(0x1234);

        let g1 = Gaussian::new_unchecked(-2.0, 1.0);
        let g2 = Gaussian::new_unchecked(2.0, 1.0);

        //let xs: Vec<f64> = rv::misc::linspace(-10.0, 10.0, 1_000);

        let n: f64 = 200.0;
        let mut data: Vec<f64> = g1.sample((0.3 * n) as usize, &mut rng);
        data.append(&mut g2.sample((0.7 * n) as usize, &mut rng));

        data.sort_by(f64::total_cmp);

        let model: ConjugateMixtureModel<f64, Gaussian, NormalGamma> = ConjugateMixtureModel::new(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            1.0,
            &mut rng,
        );

        let mut sampler = PartitionGibbs::new();

        let deltas = sampler
            .iter_sample(model, &data, &mut rng, |model| {
                total_variation_distance(
                    |x| 0.3f64.mul_add(g1.f(&x), 0.7 * g2.f(&x)),
                    |x| model.f(&x),
                    10,
                    (-10.0, 10.0),
                )
            })
            .take(1000);

        let mean_delta = deltas.sum::<f64>() / 1000.0;
        assert!(mean_delta < 0.05);
    }
}
