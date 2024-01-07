use std::fmt::Debug;

use rand::seq::SliceRandom;
use rand::Rng;
use rv::misc::ln_pflip;

use crate::mcmc::Sampler;
use crate::models::partition::PartitionModel;

/// Gibbs based sampling on the space of partitions.
#[derive(Default, Clone, Copy, Debug)]
pub struct PartitionGibbs {}

impl PartitionGibbs {
    pub fn new() -> Self {
        Self {}
    }
}

impl<X, M, D> Sampler<M, D> for PartitionGibbs
where
    X: Clone,
    M: PartitionModel<X, D> + std::fmt::Debug,
    D: std::ops::Index<usize, Output = X> + std::fmt::Debug,
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

            let new_assignment = ln_pflip(&log_weights, 1, false, rng)[0];
            model.assign(index, new_assignment, data);
        }

        model
    }
}

#[cfg(test)]
mod tests {
    use super::PartitionGibbs;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rv::data::GaussianSuffStat;
    use rv::prelude::{Crp, DataOrSuffStat, Gaussian, NormalGamma};
    use rv::traits::{ConjugatePrior, Rv, SuffStat};

    use crate::mcmc::samplers::partition::gibbs::PartitionModel;
    use crate::mcmc::{GewekeTest, PriorModel, ResampleModel, Sampler};
    use crate::models::mixture::ConjugateMixtureModel;
    use crate::utils::{convert_to_unicode, total_variation_distance};

    const N: usize = 50;

    impl ResampleModel<Vec<f64>> for ConjugateMixtureModel<f64, Gaussian, NormalGamma> {
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
                    self.partition_stats[self.assignments[i].expect("Should be assigned")]
                        .forget(&old_data[i]);
                }

                // Generate new point with the prior and data from the partition
                let new_data_point = self
                    .prior
                    .posterior(&DataOrSuffStat::SuffStat(
                        &self.partition_stats[self.assignments[i].expect("Should be assigned")],
                    ))
                    .draw(rng)
                    .draw(rng);
                // Add new data to the new_data accumulator
                new_data.push(new_data_point);

                // Update the statistics to account for this new point
                self.partition_stats[self.assignments[i].expect("Should be assigned")]
                    .observe(&new_data_point);
            }

            new_data
        }
    }
    impl PriorModel<Vec<f64>> for ConjugateMixtureModel<f64, Gaussian, NormalGamma> {
        fn draw_from_prior<R: rand::Rng>(rng: &mut R) -> Self {
            let prior = NormalGamma::new(0.0, 1.0, 1.0, 1.0).unwrap();
            let crp = Crp::new(10.0, N).unwrap();

            // Draw assignments from CRP
            let partition = crp.draw(rng);
            let assignments: Vec<Option<usize>> =
                partition.z().into_iter().map(|x| Some(*x)).collect();

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

            Self {
                prior,
                crp,
                assignments,
                counts,
                partition_stats,
                empty_stat,
                _phantom_x: std::marker::PhantomData,
                _phantom_fx: std::marker::PhantomData,
                component_weights: std::cell::OnceCell::new(),
            }
        }
    }

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
                stat_map: |(m, _data): (ConjugateMixtureModel<f64, Gaussian, NormalGamma>, Vec<f64>)| {
                    let counts =
                        <ConjugateMixtureModel<f64, rv::dist::Gaussian, NormalGamma> as PartitionModel<
                            f64,
                            Vec<f64>,
                        >>::counts(&m);
                    let max_cluster_size = counts.iter().max().cloned().unwrap_or(0_usize) as f64;
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
            14.666697610436463,
            48.60516963578369,
            -2.3617390529783155,
            1.783119534569353,
            0.9584542697111433,
            0.6047453194787094,
            31.214703868606932,
            -0.16837270944895733,
            4.494618426196251,
            -1.473907929899958,
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
        assert::close(model.partition_stats[0].mean(), 24.745297385255835, 1E-10);
        assert::close(
            model.partition_stats[0].sum_x_sq(),
            3572.133866521219,
            1E-10,
        );
        assert::close(model.partition_stats[0].sum_x(), 98.98118954102334, 1E-10);

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

        let stat = &model.partition_stats[0];

        assert::close(expected_mean, stat.mean(), 1E-9);

        let expected_var = ((data[0] - expected_mean).powi(2)
            + (data[6] - expected_mean).powi(2)
            + (data[8] - expected_mean).powi(2))
            / 3.0;

        assert::close(
            expected_var,
            (stat.sum_x_sq() - 2.0 * stat.sum_x() * stat.mean() + 3.0 * stat.mean().powi(2)) / 3.0,
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

    #[test]
    fn two_gaussian_modes() {
        let mut rng = SmallRng::from_entropy();

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
            println!()
        }

        println!(
            "{:?}",
            model
                .assignments
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Vec<usize>>()
        );

        assert_eq!(model.counts.len(), 2);
    }

    #[test]
    fn density_estimate() {
        let mut rng = SmallRng::seed_from_u64(0x1234);

        let g1 = Gaussian::new_unchecked(-2.0, 1.0);
        let g2 = Gaussian::new_unchecked(2.0, 1.0);

        //let xs: Vec<f64> = rv::misc::linspace(-10.0, 10.0, 1_000);

        let n: f64 = 200.0;
        let mut data: Vec<f64> = g1.sample((0.3 * n) as usize, &mut rng);
        data.append(&mut g2.sample((0.7 * n) as usize, &mut rng));

        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
                    |x| 0.3 * g1.f(&x) + 0.7 * g2.f(&x),
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
