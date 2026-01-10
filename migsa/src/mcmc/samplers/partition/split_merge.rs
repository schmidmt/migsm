use std::fmt::Debug;

use itertools::Itertools;

use crate::mcmc::{Model, Sampler};
use crate::models::partition::PartitionModel;

/// Split-Merge Sampler
///
/// Reference: <https://doi.org/10.1198/1061860043001>
#[derive(Clone, Copy, Debug, Default)]
pub struct SplitMergeSampler {}

impl SplitMergeSampler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

fn assignment_string(assn: &[Option<usize>]) -> String {
    #[allow(unstable_name_collisions)]
    assn.iter()
        .map(|x| x.map_or_else(|| "X".to_string(), |x| x.to_string()))
        .intersperse(",".to_string())
        .collect::<String>()
}

impl<M, D, X> Sampler<M, D> for SplitMergeSampler
where
    M: Model<D> + PartitionModel<X, D> + Clone + Debug,
    D: std::ops::Index<usize, Output = X>,
{
    fn step<R: rand::Rng>(&mut self, model: M, data: &D, rng: &mut R) -> M {
        // choose two distinct points from the data
        let i = rng.random_range(0..(model.n_data()));
        let j = {
            let potential_j = rng.random_range(0..(model.n_data() - 1));
            if potential_j >= i {
                potential_j + 1
            } else {
                potential_j
            }
        };

        // get the assignments for each point
        let c_j = model.assignments()[j].expect("should be set");
        let c_i = model.assignments()[i].expect("should be set");

        let to_assign = model
            .assignments()
            .iter()
            .enumerate()
            .filter_map(|(k, assgn)| {
                assgn
                    .as_ref()
                    .and_then(|c_k| (*c_k == c_j || *c_k == c_i).then_some(k))
            });

        // Clone the model
        let mut proposed_model = model.clone();

        if c_i == c_j {
            // If the same partition is selected, propose a split.
            let c_split_i = model.n_partitions();
            //let c_split_j = c_j;

            to_assign.for_each(|k| {
                if rng.random() {
                    proposed_model.assign(k, c_split_i, data);
                }
            });
        } else {
            // If distinct partition are selected, propose a merge.
            to_assign.for_each(|k| {
                proposed_model.assign(k, c_j, data);
            });
        }

        dbg!(assignment_string(proposed_model.assignments()));

        // Generate a MH acceptance uniformly on [0, 1)
        let alpha_threshold: f64 = rng.random();

        // Perform a Metropolis-Hastings accept-reject
        let log_alpha = dbg!(proposed_model.ln_score(data)) - dbg!(model.ln_score(data));
        let alpha = log_alpha.exp();

        if dbg!(alpha_threshold) < dbg!(alpha) {
            proposed_model
        } else {
            model
        }
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rv::prelude::{Gaussian, NormalGamma};
    use rv::traits::{HasDensity, Sampleable};

    use crate::mcmc::Sampler;
    use crate::models::Model;
    use crate::models::mixture::ConjugateMixtureModel;
    use crate::utils::convert_to_unicode;

    use super::SplitMergeSampler;

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    #[test]
    fn two_gaussian_modes() {
        let mut rng = SmallRng::from_os_rng();

        const N: usize = 50;

        let g1 = Gaussian::new_unchecked(-200.0, 1.0);
        let g2 = Gaussian::new_unchecked(200.0, 1.0);

        let mut data: Vec<f64> = g1.sample(N, &mut rng);
        data.append(&mut g2.sample(N, &mut rng));

        let model = ConjugateMixtureModel::new(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            10.0,
            &mut rng,
        );

        let mut sampler = SplitMergeSampler::new();
        let assoc_mat: Vec<Vec<f64>> = vec![vec![0.0; data.len()]; data.len()];

        let n_samples: usize = 10_000;

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

        let ideal_assignments: Vec<Option<usize>> = (0..N)
            .map(|_| 0)
            .chain((0..N).map(|_| 1))
            .map(Some)
            .collect();

        let ideal_model = ConjugateMixtureModel::with_assignment(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            10.0,
            &ideal_assignments,
        );

        println!("Ideal log_score = {}", ideal_model.ln_score(&data));
        println!("Sampled log_score = {}", model.ln_score(&data));

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
                .iter()
                .map(|x| x.expect("to be some"))
                .collect::<Vec<usize>>()
        );

        dbg!(&model.partition_stats);

        let tvd_est = data
            .iter()
            .map(|x| (model.f(x) - ideal_model.f(x)).abs())
            .sum::<f64>()
            / (data.len() as f64);
        println!("tvd_est = {tvd_est}");

        dbg!(data);

        assert_eq!(model.counts.len(), 2);
    }
}
