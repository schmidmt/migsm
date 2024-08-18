use std::fmt::Debug;

use itertools::Itertools;

use crate::mcmc::{Model, Sampler};
use crate::models::partition::PartitionModel;

/// Split-Merge Sampler
///
/// Reference: https://doi.org/10.1198/1061860043001
#[derive(Clone, Copy, Debug, Default)]
pub struct SplitMergeSampler {}

impl SplitMergeSampler {
    pub fn new() -> Self {
        Self::default()
    }
}

fn assignment_string(assn: &[Option<usize>]) -> String {
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
        dbg!(assignment_string(model.assignments()));

        // choose two distinct points from the data
        let i = rng.gen_range(0..(model.n_data()));
        let j = {
            let potential_j = rng.gen_range(0..(model.n_data() - 1));
            if potential_j >= i {
                potential_j + 1
            } else {
                potential_j
            }
        };

        dbg!((&i, &j));

        // get the assignments for each point
        let c_j = model.assignments()[j].expect("should be set");
        let c_i = model.assignments()[i].expect("should be set");

        dbg!((&c_i, &c_j));

        let to_assign = model
            .assignments()
            .iter()
            .enumerate()
            .filter_map(|(k, assgn)| {
                if let Some(c_k) = assgn {
                    (*c_k == c_j || *c_k == c_i).then_some(k)
                } else {
                    None
                }
            });

        // Clone the model
        let mut proposed_model = model.clone();

        if c_i == c_j {
            // If the same partition is selected, propose a split.
            let c_split_i = model.n_partitions();
            let c_split_j = c_j;

            let assignment_options = [c_split_i, c_split_j];
            let split_assignment_dist =
                rand::distributions::Slice::new(&assignment_options).expect("should be valid");

            to_assign.for_each(|k| {
                let new_partition = rng.sample(split_assignment_dist);
                proposed_model.assign(k, *new_partition, data);
            })
        } else {
            // If distinct partition are selected, propose a merge.
            to_assign.for_each(|k| {
                proposed_model.assign(k, c_j, data);
            });
        }

        dbg!(assignment_string(proposed_model.assignments()));

        // Generate a MH acceptance uniformly on [0, 1)
        let alpha_threshold: f64 = rng.gen();

        // Perform a Metropolis-Hastings accept-reject
        let log_alpha = dbg!(proposed_model.ln_score(data)) - dbg!(model.ln_score(data));
        let alpha = log_alpha.exp();

        if alpha_threshold < dbg!(alpha) {
            proposed_model
        } else {
            model
        }
    }
}

#[cfg(test)]
mod test {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rv::prelude::{Gaussian, NormalGamma};
    use rv::traits::Rv;

    use crate::mcmc::Sampler;
    use crate::models::mixture::ConjugateMixtureModel;
    use crate::models::Model;
    use crate::utils::convert_to_unicode;

    use super::SplitMergeSampler;

    #[test]
    fn two_gaussian_modes() {
        let mut rng = SmallRng::from_entropy();

        let g1 = Gaussian::new_unchecked(-200.0, 1.0);
        let g2 = Gaussian::new_unchecked(200.0, 1.0);

        let mut data: Vec<f64> = g1.sample(20, &mut rng);
        data.append(&mut g2.sample(20, &mut rng));

        let model = ConjugateMixtureModel::new(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            1.0,
            &mut rng,
        );

        let mut sampler = SplitMergeSampler::new();
        let assoc_mat: Vec<Vec<f64>> = vec![vec![0.0; data.len()]; data.len()];

        let n_samples: usize = 1_000;

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

        let ideal_model = ConjugateMixtureModel::with_assignment(
            NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
            data.iter(),
            1.0,
            &vec![
                0_usize, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ]
            .into_iter()
            .map(Some)
            .collect::<Vec<Option<usize>>>(),
        );

        println!("Ideal log_score = {}", ideal_model.ln_score(&data));
        println!("Sampled log_score = {}", model.ln_score(&data));

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
}
