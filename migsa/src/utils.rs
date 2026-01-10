use rv::{
    dist::{Gaussian, Mixture, MvGaussian},
    misc::gauss_legendre_quadrature,
};

#[allow(clippy::needless_pass_by_value)]
#[cfg(test)]
pub(crate) fn convert_to_unicode(data: Vec<Vec<f64>>) -> Vec<Vec<char>> {
    // Define the range for mapping values from 0 to 1 to Unicode characters.
    const MIN_VALUE: f64 = 0.0;
    const MAX_VALUE: f64 = 1.0;

    // Define the range of Unicode characters to use.
    const MIN_CHAR_CODE: u32 = 0x2588; // U+2588 FULL BLOCK
    const MAX_CHAR_CODE: u32 = 0x2591; // U+2591 LIGHT SHADE

    let unicode_range = MAX_CHAR_CODE - MIN_CHAR_CODE + 1;

    // Map the input data to Unicode characters.
    let unicode_data: Vec<Vec<char>> = data
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| {
                    if value < MIN_VALUE {
                        std::char::from_u32(MIN_CHAR_CODE).expect("to be valid by construction")
                    } else if value > MAX_VALUE {
                        std::char::from_u32(MAX_CHAR_CODE).expect("to be valid by construction")
                    } else {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let unicode_value = MIN_CHAR_CODE
                            + (f64::from(unicode_range) * (value - MIN_VALUE)
                                / (MAX_VALUE - MIN_VALUE)) as u32;
                        std::char::from_u32(unicode_value).expect("to be valid by construction")
                    }
                })
                .collect()
        })
        .collect();

    unicode_data
}

/// Prevent the inner value from being verbosely / pretty printed during a debug.
pub(crate) struct NoPrettyPrint<T: std::fmt::Debug>(pub T);

impl<T: std::fmt::Debug> NoPrettyPrint<T> {
    pub const fn new(t: T) -> Self {
        Self(t)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for NoPrettyPrint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Prevent "{:#?}" from being used.
        write!(f, "{:?}", self.0)
    }
}

#[must_use]
pub fn trapz(y: &[f64], x: &[f64]) -> f64 {
    x.iter()
        .zip(x.iter().skip(1))
        .zip(y.iter().zip(y.iter().skip(1)))
        .map(|((x0, x1), (y0, y1))| (y1 + y0) * (x1 - x0) / 2.0)
        .sum()
}

pub fn total_variation_distance<P: Fn(f64) -> f64, Q: Fn(f64) -> f64>(
    p: P,
    q: Q,
    n: usize,
    bounds: (f64, f64),
) -> f64 {
    0.5 * gauss_legendre_quadrature(|x| (p(x) - q(x)).abs(), n, bounds)
}

pub trait Multivariate {
    type Univarites;

    fn univariate_marginals(&self) -> Self::Univarites;
}

impl Multivariate for MvGaussian {
    type Univarites = Vec<Gaussian>;

    fn univariate_marginals(&self) -> Self::Univarites {
        let mu = self.mu();
        let cov = self.cov();

        mu.iter()
            .zip(cov.diagonal().iter())
            .map(|(mu, s2)| Gaussian::new_unchecked(*mu, s2.sqrt()))
            .collect()
    }
}

impl Multivariate for Mixture<MvGaussian> {
    type Univarites = Vec<Mixture<Gaussian>>;

    fn univariate_marginals(&self) -> Self::Univarites {
        let cs: Vec<Vec<Gaussian>> = self
            .components()
            .iter()
            .map(Multivariate::univariate_marginals)
            .collect();
        let mut marginal_components: Vec<Vec<Gaussian>> = vec![vec![]; cs[0].len()];

        for c in cs {
            for (i, marg) in c.into_iter().enumerate() {
                marginal_components[i].push(marg);
            }
        }

        marginal_components
            .into_iter()
            .map(|cs| Mixture::new_unchecked(self.weights().clone(), cs))
            .collect()
    }
}

/// Online Mean and Variance
#[derive(Default)]
pub struct MeanAndVariance<T> {
    count: usize,
    mean: T,
    m2: T,
}

macro_rules! impl_m_and_v {
    ($t: ty) => {
        impl MeanAndVariance<$t> {
            pub fn update(self, new_value: $t) -> Self {
                let count = self.count + 1;
                let delta = new_value - self.mean;
                #[allow(clippy::cast_precision_loss)]
                let mean = self.mean + delta / (count as $t);
                let delta2 = new_value - mean;
                let m2 = delta.mul_add(delta2, self.m2);

                Self { count, mean, m2 }
            }

            pub const fn mean(&self) -> $t {
                self.mean
            }

            #[allow(clippy::cast_precision_loss)]
            pub fn sample_variance(&self) -> $t {
                self.m2 / ((self.count + 1) as $t)
            }
        }

        impl FromIterator<$t> for MeanAndVariance<$t> {
            fn from_iter<T: IntoIterator<Item = $t>>(iter: T) -> Self {
                iter.into_iter()
                    .fold(Self::default(), |acc, x| acc.update(x))
            }
        }
    };
}

impl_m_and_v!(f64);
impl_m_and_v!(f32);
