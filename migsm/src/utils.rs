use rv::misc::gauss_legendre_quadrature;

#[cfg(test)]
pub(crate) fn convert_to_unicode(data: Vec<Vec<f64>>) -> Vec<Vec<char>> {
    // Define the range for mapping values from 0 to 1 to Unicode characters.
    let min_value = 0.0;
    let max_value = 1.0;

    // Define the range of Unicode characters to use.
    let min_unicode: u32 = 0x2588; // U+2588 FULL BLOCK
    let max_unicode: u32 = 0x2591; // U+2591 LIGHT SHADE

    let unicode_range = max_unicode - min_unicode + 1;

    // Map the input data to Unicode characters.
    let unicode_data: Vec<Vec<char>> = data
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| {
                    if value < min_value {
                        std::char::from_u32(min_unicode).unwrap()
                    } else if value > max_value {
                        std::char::from_u32(max_unicode).unwrap()
                    } else {
                        let unicode_value = min_unicode
                            + (unicode_range as f64 * (value - min_value) / (max_value - min_value))
                                as u32;
                        std::char::from_u32(unicode_value).unwrap()
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
    pub fn new(t: T) -> Self {
        Self(t)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for NoPrettyPrint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Prevent "{:#?}" from being used.
        write!(f, "{:?}", self.0)
    }
}

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
    (a, b): (f64, f64),
) -> f64 {
    0.5 * gauss_legendre_quadrature(|x| (p(x) - q(x)).abs(), n, (a, b))
}
