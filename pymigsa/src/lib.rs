use std::num::NonZeroUsize;

use ::migsa::mcmc::samplers::partition::gibbs::PartitionGibbs;
use ::migsa::mcmc::Sampler;
use ::migsa::models::mixture::ConjugateMixtureModel;
use nalgebra::{DMatrix, DVector, Matrix};
use numpy::{
    PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;
use pyo3::{prelude::*, Bound};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rv::dist::{Gaussian, Mixture, NormalGamma};
use rv::traits::Rv;

#[pyclass]
#[derive(Debug)]
pub struct Igmm {
    mixture_model: Option<ConjugateMixtureModel<f64, Gaussian, NormalGamma>>,
    rng: rand::rngs::SmallRng,
    sampler: PartitionGibbs,
    data: Vec<f64>,
}

#[pymethods]
impl Igmm {
    #[new]
    #[pyo3(signature = (data, seed = None))]
    fn new(data: Vec<f64>, seed: Option<u64>) -> Self {
        let mut rng = seed.map_or_else(
            || rand::rngs::SmallRng::from_entropy(),
            |s| rand::rngs::SmallRng::seed_from_u64(s),
        );
        Self {
            mixture_model: Some(ConjugateMixtureModel::new(
                NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0),
                data.iter(),
                1.0,
                &mut rng,
            )),
            rng,
            sampler: PartitionGibbs::new(),
            data,
        }
    }

    fn step(&mut self) -> GaussianMixture {
        let next_model = self.sampler.step(
            self.mixture_model.take().unwrap(),
            &self.data,
            &mut self.rng,
        );

        let mixture: Mixture<Gaussian> = next_model.draw(&mut self.rng);

        self.mixture_model = Some(next_model);

        GaussianMixture(mixture)
    }
}

#[pyclass]
#[derive(Debug)]
pub struct GaussianMixture(Mixture<Gaussian>);

#[pymethods]
impl GaussianMixture {
    fn weights(&self) -> Vec<f64> {
        self.0.weights().to_owned()
    }

    fn p(&self, x: f64) -> f64 {
        self.0.f(&x)
    }
}

/// Borgonovo Indicies from DPMM density estimation.
///
/// Parameters
/// ==========
/// * `func` - Scalar model function.
/// * `means` - Mean of parameter distributions.
/// * `cov` - Covariance matrix for parameter distribution.
/// * `n_evals` - Number of evaluations of the model function.
/// * `n` - Number of MCMC iterations to take.
/// * `warmup` - Number of MCMC iterations to discard at startup.
/// * `thinning` - Number of MCMC iterations to discard between steps.
/// * `seed` - An optional seed for the rng.
#[pyfunction]
fn single_loop_deltas<'a>(
    py: Python<'a>,
    func: &'a Bound<PyAny>,
    means: &'a Bound<PyArray1<f64>>,
    cov: &'a Bound<PyArray2<f64>>,
    n_evals: usize,
    n: usize,
    warmup: usize,
    thinning: NonZeroUsize,
    seed: Option<&'a Bound<PyBytes>>,
) -> PyResult<Bound<'a, PyArray<f64, numpy::ndarray::Dim<[usize; 2]>>>> {
    let means = means.try_readonly()?;
    let means: DVector<f64> = DVector::from(means.as_matrix().column(0));

    //dbg!(&means);

    let cov = cov.try_readonly()?;
    let cov: DMatrix<f64> = DMatrix::from(cov.as_matrix());

    //dbg!(&cov);

    let parameter_dist = rv::dist::MvGaussian::new(means, cov).map_err(|err| {
        PyValueError::new_err(format!(
            "Invalid arguments for Multivariate Gaussian: {err}"
        ))
    })?;

    let mut rng = seed.map_or_else(
        || StdRng::from_entropy(),
        |seed| {
            let bytes = seed.as_bytes();

            let mut seed: [u8; 32] = [0; 32];
            let n = std::cmp::min(seed.len(), bytes.len());
            seed[0..n].copy_from_slice(&bytes[0..n]);
            StdRng::from_seed(seed)
        },
    );

    let f = |x: &DVector<f64>| -> f64 {
        func.call1((x.to_pyarray_bound(py),))
            .unwrap()
            .extract()
            .unwrap()
    };

    let deltas = ::migsa::misa::single_loop_deltas(
        parameter_dist,
        f,
        n_evals,
        n,
        warmup,
        thinning,
        &mut rng,
    );

    let m = deltas.first().unwrap().len();
    let out = DMatrix::from_row_iterator(n, m, deltas.into_iter().flat_map(|x| x.into_iter()));

    Ok(out.to_pyarray_bound(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn migsa(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Igmm>()?;
    m.add_class::<GaussianMixture>()?;
    m.add_function(wrap_pyfunction!(single_loop_deltas, m)?)?;
    Ok(())
}
