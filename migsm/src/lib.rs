pub mod data;
pub mod dpmm;
pub mod mcmc;
pub mod misa;

pub mod models;

pub mod utils;

/// Lens to an inner value to a struct or tuple.
pub trait Lens<S, X> {
    /// Get the lensed value
    fn get<'b, 'a: 'b>(&'b self, state: &'a S) -> &'a X;
    /// Set the lensed value
    fn set(&self, state: S, x: X) -> S;
}

/// The identity lens for accessing an object itself.
pub struct IdentityLens;

impl<X> Lens<X, X> for IdentityLens {
    fn get<'b, 'a: 'b>(&'b self, state: &'a X) -> &'a X {
        state
    }

    fn set(&self, _state: X, x: X) -> X {
        x
    }
}

impl<'c, M, X, G, S> Lens<M, X> for (G, S)
where
    G: Fn(&M) -> &X,
    S: Fn(M, X) -> M,
    M: 'c,
    X: 'c,
{
    fn set(&self, state: M, x: X) -> M {
        self.1(state, x)
    }

    fn get<'b, 'a: 'b>(&'b self, state: &'a M) -> &'a X {
        self.0(state)
    }
}

