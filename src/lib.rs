mod core;
pub use core::*;

pub mod layers;
//pub use layers;

pub mod utils;

#[macro_use(array)]
extern crate ndarray;
