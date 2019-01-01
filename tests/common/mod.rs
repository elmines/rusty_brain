extern crate ndarray;
use ndarray::{ArrayD};

pub fn close(x: ArrayD<f32>, y: ArrayD<f32>) -> bool {
	(x - y).fold(true, |accum, element| accum && (element.abs() < 0.01) )
}
