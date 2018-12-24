extern crate ndarray;

use ndarray::Array;

use crate::core::tensor::Tensor;

pub struct Session {
	number: u32
}

impl Session {
	fn new(){
		Session {number: 1}
	}

}
