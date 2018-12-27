use std::collections::HashMap;

use crate::core::tensor::Tensor;
use crate::core::types::RBArray;

///A pointer to a function taking in the concrete operands of a Tensor, and returning the concrete result
pub type EvalFunc = fn(&Vec<&RBArray>) -> RBArray;

pub fn eval_placeholder(_operands: &Vec<&RBArray>) -> RBArray {
	panic!("You failed to feed a placeholder.");
}

pub fn eval_reversed_mul(operands: &Vec<&RBArray>) -> RBArray {
	if operands.len() != 2 {
		panic!("Tried to perform a multiplication operation on {} operands rather than 2.", operands.len())
	}
	eval_mul(&vec![operands[1], operands[0]])
}

pub fn eval_mul(operands: &Vec<&RBArray>) -> RBArray {

	if operands.len() != 2 {
		panic!("Tried to perform a multiplication operation on {} operands rather than 2.", operands.len())
	}
	operands[0] * operands[1]
}

/*
	///Compute the dot (or inner) product of two Tensors
	///
	///Panics if the last dimension of self and the first dimension of x aren't equal
	pub fn dot(&self, x: &Tensor) -> Tensor {
		if self.last_dim() != x.first_dim() {
			panic!("Cannot compute inner product of lefthand Tensor of shape {:?} and righthand Tensor of shape {:?}. {} != {}",
				self.shape, x.shape, self.last_dim(), x.first_dim())
		}

		let mut shape: Vec<u64> = vec![];
		for &dim in (&self.shape[0..self.shape.len()-1]).iter() { shape.push(dim); }
		for &dim in (&x.shape[1..]).iter() { shape.push(dim); }
		if shape.len() < 1 { shape.push(1) }; //The inner product is a scalar

		Tensor::placeholder(shape)
	}
*/

