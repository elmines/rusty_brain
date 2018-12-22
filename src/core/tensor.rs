use std::ops::Mul;

#[derive(Debug)]
pub struct Tensor {
	pub shape: Vec<u64>,
	preds: Vec<Tensor>
}


impl Tensor {
	///Construct a Tensor from the given known shape
	pub fn new(shape: Vec<u64>) -> Tensor {
		Tensor {shape, preds: vec![]}
	}

	///Access the first dimension (it can make your code more readable, okay?)
	pub fn first_dim(&self) -> u64 {self.shape[0]}

	///Access the last dimension (sounds like something from science fiction, doesn't it?)
	pub fn last_dim(&self) -> u64 {self.shape[self.shape.len()-1]}

	pub fn dot(self, x: Tensor) -> Tensor {
		if self.last_dim() != x.first_dim() {
			panic!("Cannot compute inner product of lefthand Tensor of shape {:?} and righthand Tensor of shape {:?}. {} != {}",
				self.shape, x.shape, self.last_dim(), x.first_dim())
		}

		self //FIXME
	}

}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	#[should_panic]
	fn bad_mul_dims() {
		let a = Tensor::new(vec![32, 5, 6]);
		let b = Tensor::new(vec![5, 12]);
		let c = a.dot(b);
	}
}


