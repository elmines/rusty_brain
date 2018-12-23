use std::ops::Mul;

#[derive(Debug)]
pub struct Tensor {
	pub shape: Vec<u64>,
	//preds: Vec<&'a Tensor<'a>>
}


impl Tensor {
	///Construct a Tensor from the given known shape
	pub fn new(shape: Vec<u64>) -> Tensor {
		Tensor {shape}
	}

	///Access the first dimension (it can make your code more readable, okay?)
	pub fn first_dim(&self) -> u64 {self.shape[0]}
	///Access the last dimension (sounds like something from science fiction, doesn't it?)
	pub fn last_dim(&self) -> u64 {self.shape[self.shape.len()-1]}

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

		Tensor::new(shape)
	}

}

fn broadcast(l: &Tensor, r: &Tensor) -> Vec<u64> {
	let mut shape = vec![];
	let (l_shape, r_shape) = (l.shape.clone(), r.shape.clone());//( &(l.shape), &(r.shape) );
	let mut i = l_shape.len();
	let mut j = r_shape.len();
	while i > 0 && j > 0 {
		let dim = match (l_shape[i-1], r_shape[j-1]) {

			(1, 1) => 1,
			(x, 1) | (1, x) => x,
			(x, y) => {
					if x != y {
						panic!("Failed to broadcast Tensors of shapes {:?} and {:?} since {} != {}.",
						l_shape, r_shape, l_shape[i-1], r_shape[j-1]);
					}
					x
			}
		};
		shape.insert(0, dim);
		i -= 1;
		j -= 1;
	}

	while i > 0 {shape.insert(0, l_shape[i-1]); i -= 1;}
	while j > 0 {shape.insert(0, r_shape[j-1]); j -= 1;}
	shape
}

impl<'a> Mul<&'a Tensor> for &'a Tensor {
	type Output = Tensor;

	fn mul(self, rhs: &'a Tensor) -> Tensor {

		let shape = broadcast(self, rhs);

		Tensor::new(shape)
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	#[should_panic]
	fn bad_dot_dims() {
		let a = Tensor::new(vec![32, 5, 6]);
		let b = Tensor::new(vec![5, 12]);
		let c = a.dot(&b);
	}

	#[test]
	fn dot_dims() {
		let a = Tensor::new(vec![32, 5, 6]);
		let b = Tensor::new(vec![6, 48, 5]);
		let c = a.dot(&b);
		assert_eq!(c.shape, vec![32, 5, 48, 5]);

		let d = Tensor::new(vec![5]);
		let e = Tensor::new(vec![5]);
		let f = d.dot(&e);
		assert_eq!(f.shape, vec![1]);
	}

	#[test]
	fn broad_dims() {
		let a = Tensor::new(vec![20, 50, 5]);
		let b = Tensor::new(vec![20, 50, 5]);
		assert_eq!(broadcast(&a, &b), vec![20, 50, 5]);

		let c = Tensor::new(vec![1, 5]);
		assert_eq!(broadcast(&a, &c), vec![20, 50, 5]);

		let d = Tensor::new(vec![80, 20, 1, 5]);
		assert_eq!(broadcast(&a, &d), vec![80, 20, 50, 5]);

		let scalar = Tensor::new(vec![1]);
		assert_eq!(broadcast(&a, &scalar), vec![20, 50, 5]);

		let e = Tensor::new(vec![1, 25, 30, 40, 20, 1, 80]);
		let f = Tensor::new(vec![       30,  1,  1, 1, 80]);
		assert_eq!(broadcast(&e, &f), vec![1, 25, 30, 40, 20, 1, 80]);

	}

	#[test]
	#[should_panic(expected = "Failed to broadcast Tensors of shapes [12, 10, 80, 1, 1, 2] and [12, 20, 80, 4, 1, 1] since 10 != 20.")]
	fn bad_broad_dims() {
		let a = Tensor::new(vec![12, 10, 80, 1, 1, 2]);
		let b = Tensor::new(vec![12, 20, 80, 4, 1, 1]);
		let broad_shape = broadcast(&a, &b);
	}
}
