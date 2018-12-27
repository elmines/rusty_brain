use std;
use crate::core::ops;

pub struct Tensor<'a> {
	pub shape: Vec<u64>,
	pub id: u128,
	pub preds: Vec<&'a Tensor<'a>>,
	pub eval: ops::EvalFunc
}

impl<'a> Tensor<'a> {
	///Construct a Tensor from the given known shape
	pub fn placeholder(shape: Vec<u64>) -> Tensor<'a> {
		Tensor {shape, id: 0, preds: vec![], eval: ops::eval_placeholder}
	}
}

impl<'a> std::ops::Mul<&'a Tensor<'a>> for &'a Tensor<'a> {
	type Output = Tensor<'a>;

	fn mul(self, rhs: &'a Tensor<'a>) -> Tensor<'a> {
		let shape = broadcast(self, rhs);
		let id = std::cmp::max(self.id, rhs.id);
		let preds: Vec<&Tensor> = vec![self, rhs];
		let eval = if self.shape.len() < rhs.shape.len() {ops::eval_reversed_mul} else {ops::eval_mul};

		Tensor {shape, id, preds, eval}
	}
}

//Common trait implementations
impl<'a> std::cmp::PartialEq for &'a Tensor<'a> {
	fn eq(&self, other: &&Tensor) -> bool {
		std::ptr::eq(*self, *other)
	}
}
impl<'a> std::cmp::Eq for &'a Tensor<'a> {}
impl<'a> std::hash::Hash for &'a Tensor<'a> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		self.id.hash(state);
	}
}
impl<'a> std::fmt::Debug for Tensor<'a> {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Tensor {{ shape: {:?} }}", self.shape)
	}
}


pub fn broadcast(l: &Tensor, r: &Tensor) -> Vec<u64> {
	let mut shape = vec![];
	let (l_shape, r_shape) = (l.shape.clone(), r.shape.clone());
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



#[cfg(test)]
mod tests {
	use super::*;

	use std::collections::hash_map::DefaultHasher;

	#[test]
	fn debug_trait() {
		let x = Tensor::placeholder(vec![5, 28, 4]);
		let formatted = format!("{:?}", x);
		assert_eq!(String::from("Tensor { shape: [5, 28, 4] }"), formatted);
	}

/*

	#[test]
	fn tensor_hash() {
		let a = Tensor::computation(vec![5, 5, 5], 0);
		let b = Tensor::computation(vec![3], 0);
		let c = Tensor::computation(vec![5, 5, 5], 1);

		let mut hasher_a = DefaultHasher::new();
		let mut hasher_b = DefaultHasher::new();
		let mut hasher_c = DefaultHasher::new();

		(&a).hash(&mut hasher_a);
		(&b).hash(&mut hasher_b);
		(&c).hash(&mut hasher_c);

		let a_hash = hasher_a.finish();
		let b_hash = hasher_b.finish();
		let c_hash = hasher_c.finish();

		assert_eq!(a_hash, b_hash);
		assert_ne!(a_hash, c_hash);
		assert_ne!(b_hash, c_hash);
	}

	#[test]
	fn tensor_eq() {
		let a = Tensor::computation(vec![3, 3, 3], 0);
		let b = Tensor::computation(vec![3, 3, 3], 0);
		assert_ne!(&a, &b);

		let c = &a;
		assert_eq!(&a, c);	

	}
*/


	#[test]
	#[should_panic]
	fn bad_dot_dims() {
		let a = Tensor::placeholder(vec![32, 5, 6]);
		let b = Tensor::placeholder(vec![5, 12]);
		let c = a.dot(&b);
	}

	#[test]
	fn dot_dims() {
		let a = Tensor::placeholder(vec![32, 5, 6]);
		let b = Tensor::placeholder(vec![6, 48, 5]);
		let c = a.dot(&b);
		assert_eq!(c.shape, vec![32, 5, 48, 5]);

		let d = Tensor::placeholder(vec![5]);
		let e = Tensor::placeholder(vec![5]);
		let f = d.dot(&e);
		assert_eq!(f.shape, vec![1]);
	}

	#[test]
	fn broad_dims() {
		let a = Tensor::placeholder(vec![20, 50, 5]);
		let b = Tensor::placeholder(vec![20, 50, 5]);
		assert_eq!(broadcast(&a, &b), vec![20, 50, 5]);

		let c = Tensor::placeholder(vec![1, 5]);
		assert_eq!(broadcast(&a, &c), vec![20, 50, 5]);

		let d = Tensor::placeholder(vec![80, 20, 1, 5]);
		assert_eq!(broadcast(&a, &d), vec![80, 20, 50, 5]);

		let scalar = Tensor::placeholder(vec![1]);
		assert_eq!(broadcast(&a, &scalar), vec![20, 50, 5]);

		let e = Tensor::placeholder(vec![1, 25, 30, 40, 20, 1, 80]);
		let f = Tensor::placeholder(vec![       30,  1,  1, 1, 80]);
		assert_eq!(broadcast(&e, &f), vec![1, 25, 30, 40, 20, 1, 80]);

	}

	#[test]
	#[should_panic(expected = "Failed to broadcast Tensors of shapes [12, 10, 80, 1, 1, 2] and [12, 20, 80, 4, 1, 1] since 10 != 20.")]
	fn bad_broad_dims() {
		let a = Tensor::placeholder(vec![12, 10, 80, 1, 1, 2]);
		let b = Tensor::placeholder(vec![12, 20, 80, 4, 1, 1]);
		let broad_shape = broadcast(&a, &b);
	}
}
