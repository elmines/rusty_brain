use std;
use crate::core::ops;
use crate::core::ops::{eval_add, eval_reversed_add,
			eval_sub, eval_reversed_sub,
			eval_mul, eval_reversed_mul,
			eval_div, eval_reversed_div
			};
use crate::ndarray::ArrayD;

pub struct Tensor<'a> {
	shape: Vec<u64>,
	id: u128,
	name: String,
	preds_list: Vec<&'a Tensor<'a>>,
	eval_fn: ops::EvalFunc
}
//TODO: Use lifetime subtyping to let preds outlive Tensor

impl<'a> Tensor<'a> {
	///Construct a Tensor from the given known shape
	pub fn placeholder(shape: Vec<u64>, name: Option<String>) -> Tensor<'a> {

		let name_val = if let Some(val) = name {val} else {String::from("placeholder")};

		Tensor {shape, id: 0, name: name_val, preds_list: vec![], eval_fn: ops::eval_placeholder}
	}

	pub fn eval(&self, operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32>{ (self.eval_fn)(operands) }

	pub fn preds(&self) -> &Vec<&Tensor> {&(self.preds_list)}

}

fn reverse_operands(l_shape: &Vec<u64>, r_shape: &Vec<u64>) -> bool {
	l_shape.len() < r_shape.len()
}

macro_rules! binary_op{
	($trait: ident, $method: ident, $normal: ident, $reversed: ident) => {

		impl<'a> std::ops::$trait<&'a Tensor<'a>> for &'a Tensor<'a> {
			type Output = Tensor<'a>;
		
			fn $method(self, rhs: &'a Tensor<'a>) -> Tensor<'a> {
				let reverse = reverse_operands(&self.shape, &rhs.shape);

		
				let id = std::cmp::max(self.id, rhs.id);
				let preds_list: Vec<&Tensor> = vec![self, rhs];
		
				let shape =   if reverse { broadcast(rhs, &self) } else { broadcast(&self, rhs) };
				let eval_fn = if reverse { $reversed } else { $normal };
		
				Tensor {shape, id, name: String::from("product"), preds_list, eval_fn}
			}
		}
	}
}
binary_op!(Add, add, eval_add, eval_reversed_add);
binary_op!(Sub, sub, eval_sub, eval_reversed_sub);
binary_op!(Mul, mul, eval_mul, eval_reversed_mul);
binary_op!(Div, div, eval_div, eval_reversed_div);

//Common trait implementations
impl<'a> std::cmp::PartialEq for &'a Tensor<'a> {
	fn eq(&self, other: &&Tensor) -> bool {
		std::ptr::eq(*self, *other)
	}
}
impl<'a> std::cmp::Eq for &'a Tensor<'a> {}
impl<'a> std::hash::Hash for &'a Tensor<'a> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		self.name.hash(state);
	}
}
impl<'a> std::fmt::Debug for Tensor<'a> {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Tensor {{ shape: {:?}, name: \"{}\" }}", self.shape, self.name)
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
	use std::hash::{Hash, Hasher};

	#[test]
	fn debug_trait() {
		let x = Tensor::placeholder(vec![5, 28, 4], None);
		let formatted_x = format!("{:?}", x);
		assert_eq!(String::from("Tensor { shape: [5, 28, 4], name: \"placeholder\" }"), formatted_x);

		let y = Tensor::placeholder(vec![1], Some(String::from("why")));
		let formatted_y = format!("{:?}", y);
		assert_eq!(String::from("Tensor { shape: [1], name: \"why\" }"), formatted_y);
	}


	#[test]
	fn tensor_eq() {
		let a =       Tensor::placeholder(vec![3, 3, 3], Some(String::from("a")));
		let other_a = Tensor::placeholder(vec![3, 3, 3], Some(String::from("a")));
		assert_ne!(&a, &other_a);

		let c = &a;
		assert_eq!(&a, c);	

	}

	#[test]
	fn tensor_hash() {
		let a = Tensor::placeholder(vec![5, 5, 5], Some(String::from("a placeholder")));
		let b = Tensor::placeholder(vec![3], Some(String::from("a placeholder")));
		let c = Tensor::placeholder(vec![5, 5, 5], Some(String::from("more than just a placeholder")));

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
	fn broad_dims() {
		let a = Tensor::placeholder(vec![20, 50, 5], None);
		let b = Tensor::placeholder(vec![20, 50, 5], None);
		assert_eq!(broadcast(&a, &b), vec![20, 50, 5]);

		let c = Tensor::placeholder(vec![1, 5], None);
		assert_eq!(broadcast(&a, &c), vec![20, 50, 5]);

		let d = Tensor::placeholder(vec![80, 20, 1, 5], None);
		assert_eq!(broadcast(&a, &d), vec![80, 20, 50, 5]);

		let scalar = Tensor::placeholder(vec![1], None);
		assert_eq!(broadcast(&a, &scalar), vec![20, 50, 5]);

		let e = Tensor::placeholder(vec![1, 25, 30, 40, 20, 1, 80], None);
		let f = Tensor::placeholder(vec![       30,  1,  1, 1, 80], None);
		assert_eq!(broadcast(&e, &f), vec![1, 25, 30, 40, 20, 1, 80]);

	}

	#[test]
	#[should_panic(expected = "Failed to broadcast Tensors of shapes [12, 10, 80, 1, 1, 2] and [12, 20, 80, 4, 1, 1] since 10 != 20.")]
	fn bad_broad_dims() {
		let a = Tensor::placeholder(vec![12, 10, 80, 1, 1, 2], None);
		let b = Tensor::placeholder(vec![12, 20, 80, 4, 1, 1], None);
		let broad_shape = broadcast(&a, &b);
	}

}
