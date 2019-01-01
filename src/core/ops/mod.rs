mod binary;
pub use self::binary::*;

use crate::ndarray::ArrayD;

///A pointer to a function taking in the concrete operands of a Tensor, and returning the concrete result
pub type EvalFunc = fn(&Vec<&ArrayD<f32>>) -> ArrayD<f32>;

///Dummy evaluation function for placeholder Tensors
///
///This function will only be called if the client fails to feed the placeholder.
///Thus, this function always panics.
pub fn eval_placeholder(_operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	panic!("You failed to feed a placeholder.");
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::ndarray::Array;
	use crate::utils::into_dynamic;

	#[test]
	#[should_panic(expect="You failed to feed a placeholder.")]
	fn placeholder() {
		let fake_operand = into_dynamic(Array::ones((5, 5)));
		eval_placeholder(&vec![&fake_operand]);		
	}

}
