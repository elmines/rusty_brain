use crate::ndarray::ArrayD;

pub fn eval_reversed_add(operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	binary_op_check(operands, "addition");
	eval_add(&vec![operands[1], operands[0]])
}
pub fn eval_add(operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	binary_op_check(operands, "addition");
	operands[0] + operands[1]
}
pub fn eval_reversed_sub(operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	binary_op_check(operands, "subtraction");
	-operands[1] + operands[0]
}
pub fn eval_sub(operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	binary_op_check(operands, "subtraction");
	operands[0] - operands[1]
}
pub fn eval_reversed_mul(operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	binary_op_check(operands, "multiplication");
	eval_mul(&vec![operands[1], operands[0]])
}
pub fn eval_mul(operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	binary_op_check(operands, "multiplication");
	operands[0] * operands[1]
}
pub fn eval_reversed_div(operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	binary_op_check(operands, "division");
	eval_div(&vec![operands[1], operands[0]])
}
pub fn eval_div(operands: &Vec<&ArrayD<f32>>) -> ArrayD<f32> {
	binary_op_check(operands, "division");
	operands[0] / operands[1]
}

fn binary_op_check(operands: &Vec<&ArrayD<f32>>, op_name: &'static str) {
	if operands.len() != 2 {
		panic!("Tried to perform {} on {} operands rather than 2.", op_name, operands.len())
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[macro_use(array)]	
	use crate::ndarray::Array;
	use crate::utils::into_dynamic;


	#[test]
	#[should_panic(expect="Tried to perform addition on 3 operands rather than 2.")]
	fn bad_num_ops() {
		let x = into_dynamic(Array::ones((5, 5)));
		let y = into_dynamic(Array::ones((5, 5)));
		let z = into_dynamic(Array::ones((5, 5)));
		let operands = vec![&x, &y, &z];

		binary_op_check(&operands, "addition");
	}

}
