extern crate rusty_brain;
//use rusty_brain::core::Tensor;
use rusty_brain::Tensor;


fn main() {
	//let x: Tensor<f32> = Tensor {shape: vec![32, 5, 6], value: 3.5};
	let a = Tensor::placeholder(vec![32, 5, 6]);
	let b = Tensor::placeholder(vec![6, 8]);

}
