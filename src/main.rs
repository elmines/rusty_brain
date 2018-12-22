extern crate rusty_brain;
//use rusty_brain::core::Tensor;
use rusty_brain::Tensor;


fn main() {
	//let x: Tensor<f32> = Tensor {shape: vec![32, 5, 6], value: 3.5};
	let a = Tensor::new(vec![32, 5, 6]);
	let b = Tensor::new(vec![6, 8]);

	let c = a.dot(b);

	let d = Tensor::new(vec![4]);
	let e = c.dot(d);
}
