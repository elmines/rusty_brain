extern crate rusty_brain;
use rusty_brain::Tensor;
use rusty_brain::RBArray;
use rusty_brain::Session;

#[macro_use(array)]
extern crate ndarray;
use ndarray::ArrayD;
use ndarray::IxDyn;

use std::collections::HashMap;

fn main() {

	let x = Tensor::placeholder(vec![3], Some(String::from("x")));
	let y = Tensor::placeholder(vec![1], Some(String::from("y")));
	let z = &y * &x;

	println!("z = {:?}", z);

	let mut x_feed: RBArray = ArrayD::<f32>::zeros(IxDyn(&[3]));
	x_feed[0] = 5.; x_feed[1] = 2.;

	let mut y_feed: RBArray = ArrayD::<f32>::zeros(IxDyn(&[1]));
	y_feed[0] = 3.;

	let mut feeds: HashMap<&Tensor, &RBArray> = HashMap::new();

	feeds.insert(&x, &x_feed);
	feeds.insert(&y, &y_feed);
	let fetches: Vec<&Tensor> = vec![&z];

	let mut sess = Session::new();
	let results: Vec<RBArray> = sess.run(feeds, fetches);

	println!("{:?}", results);

}
