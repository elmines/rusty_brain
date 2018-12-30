extern crate rusty_brain;
use rusty_brain::Tensor;
use rusty_brain::RBArray;
use rusty_brain::Session;

#[macro_use(array)]
extern crate ndarray;
use ndarray::ArrayD;
use ndarray::IxDyn;

use std::collections::HashMap;

use rusty_brain::utils::into_dynamic;

fn main() {

	let x = Tensor::placeholder(vec![3], Some(String::from("x")));
	let y = Tensor::placeholder(vec![1], Some(String::from("y")));
	let z = &y * &x;

	println!("z = {:?}", z);

	let x_feed = into_dynamic(array![ [5., 2.] ]);
	let y_feed = into_dynamic(array![ 3. ]);


	let mut feeds: HashMap<&Tensor, &RBArray> = HashMap::new();
	feeds.insert(&x, &x_feed);
	feeds.insert(&y, &y_feed);
	let fetches: Vec<&Tensor> = vec![&z];

	let mut sess = Session::new();
	let results: Vec<RBArray> = sess.run(feeds, fetches);

	println!("{:?}", results);

}
