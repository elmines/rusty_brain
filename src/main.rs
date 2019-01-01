extern crate rusty_brain;
use rusty_brain::Tensor;
use rusty_brain::Session;

#[macro_use(array)]
extern crate ndarray;
use ndarray::ArrayD;

use std::collections::HashMap;

use rusty_brain::utils::into_dynamic;

fn main() {

	let x = Tensor::placeholder(vec![3], Some(String::from("x")));
	let y = Tensor::placeholder(vec![1], Some(String::from("y")));
	let sum = &y + &x;
	let difference = &y - &x;
	let product = &y * &x;
	let quotient = &y / &x;

	let x_feed = into_dynamic(array![ [5., 2.] ]);
	let y_feed = into_dynamic(array![ 3. ]);


	let mut feeds: HashMap<&Tensor, &ArrayD<f32>> = HashMap::new();
	feeds.insert(&x, &x_feed);
	feeds.insert(&y, &y_feed);
	let fetches: Vec<&Tensor> = vec![&sum, &difference, &product, &quotient];

	let mut sess = Session::new();
	let results: Vec<ArrayD<f32>> = sess.run(feeds, fetches);

	println!("x =\n{}\n", x_feed);
	println!("y =\n{}\n", y_feed);

	for result in results.iter() {
		println!("{:?}", result);
	}

}
