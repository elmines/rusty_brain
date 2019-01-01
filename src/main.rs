extern crate rusty_brain;
use rusty_brain::Tensor;
use rusty_brain::Session;

#[macro_use(array)]
extern crate ndarray;
use ndarray::{ArrayD, Array};

use std::collections::HashMap;

use rusty_brain::utils::into_dynamic;

fn close(x: ArrayD<f32>, y: ArrayD<f32>) -> bool {
	(x - y).fold(true, |accum, element| accum && (element.abs() < 0.01) )
}

fn main() {

	let a = Tensor::placeholder(vec![2, 2], Some(String::from("a")));
	let b = Tensor::placeholder(vec![2], Some(String::from("b")));
	let c = Tensor::placeholder(vec![2, 2, 2], Some(String::from("c")));
	let d = Tensor::placeholder(vec![2, 2], Some(String::from("d")));

	let sum = &b + &a;
	let product = &sum * &a;
	let quotient = &c / &product;
	let difference = &d - &c;

	let a_feed = into_dynamic(array![[1., 2.], [3., 4.]]);
	let b_feed = into_dynamic(array![ 5., 6. ]);
	let c_feed: ArrayD<f32> = into_dynamic( Array::ones( (2, 2, 2) ) );
	let d_feed: ArrayD<f32> = into_dynamic( Array::zeros( (2, 2) ) );


	let mut feeds: HashMap<&Tensor, &ArrayD<f32>> = HashMap::new();
	feeds.insert(&a, &a_feed);
	feeds.insert(&b, &b_feed);
	feeds.insert(&c, &c_feed);
	feeds.insert(&d, &d_feed);

	let fetches = vec![&sum, &product, &quotient, &difference];
	let expected_results = vec![
		into_dynamic(array![[6., 8.], [8., 10.]]),
		into_dynamic(array![[6., 16.], [24., 40.]]),

		into_dynamic(array![  [[1./6., 1./16.], [1./24., 1./40.]],
                                      [[1./6., 1./16.], [1./24., 1./40.]]
		]),

		into_dynamic(array![ [[-1., -1.], [-1., -1.]],
				     [[-1., -1.], [-1., -1.]]
		])

	];

	//let mut sess = Session::new();
	let results = Session::new().run(feeds, fetches);

	for (result, expected) in results.into_iter().zip(expected_results) {
		assert!(close(result, expected));
	}

}
