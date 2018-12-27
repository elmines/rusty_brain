extern crate ndarray;
use self::ndarray::Array;

use std::collections::HashMap;

use crate::core::tensor::Tensor;



fn push_preds(comp_stack: &mut Vec<&Tensor>, x: &Tensor) {
	for pred in x.preds().iter() { comp_stack.push(pred); }
}

fn extract_pred(x: &Tensor, feeds: &HashMap<&Tensor, &Array<f32, f64>>, fetches: Vec<&Tensor>) => {

	match (feeds.get(x), fetches.get(x)) {

		(Some(feed), _

	}

}

pub struct Session {}
impl Session {
	fn new() -> Session {
		Session {}
	}


	fn run(&self, feeds: &HashMap<&Tensor, &Array<f32, u64>>, fetches: Vec<&Tensor>) -> Vec<Array<f32, u64>> {
		
		let mut comps: HashMap<&Tensor, Array<f32, u64>> = HashMap::new();
		let mut evaluations: Vec<Array<f32, u64>> = vec![];

		let mut fetch_stack: Vec<&Tensor> = vec![];

		for fetch in fetches.iter() { fetch_stack.push(fetch); }

		//Invariants
		// Pre-execution: evaluations is empty
		// After an iteration: evaluations has at its tail the evaluation of the fetch

		for fetch in fetches.iter() {

			if Some(&feed) = feeds.get(fetch) {
				eprintln!("WARNING: You are fetching {:?}, which you already feeded.", fetch);
				evaluations.push(feed.clone());
				continue;
			}

			let mut comp_stack: Vec<&Tensor> = vec![fetch];

			while comp_stack.len() > 0 {

				let mut next_comp = comp_stack[comp_stack.len() - 1];

				while next_comp.preds.len() > 0 {
					//Check for a feed made by an intermediate computation
					if let None = feeds.get(next_comp) {
						push_preds(&mut comp_stack, next_comp);

					}
				}

				while comp_stack[comp_stack.len() - 1] != next_comp {

					//All the Tensors predecessors have been computed
					let ready_comp = comp_stack.pop().unwrap();

					//The Tensor was already computed somewhere else
					if let Some(feed) = feeds.get(ready_comp) | let Some(comp) = comps.get(ready_comp) {continue;}

					let pred_vals = vec![];
					for pred in ready_comp.preds() {
						pred_vals.push( 

				}

			}
		
		}

		evaluations
	}

}

