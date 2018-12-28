use std::collections::HashMap;
use crate::core::tensor::Tensor;
use crate::core::types::RBArray;

pub struct Session<'a> {
	feeds:      HashMap<&'a Tensor<'a>, &'a RBArray>,
	comps:      HashMap<&'a Tensor<'a>, RBArray>,
	comp_stack: Vec<&'a Tensor<'a>>,
}
impl<'a> Session<'a> {
	pub fn new() -> Session<'a> {
		Session {feeds: HashMap::new(), comps: HashMap::new(), comp_stack: vec![]}
	}

	pub fn run(&mut self, feeds: HashMap<&'a Tensor, &'a RBArray>, fetches: Vec<&'a Tensor>) -> Vec<RBArray> {
		
		self.feeds = feeds;
		self.comp_stack.clear();
		self.comps.clear();

		let mut evaluations: Vec<RBArray> = vec![];
		//Invariants
		// Pre-execution: evaluations is empty
		// After an iteration: evaluations has at its tail the evaluation of the fetch
		for fetch in fetches.iter() {
			if let Some(&feed) = self.feeds.get(fetch) {
				eprintln!("WARNING: You are fetching {:?}, which you already feeded.", fetch);
				evaluations.push(feed.clone());
				continue;
			}
			self.comp_stack.push(fetch);
			self.push_preds(fetch);


			//Invariant: the top element of the stack already has its predecessors computed
			//Invariant: After the loop has been executed, fetch's value is in comps
			while self.comp_stack.len() > 0 {
				let ready_comp = self.comp_stack.pop().unwrap();

				//The Tensor was already computed earlier
				if None == self.extract_val(ready_comp) {continue;}

				let result: RBArray = {
					let mut pred_vals: Vec<&RBArray> = vec![];
					for pred in ready_comp.preds.iter() {
						let pred_value = self.extract_val(pred);
						assert_ne!(pred_value, None);
						pred_vals.push(pred_value.unwrap());
					}
					(ready_comp.eval)(&pred_vals)
				};
				self.comps.insert(ready_comp, result);
			}
		}

		//Second pass: Move the fetch values from comps to evaluations
		for fetch in fetches.iter() {
			let fetch_value = self.comps.remove(fetch);
			assert_ne!(fetch_value, None);
			evaluations.push(fetch_value.unwrap());

		}
		evaluations
	}

	///Determines whether x has already been feeded or computed
	fn extract_val(&'a self, x: &'a Tensor) -> Option<&'a RBArray> {

		if let Some(&feed) = self.feeds.get(&x) {
			return Some(feed);
		}
		else if let Some(comp) = self.comps.get(&x) {
			return Some(comp);
		}
		else {
			return None;
		}
	}

	///Push all predecessors of x, direct or indirect, to the comp_stack
	fn push_preds(&mut self, x: &'a Tensor) {
		for pred in x.preds.iter() {
			if None == self.extract_val(pred){
				self.comp_stack.push(pred);
				self.push_preds(pred);
			}
		}
	}
	

}

