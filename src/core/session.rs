use std::collections::HashMap;
use crate::core::tensor::Tensor;
use crate::core::types::RBArray;

pub struct Session {
	mut feeds: &HashMap<&Tensor, &RBArray>
	mut comps: HashMap<&Tensor, RBArray>,
	mut comp_stack: &mut Vec<&Tensor>
}
impl Session {
	fn new() -> Session {
		Session {feeds: HashMap::new(), comps: HashMap::new()}
	}

	///Determines whether x has already been feeded or computed
	fn extract_val(&self, x: &Tensor) -> Option<&RBArray> {
		match (self.feeds.get(x), self.comps.get(x)) {
			(Some(&feed), _          ) => Some(feed),
			(_          , Some(comp) ) => Some(comp)
			 _                         => None
		}
	}

	///Push all predecessors of x, direct or indirect, to the comp_stack
	fn push_preds(&self, x: &Tensor) {
		for pred in x.preds().iter() {
			if None == self.extract_val(pred){
				self.comp_stack.push(pred);
				self.push_preds(pred);
			}
		}
	}
	

	fn run(&self, feeds: &HashMap<&Tensor, &RBArray>, fetches: Vec<&Tensor>) -> Vec<RBArray> {
		
		self.feeds = feeds;
		self.comp_stack.clear();
		self.comps.clear();

		let mut evaluations: Vec<RBArray> = vec![];
		//Invariants
		// Pre-execution: evaluations is empty
		// After an iteration: evaluations has at its tail the evaluation of the fetch
		for fetch in fetches.iter() {
			if Some(&feed) = feeds.get(fetch) {
				eprintln!("WARNING: You are fetching {:?}, which you already feeded.", fetch);
				evaluations.push(feed.clone());
				continue;
			}
			self.comp_stack.push(fetch);
			self.push_preds(fetch);


			//Invariant: the top element of the stack already has its predecessors computed
			//Invariant: After the loop has been executed, fetch's value is in comps
			while comp_stack.len() > 0 {
				let ready_comp = comp_stack.pop().unwrap();

				//The Tensor was already computed earlier
				if None == self.extract_val(ready_comp) {continue;}

				let pred_vals: Vec<RBArray> = vec![];
				for pred in ready_comp.preds.iter() {
					let pred_value = extract_val(pred, &feeds, &fetches);
					assert_ne!(pred_value, None);
					pred_vals.push(pred_value.unwrap());
				}
				let result: RBArray = (ready_comp.eval)(&pred_vals);
				comps.insert(pred, result);
			}
			let fetch_value = comps.get(fetch);
			assert_ne!(fetch_value, None);
			evaluations.push(fetch_value);
		}
		evaluations
	}
}

