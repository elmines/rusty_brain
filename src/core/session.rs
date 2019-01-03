use std::collections::HashMap;
use crate::core::tensor::Tensor;
use crate::ndarray::ArrayD;

pub struct Session<'a> {
	feeds:      HashMap<&'a Tensor<'a>, &'a ArrayD<f32>>,
	comps:      HashMap<&'a Tensor<'a>, ArrayD<f32>>,
	comp_stack: Vec<&'a Tensor<'a>>,
}
impl<'a> Session<'a> {
	pub fn new() -> Session<'a> {
		Session {feeds: HashMap::new(), comps: HashMap::new(), comp_stack: vec![]}
	}

	pub fn run(&mut self, feeds: HashMap<&'a Tensor, &'a ArrayD<f32>>, fetches: Vec<&'a Tensor>) -> Vec<ArrayD<f32>> {
		
		self.feeds = feeds;
		self.comp_stack.clear();
		self.comps.clear();

		let mut evaluations: Vec<ArrayD<f32>> = vec![];
		for fetch in fetches.iter() {

			if let Some(&feed) = self.feeds.get(fetch) {
				eprintln!("WARNING: You are fetching {:?}, which you already feeded.", fetch);
				evaluations.push(feed.clone());
				continue;
			}
			self.comp_stack.push(fetch);
			self.push_preds(fetch);

			while self.comp_stack.len() > 0 {
				let ready_comp = self.comp_stack.pop().unwrap();

				//The Tensor was already computed earlier
				if None != self.extract_val(ready_comp) {continue;}

				let result: ArrayD<f32> = {
					let mut pred_vals: Vec<&ArrayD<f32>> = vec![];
					for pred in ready_comp.preds().iter() {
						let pred_value = self.extract_val(pred);
						pred_vals.push(pred_value.unwrap());
					}
					ready_comp.eval(&pred_vals)
				};
				self.comps.insert(ready_comp, result);
			}
		}

		//Second pass: Move the fetch values from comps to evaluations
		for fetch in fetches.iter() {
			let fetch_value = self.comps.remove(fetch);
			evaluations.push(fetch_value.unwrap());
		}
		evaluations
	}

	///Determines whether x has already been feeded or computed
	fn extract_val(&'a self, x: &'a Tensor) -> Option<&'a ArrayD<f32>> {

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
		for pred in x.preds().iter() {
			if None == self.extract_val(pred){
				self.comp_stack.push(pred);
				self.push_preds(pred);
			}
		}
	}
	
}

