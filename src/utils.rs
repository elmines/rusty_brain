use crate::ndarray::{Array, IxDyn, ArrayD};


pub fn into_dynamic<A, D>(x: Array<A, D>) -> ArrayD<A>
	where D: ndarray::Dimension
{

	match x.into_dimensionality::<IxDyn>() {
		Ok(array) => array,
		Err(message) => panic!("{}", message)
	}

}
