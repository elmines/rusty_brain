use crate::core::Tensor;

///Apply a fully-connected layer to input x
///
///The number of weights is the size of x's last dimension times the projection size
pub fn dense(x: Tensor, size: u64) -> Tensor {

	let weights = Tensor::new( vec![x.shape[x.shape.len() - 1], size] );

	x.dot(weights)
}
