import theano
import numpy as np

class stepwise_tanh_op(theano.Op):
	__props__ = ()

	def make_node(self, x):
		# check that the theano version has support for __props__.
		# This next line looks like it has a typo,
		# but it's actually a way to detect the theano version
		# is sufficiently recent to support the use of __props__.
		assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
		x = theano.tensor.as_tensor_variable(x)
		return theano.Apply(self, [x], [x.type()])

	def perform(self, node, inputs, output_storage):
		x = inputs[0]
		z = output_storage[0]
		y = x.copy()
		a = 1
		y[y > a] = a
		y[y < -a] = -a
		z[0] = y

	def infer_shape(self, node, i0_shapes):
		return i0_shapes

	def grad(self, inputs, output_grads):
		g = np.ones(inputs[0].shape)
		a = 1
		g[inputs[0] > a] = 0
		g[inputs[0] < -a] = 0
		return [g[0]]
        #return [output_grads[0]]

	def R_op(self, inputs, eval_points):
		# R_op can receive None as eval_points.
		# That mean there is no diferientiable path through that input
		# If this imply that you cannot compute some outputs,
		# return None for those.
		if eval_points[0] is None:
			return eval_points
		return self.grad(inputs, eval_points)

def create():
	x = theano.tensor.matrix()
	return theano.function([x], stepwise_tanh_op()(x))

if __name__ == "__main__":
	stepwise_tanh = create()
	inp = np.random.rand(5, 4) * 4 - 2
	out = stepwise_tanh(inp)
	#assert numpy.allclose(inp * 2, out)
	print(inp)
	print(out)
