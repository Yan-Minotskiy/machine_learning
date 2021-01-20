import numpy

class MGAA(object):
	"""metric: linear, quadratic"""
	def __init__(self, metric=mse, ref_func=2, max_selection=None, mixed_selection=False):
		super(MGAA, self).__init__()
		self.metric = metric
		if 1 <= ref_func <= 4 and type(ref_func) == int:
			self.ref_func = ref_func
		else:
			raise Error
		self.max_selection = max_selection
		self.mixed_selection = mixed_selection
		self.coef = [1, 1, 1, 1, 1, 1]
		pass

	def train(self, X, y):
		if ref_func == 2:
			pass

	def predict(self, x1, x2):
		if self.ref_func == 1:
			self.coef[0] + self.coef[1] * x1 * x2
		elif self.ref_func == 2:
			self.coef[0] + self.coef[1] * x1 + self.coef[2] * x2
		elif self.ref_func == 3:
			self.coef[0] + self.coef[1] * x1 + self.coef[2] * x2 + self.coef[3] * x1 * x2
		else:
			self.coef[0] + self.coef[1] * x1 + self.coef[2] * x2 + self.coef[3] * x1 * x2 + self.coef[4] * x2 * x2 + self.coef[5] * x2 * x2

