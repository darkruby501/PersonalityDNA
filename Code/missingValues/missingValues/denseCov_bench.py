"""
Time trial benchmarks of various implementations of covariance of two
vectors

"""

import numpy
import timeit

def subtract(a, b):
	"""
	Return the result of doing a missing-aware subtraction
	of b from a. We take zero values in a to be missing, and a missing
	value will be preserved in the result regardless of the contents
	of b.

	"""
	# array case
	if hasattr(b, '__len__'):
		ret = numpy.zeros(len(a))
		for i, val in enumerate(a):
			if val != 0:
				ret[i] = val - b[i]
	# scalar case
	else:
		ret = numpy.zeros(len(a))
		for i, val in enumerate(a):
			if val != 0:
				ret[i] = val - b

	return ret

def mean(a):
	"""
	Returns the missing-aware mean of the vector a, where a value of
	0 indicates a missing value. This is equivalent to the sum of
	values divided by the number of non-zero values. a and b are
	lists of values of the same length. If given an iterable with no
	elements, we define the mean to be 0; if given an iterable with
	no non-zero elements, we define the mean to be 0.

	"""
	div = numpy.count_nonzero(a)
	# This occurs if len(a) == 0, or if all elements are equal to 0
	if div == 0:
		return 0

	return numpy.sum(a) / float(div)

def _countNonMissingInBoth(a,b):
	"""
	Returns the number of indices which contain non-zero elements
	in both arrays.

	"""
	return numpy.sum(numpy.multiply(a != 0, b != 0))

def simpleCov(a,b):
	nz = _countNonMissingInBoth(a,b)
	# check to see if we are able to use this as a divisor
	if (nz == 1) or (nz == 0):
		return 0

	aSub = subtract(a, mean(a))
	bSub = subtract(b, mean(b))

	# regular division suffices here because the 0s will be preserved.
	return numpy.dot(aSub, bSub) / (nz - 1)


def vectorizedCov(a,b):
	nz = _countNonMissingInBoth(a,b)
	# check to see if we are able to use this as a divisor
	if (nz == 1) or (nz == 0):
		return 0

	def subtractNumpy(a, b):
		#isinstance(a, numpy.ndarray)
		aMask = a != 0
		return a - aMask * b

	aSub = subtractNumpy(a, mean(a))
	bSub = subtractNumpy(b, mean(b))

	# regular division suffices here because the 0s will be preserved.
	return numpy.dot(aSub, bSub) / (nz - 1)



def maCov(a,b):
	nz = numpy.sum(numpy.multiply(a.mask, b.mask))
	if (nz == 1) or (nz == 0):
		return 0

	aSub = a - (numpy.sum(a)/a.count())
	bSub = b - (numpy.sum(b)/b.count())

	return numpy.ma.dot(aSub, bSub) / (nz - 1)


def denseCovarianceBenchmark():
	setup = """
import numpy
from __main__ import simpleCov
from __main__ import vectorizedCov
from __main__ import maCov

num = 1000
raw1 = numpy.random.rand(num)
raw2 = numpy.random.rand(num)
mask1 = ~(numpy.random.randint(2,size=num) * numpy.random.randint(2,size=num))
mask2 = ~(numpy.random.randint(2,size=num) * numpy.random.randint(2,size=num))
v1 = numpy.multiply(raw1,mask1)
v2 = numpy.multiply(raw2,mask2)
ma1 = numpy.ma.array(raw1, mask=mask1)
ma2 = numpy.ma.array(raw2, mask=mask2)
	"""

	numTrials = 5
	numInTrial = 1000

	tp = "Timings down over " + str(numTrials) + " trials each with "
	tp += str(numInTrial) + " repetitions on two objects with shape "
	tp += "(10000,) and ~7500 non-zero values\n"
	print tp

	stmt = "numpy.cov(v1,v2)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "standard numpy covariance"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "simpleCov(v1,v2)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "python implementations, 0 as missing"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "vectorizedCov(v1,v2)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "numpy vectorized implementations, 0 as missing"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "maCov(ma1,ma2)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "numpy masked array"
	print "Average: " + str(numpy.mean(results)) + '\n'


######################
### Demonstration? ###
######################

if __name__ == "__main__":
	denseCovarianceBenchmark()
