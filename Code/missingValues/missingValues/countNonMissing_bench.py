"""
Time trial benchmarks of various implementations of counting how many
pairs of elements are non-zero in two vectors

"""

import numpy
import timeit

def _countNonMissingInBoth_set(a,b):
	"""
	Returns the number of indices which contain non-zero elements
	in both arrays.

	"""
	# get indices of non-zero values
	aSet = set(numpy.nonzero(a)[0])
	bSet = set(numpy.nonzero(b)[0])
	# get the length of the intersection of indices
	return len(aSet & bSet)

def _countNonMissingInBoth_numpy(a,b):
	"""
	Returns the number of indices which contain non-zero elements
	in both arrays.

	"""
	return numpy.sum(numpy.multiply(a != 0, b != 0))


def countNonMissingBenchmark():
	setup = """
import numpy
from __main__ import _countNonMissingInBoth_set
from __main__ import _countNonMissingInBoth_numpy
num = 1000000
mask1 = numpy.random.randint(2,size=num)
mask2 = numpy.random.randint(2,size=num)
	"""

	numTrials = 5
	numInTrial = 1000

	tp = "Timings down over " + str(numTrials) + " trials each with "
	tp += str(numInTrial) + " repetitions on an object with shape "
	tp += "(10000,) and ~5000 non-zero values\n"
	print tp

#	stmt = "_countNonMissingInBoth_set(mask1,mask2)"
#	t = timeit.Timer(stmt=stmt, setup=setup)
#	results = t.repeat(numTrials,numInTrial)
#	print "Set of indices, len of intersection"
#	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_countNonMissingInBoth_numpy(mask1,mask2)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "sum of numpy zero test arrays"
	print "Average: " + str(numpy.mean(results)) + '\n'


######################
### Demonstration? ###
######################

if __name__ == "__main__":
	countNonMissingBenchmark()
