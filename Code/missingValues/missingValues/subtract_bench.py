"""
Time trial benchmarks of various implementations of subtracting from
missing-aware vectors

"""

import numpy
import timeit

def subtractPython(a, b):
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


def subtractNumpy(a, b):
	aMask = a != 0
	return a - aMask * b


def subBenchmark():
	setup = """
import numpy
from __main__ import subtractPython
from __main__ import subtractNumpy
num = 1000000
	"""

	numTrials = 5
	numInTrial = 1000

	tp = "Timings down over " + str(numTrials) + " trials each with "
	tp += str(numInTrial) + " repetitions on an object with shape "
	tp += "(10000,) and ~5000 non-zero values\n"
	print tp

	stmt = "foo(mask1,mask2)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "sum of numpy zero test arrays"
	print "Average: " + str(numpy.mean(results)) + '\n'


######################
### Demonstration? ###
######################

if __name__ == "__main__":
	subBenchmark()
