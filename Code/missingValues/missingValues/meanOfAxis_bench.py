"""
Time trial benchmarks of various implementations of getting the mean of
each row or column

"""

import numpy
import timeit

def foo(X):
	pass


def meanOfAxisBenchmark():
	setup = """
import numpy
from __main__ import foo
num = 1000000
	"""

	numTrials = 5
	numInTrial = 1000

	tp = "Timings down over " + str(numTrials) + " trials each with "
	tp += str(numInTrial) + " repetitions on an object with shape "
	tp += "(10000,) and ~5000 non-zero values\n"
	print tp

	stmt = "foo()"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "sum of numpy zero test arrays"
	print "Average: " + str(numpy.mean(results)) + '\n'


######################
### Demonstration? ###
######################

if __name__ == "__main__":
	meanOfAxisBenchmark()
