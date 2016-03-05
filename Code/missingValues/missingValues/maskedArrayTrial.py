"""
Script trialing the use of masked arrays as UML's Dense type's
missing values implementation

"""

# Things we care about:
# A) numerical ops - missing should propagate
#	- add, sub, mult, div, power
# B) stats functions
#	- min mean med, cov, corr, std,
# C) structural
#	- append extract
# D) query
#	- should 0 or nan be returned? is it conditional?

# A) we get for free from masked arrays- masks propagate
# B) not sure if we get for free, but can calculate using (A)
# C) 
# D) we can set via filled value


# ISSUES

# SUPER SLOW - apparently it is all pure python.

# this touches on whether Dense should still internally be a matrix instead
# of a 2d array

# hash codes? do we treat missing as zero?

# how does missing work wrt to views: what does the user see if
# they peek that value?


import numpy
import timeit

def ti():
	print timeit.timeit('numpy.ones((270000,50), float)/numpy.ones((270000,50), float)', setup='import numpy')
	return

	x = numpy.ma.ones((270000,50), float)
	y = numpy.ma.ones((270000,50), float)
	x[1,1] = numpy.ma.masked
	y[1,2] = numpy.ma.masked

	print timeit.timeit('x/y')

# run through and confirm behaviour
if __name__ == '__main__':

	ti()
	exit(0)
	raw = [[1,2,3,4,5]]
	mask = [[0,0,0,0,1]]

	test = numpy.ma.array(raw, mask=mask)

	print "original:"
	print test

	#A) numerical ops - missing should propagate

	# adding
	toAdd = [[1, 1, float('nan'), 10, 10]]
	toAdd = numpy.ma.array(toAdd, mask=[[0,0,1]])

	exp = numpy.ma.array([[12,0,0]], mask=[[0,1,1]])
	numpy.testing.assert_array_equal(test + toAdd, exp)

	# multiply
	toMul = [[-1,1,float('nan')]]
	toMul = numpy.ma.array(toMul, mask=[[0,0,1]])

	exp = numpy.ma.array([[12,0,0]], mask=[[0,1,1]])
	numpy.testing.assert_array_equal(test - toMul, exp)

