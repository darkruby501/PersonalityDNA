"""
Time trial benchmarks of various implementations of subtraction of a
vector of values from a csr or csc sparse matrix, where the vector
is broadcast to the shape of the matrix.

"""

import numpy
from numpy.lib.stride_tricks import as_strided
import timeit

def _sub_projAlongCompressedAxis_stridedIterateIndptr(X,vector):
	row_start_stop = as_strided(X.indptr, shape=(X.shape[0], 2),
		strides=2*X.indptr.strides)
	for row, (start, stop) in enumerate(row_start_stop):
		axisSlice = X.data[start:stop]
		axisSlice -= vector[row]

	return

def _sub_projAlongCompressedAxis_iterateIndptr(X, vector):
	# index into X.indptr, and conseqently, also the index into
	# the matching location in vector
	startIndex = 0

	# indptr is a sorted list of indices into X.data, with
	# the exception of its last value, which equals len(X.data),
	# or, in other words, the number of non-zero values.
	while(X.indptr[startIndex] != X.nnz):
		vecStart = X.indptr[startIndex]
		vecEnd = X.indptr[startIndex + 1]
		# get a view into X.data that corresponds to the nonzero
		# values of the current row or column
		axisSlice = X.data[vecStart:vecEnd]
		# the view can be used to modify the source object, so we
		# only need to do an inplace scalar subtract
		axisSlice -= vector[startIndex]
		startIndex += 1
	return

def _sub_projAlongCompressedAxis_UFunc(X, vector):
	def lookup(index):
		return vector[X.indices[index]]

	lookupUFunc = numpy.frompyfunc(lookup, 1, 1)

	numpy.subtract(X.data, lookupUFunc(X.data), X.data)
	return

def _sub_projAlongFullAxis_stridedIterate(X,vector):
	dataValues = as_strided(X.data, shape=(X.data.shape[0],),
		strides=X.data.strides)
	for i, (val) in enumerate(dataValues):
		X.data[i] = val - vector[X.indices[i]]

def _sub_projAlongFullAxis_iterate(X, vector):
	for i, val in enumerate(X.data):
		X.data[i] = val - vector[X.indices[i]]
	return

def _sub_projAlongFullAxis_UFunc(X, vector):
	def lookup(index):
		return vector[index]

	lookupUFunc = numpy.frompyfunc(lookup, 1, 1)

	numpy.subtract(X.data, lookupUFunc(X.indices), X.data)
	return

def _sub_projAlongFullAxis_vectorize(X, vector):
	def lookUpAndSub(index):
		X.data[index] -= vector[X.indices[index]]

	vecFunc = numpy.vectorize(lookUpAndSub)
	vecFunc(numpy.indices(X.data.shape))

	return

def _sub_projAlongFullAxis_numpyOpsVec(X, vector):
	X.data -= vector[X.indices]
	return



def sparseSubtractionBenchmark():
	setup_LargeSize_quarterSparse = """
import numpy
import scipy.sparse
from __main__ import _sub_projAlongFullAxis_UFunc
from __main__ import _sub_projAlongFullAxis_iterate
from __main__ import _sub_projAlongFullAxis_stridedIterate
from __main__ import _sub_projAlongCompressedAxis_iterateIndptr
from __main__ import _sub_projAlongCompressedAxis_stridedIterateIndptr
from __main__ import _sub_projAlongFullAxis_vectorize
from __main__ import _sub_projAlongFullAxis_numpyOpsVec
num = 1000000
mask = numpy.random.randint(2,size=num) * numpy.random.randint(2,size=num) * numpy.random.randint(2,size=num)
x = numpy.random.rand(num) * mask
x.resize(5000,200)
s = scipy.sparse.csr_matrix(x)
v = numpy.ones(5000)
r = numpy.random.rand(len(s.data)) * numpy.random.rand(len(s.data)) * 100
t = s.T
	"""

	setup = setup_LargeSize_quarterSparse
	numTrials = 5
	numInTrial = 20

	tp = "Timings down over " + str(numTrials) + " trials each with "
	tp += str(numInTrial) + " repetitions on an object with shape "
	tp += "(2500, 400) and ~125,000 non-zero values\n"
	print tp

	stmt = "numpy.subtract(s.data, r, s.data)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "Baseline - numpy vectorized subtraction from spase object's data attribute"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongFullAxis_UFunc(s,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along full axis - ufunc lookup, numpy vector subtract"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongFullAxis_iterate(s,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along full axis - python iterate, individual subtract"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongFullAxis_stridedIterate(s,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along full axis - strided iteration, individual subtract"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongFullAxis_vectorize(s,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along full axis - numpy.vectorize lookup and sub func"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongFullAxis_numpyOpsVec(s,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along full axis - numpy indexing tricks, pure numpy ops"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongFullAxis_numpyOpsVec(s.T,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along full axis - transpose, numpy indexing tricks, pure numpy ops"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongCompressedAxis_iterateIndptr(s.T,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along compressed axis - transpose, python iterate, slice subtract"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongCompressedAxis_stridedIterateIndptr(s.T,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along compressed axis - transpose, strided iteration, slice subtract"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongCompressedAxis_iterateIndptr(t,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along compressed axis - python iterate, slice subtract"
	print "Average: " + str(numpy.mean(results)) + '\n'

	stmt = "_sub_projAlongCompressedAxis_stridedIterateIndptr(t,v)"
	t = timeit.Timer(stmt=stmt, setup=setup)
	results = t.repeat(numTrials,numInTrial)
	print "along compressed axis - strided iteration, slice subtract"
	print "Average: " + str(numpy.mean(results)) + '\n'


######################
### Demonstration? ###
######################

if __name__ == "__main__":
	sparseSubtractionBenchmark()
