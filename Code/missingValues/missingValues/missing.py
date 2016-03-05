"""
Contains functions which operate intelligently on data encoding
missing values as a 0. Each such function will take either arrays
or scipy.sparse objects as inputs, returning objects as appropriate.


"""

import numpy
import scipy.sparse

from numpy.lib.stride_tricks import as_strided

__all__ = ['mean', 'covariance', 'variance', 'standardDeviation',
			'meanOfRows', 'meanOfColumns', 'covarianceOfRows']

################################
### Forward Facing Functions ###
################################


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


def covariance(a, b):
	"""
	Returns the covariance between iterables a and b calculated using
	missing-aware operations. Any 0 values in a or b are taken to be
	missing. Relies on the missing aware mean() function.

	"""
	nz = _countNonMissingInBoth(a,b)
	# check to see if we are able to use this as a divisor
	if (nz == 1) or (nz == 0):
		return 0

	aSub = _subtract(a, mean(a))
	bSub = _subtract(b, mean(b))

	# regular division suffices here because the 0s will be preserved.
	return numpy.dot(aSub, bSub) / (nz - 1)


def variance(a):
	"""
	Returns the variance of the iterable a calculated using missing-
	aware operations. Any 0 values in a or b are taken to be missing.
	Relies on the missing aware mean() and covariance() functions.

	"""
	return covariance(a,a)


def standardDeviation(a):
	"""
	Returns the standard deviation of the iterable a calculated using
	missing-aware operations. Any 0 values in a or b are taken to be
	missing. Relies on the missing aware mean() and covariance()
	functions.

	"""
	return numpy.sqrt(variance(a))


def meanOfRows(X):
	"""
	Returns a vector containing the missing-aware mean of each row in
	the given sparse matrix X, where a value of 0 indicates a missing
	value.

	"""
	if not scipy.sparse.isspmatrix_csr(X):
		X = scipy.sparse.csr_matrix(X)
	return _meanOfCompressedAxis(X)


def meanOfColumns(X):
	"""
	Returns a vector containing the missing-aware mean of each column
	in the given sparse matrix X where a value of 0 indicates a missing
	value.
	
	"""
	if not scipy.sparse.isspmatrix_csc(X):
		X = scipy.sparse.csc_matrix(X)
	return _meanOfCompressedAxis(X)

def covarianceOfRows(X):
	"""
	Returns a covariance matrix calculated using missing-aware operations
	where we take the rows of the sparse matrix X to be variables and
	columns to be observations. The 0 values in X are taken to be
	missing. This is done roughly via a vectorized version of the
	two-pass algorithm for calculating covariance.

	"""
	# get mean of each row as vector?
	rowMeans = meanOfRows(X)

	# subtract means from X
	XMeanSub = X.copy()
	_subtractProjectedVector(XMeanSub, rowMeans, 'row')

	# get transpose of mean subtracted X
	XMeanSub_t = XMeanSub.transpose()

	# normal dot product operates as we want for missing-aware
	# operations, therefore normal matrix multiply will too.
	toDivide = XMeanSub * XMeanSub_t

	# We are taking columns to be observations, and since we are, so we
	# need the row pairwise number nonzero
	if scipy.sparse.isspmatrix_csr(X):
		nnzMask = scipy.sparse.csr_matrix(
				(numpy.ones(len(X.data)),X.indices,X.indptr), copy=False)
	else:
		nnzMask = scipy.sparse.csc_matrix(
				(numpy.ones(len(X.data)),X.indices,X.indptr), copy=False)
	# these are sparse, so this is a dot product, like we want
	divisors = nnzMask * nnzMask.T
	# doing sample covariance calculation, we subtract 1 (not going below 0)
	divisors.data -= 1
	divisors.eliminate_zeros()

	# we want a zero division (pairwise nnz = 0 or 1) to result in
	# a 0, not positive infinity, so we make sure all of those
	# entires will be doing 0/0 instead of x/0, which in a sparse
	# matrix wsill result in a 0
	if scipy.sparse.isspmatrix_csr(divisors):
		divMask = scipy.sparse.csr_matrix(
				(numpy.ones(divisors.data.shape[0]),divisors.indices,divisors.indptr), copy=False)
	else:
		divMask = scipy.sparse.csc_matrix(
				(numpy.ones(divisors.data.shape[0]),divisors.indices,divisors.indptr), copy=False)

	ret = (toDivide.multiply(divMask)) / divisors
	return ret.A  # return dense matrix type


###############
### Helpers ###
###############

def _subtract(a, b):
	"""
	Return the result of doing a missing-aware subtraction
	of b from a. We take zero values in a to be missing, and a missing
	value will be preserved in the result regardless of the contents
	of b. a must be a one dimensional numpy array, b may either be
	a scalar, or a one dimensional numpy array

	"""
	# array case
	if isinstance(b, numpy.ndarray):
		mask = numpy.multiply(a != 0, b != 0)
		return (a - b) * mask
	# scalar case
	else:
		aMask = a != 0
		return a - aMask * b


def _countNonMissingInBoth(a,b):
	"""
	Returns the number of indices which contain non-zero elements
	in both arrays.

	"""
	return numpy.sum(numpy.multiply(a != 0, b != 0))


def _meanOfCompressedAxis(X):
	"""
	Given a compressed sparse row or compressed sparse column matrix,
	return the missing-aware mean of the compressed axis (row for a
	csr matrix, col for a csc matrix).

	"""
	startIndex = 0
	ret = numpy.empty(len(X.indptr)-1)

	# indptr is a sorted list of indices into X.data, with
	# the exception of its last value, which equals len(X.data),
	# or, in other words, the number of non-zero values.
	while(X.indptr[startIndex] != X.nnz):
		vecStart = X.indptr[startIndex]
		vecEnd = X.indptr[startIndex + 1]

		ret[startIndex] = mean(X.data[vecStart:vecEnd])

		startIndex += 1

	return ret


def _subtractProjectedVector(X, vector, projectionAxis):
	"""
	From X, do a non-zero elementwise subtraction of the projection of
	the given vector out along the given axis.

	"""
	if scipy.sparse.isspmatrix_csc(X):
		if projectionAxis == 'row':
			_sub_projAlongFullAxis(X, vector)
		else:
			_sub_projAlongCompressedAxis(X, vector)
	elif scipy.sparse.isspmatrix_csr(X):
		if projectionAxis == 'row':
			_sub_projAlongCompressedAxis(X, vector)
		else:
			_sub_projAlongFullAxis(X, vector)
	else:
		raise ValueError("Must convert X into a csr_matrix or csc_matrix")

	# we may have introduced zeros into the data attribute, so clean
	# them up
	X.eliminate_zeros()
	return


def _sub_projAlongCompressedAxis(X, vector):
	"""
	From X, do a non-zero elementwise subtraction of the projection of
	the given vector out along compressed axis. This makes use of
	as_strided to generate the beginning and end points of each row
	slice of the data attribute, as specified at
	http://stackoverflow.com/questions/20060753/efficiently-subtract-vector-from-matrix-scipy?rq=1

	"""
	rowSliceIndices = as_strided(X.indptr, shape=(X.shape[0], 2),
		strides=2*X.indptr.strides)
	for row, (start, stop) in enumerate(rowSliceIndices):
		axisSlice = X.data[start:stop]
		axisSlice -= vector[row]
	return


def _sub_projAlongFullAxis(X, vector):
	"""
	From X, do a non-zero elementwise subtraction of the projection of
	the given vector out along compressed axis. This makes use of
	the indices attribute as an index array into the vector, and inplace
	vectorized subtraction from X.data

	"""
	X.data -= vector[X.indices]
	return



##################
### Unit tests ###
##################

def test_mean():
	"""
	Validity of the results of the mean function.
	If given an empty array or one with all zeros, we define the mean
	to be 0. If there are no zeros, it should match the results of a
	normal mean calculation (we use numpy.mean as a baseline).

	"""
	assert mean([]) == 0
	assert mean([0,0,0]) == 0
	assert mean([11,0,0]) == 11
	assert mean([123, 12, 16]) == numpy.mean([123, 12, 16])

	for i in xrange(10):
		data = numpy.random.rand(10)
		assert mean(data) == numpy.mean(data)


def test_meanOfCompressedAxis():
	""" Validity of results of the _meanOfCompressedAxis helper """
	data = [[0,0,0],[1,0,0],[0,5.5,0],[4,6,0],[-1,0,1]]

	npData = numpy.array(data)
	csrData = scipy.sparse.csr_matrix(npData)
	cscData = scipy.sparse.csc_matrix(npData.T)

	retCSR = _meanOfCompressedAxis(csrData)
	numpy.testing.assert_array_equal(retCSR, [0,1,5.5,5,0])

	retCSC = _meanOfCompressedAxis(cscData)
	numpy.testing.assert_array_equal(retCSC, [0,1,5.5,5,0])


def test_meanOfRows_meanOfColumns():
	""" Test behaviour of forward facing meanOfRows and meanOfColumns.
	Looks at validity of results, and ability to handle conversions
	between sparse types.

	"""
	data = [[0,0,0],[1,0,0],[0,5.5,0],[4,6,0],[-1,0,1]]
	npData = numpy.array(data)

	possible = ['csr_matrix', 'csc_matrix', 'coo_matrix']
	for t in possible:
		construct = getattr(scipy.sparse, t)
		
		forRow = construct(npData)
		forCol = construct(npData.T)

		retRow = meanOfRows(forRow)
		numpy.testing.assert_array_equal(retRow, [0,1,5.5,5,0])

		retCol = meanOfColumns(forCol)
		numpy.testing.assert_array_equal(retCol, [0,1,5.5,5,0])


def test_covariance_matchesNoneMissing():
	"""
	covariance - matches numpy results on all non-zero data.
	Since there are no zeros, this should match a normal covariance
	calculation, and serves as a basic check for correctness. We use
	numpy.cov as a baseline implementation.

	"""
	for i in xrange(10):
		data = numpy.random.rand(2,10)
		dataA = data[0,:]
		dataB = data[1,:]

		ourRet = covariance(dataA, dataB)
		# this is a cov matrix, we want the entry at [0,1]
		# or [1,0]
		npRet = numpy.cov(data)[0,1]

		numpy.testing.assert_array_equal(ourRet, npRet)


def test_covariance_handmade():
	"""
	covariance - correct output for handmade dataset.

	"""
	data = numpy.array([[-1,1,1],[-1,3,1]])

	dataA = data[0]
	dataB = data[1]

	ourRet = covariance(dataA, dataB)
	# this is a cov matrix, we want the entry at [0,1]
	# or [1,0]
	npRet = numpy.cov(data)[0,1]

	numpy.testing.assert_array_equal(ourRet, npRet)


def test_covariance_edgeCases():
	"""
	covariance - divisor edge cases
	"""
	data = [[1,1,0],[1,0,1]]

	# Case: pairwise number non zero is 1
	ourRet = covariance(data[0], data[1])
	assert ourRet == 0

	data = [[0,1,0],[1,0,1]]

	# Case: pairwise number non zero is 0
	ourRet = covariance(data[0], data[1])
	assert ourRet == 0


def test_covarianceOfRows_matchesNoneMissing():
	"""
	covarianceOfRows - matches numpy results on all non-zero data.
	Since there are no zeros, this should match a normal covariance
	calculation, and serves as a basic check for correctness. We use
	numpy.cov as a baseline implementation.

	"""
	for i in xrange(10):
		data = numpy.random.rand(5,10)
		sCSR = scipy.sparse.csr_matrix(data)
		sCSC = scipy.sparse.csc_matrix(data.T)

		csrRet = covarianceOfRows(sCSR)
		cscRet = covarianceOfRows(sCSC)

		# rows as varables
		npRetRowMaj = numpy.cov(data)
		# columns as variables
		npRetColMaj = numpy.cov(data, rowvar=0)

		numpy.testing.assert_allclose(csrRet, npRetRowMaj, rtol=1e-10)
		numpy.testing.assert_allclose(cscRet, npRetColMaj, rtol=1e-10)


def test_covarianceOfRows_handmade_withMissingValues():
	"""
	covarianceOfRows - correct output for handmade dataset.

	"""
	data = [[-1,0,1],[-1,3,0], [1,0,3]]
	npData = numpy.array(data)
	sparse = scipy.sparse.csr_matrix(npData)

	ret = covarianceOfRows(sparse)
	exp = [[2,0,2],[0,8,0],[2,0,2]]

	numpy.testing.assert_array_equal(ret, exp)


def test_covarianceOfRows_edgeCases():
	"""
	covarianceOfRows - divisor edge cases

	"""
	data = [[0,3,0],[3,0,5]]
	data = numpy.array(data)
	data = scipy.sparse.csr_matrix(data)

	ourRet = covarianceOfRows(data)

	# Case: pairwise number non zero is 1
	assert ourRet[0,0] == 0

	# Case: pairwise number non zero is 0
	assert ourRet[0,1] == 0

	# should be symetric
	assert ourRet[1,0] == 0


#########################
### Helper Unit tests ###
#########################

def test_subtract():
	a1 = numpy.array([0,1,1])
	a2 = numpy.array([0,0,-1])
	ret = _subtract(a1,a2)
	exp = numpy.array([0,0,2])
	numpy.testing.assert_array_equal(ret, exp)

	ret = _subtract(a1, -2)
	exp = numpy.array([0,3,3])
	numpy.testing.assert_array_equal(ret, exp)


def test_subtractProjectedVector_Compressed():
	"""
	_subtractProjectedVector - correct for handmade data, compressed axis

	"""
	Xdata = [[1,1,1], [10,0,0], [0,10,0], [0,10,10]]
	vData = [-5, 20, 10, 1]

	X = scipy.sparse.csr_matrix(numpy.array(Xdata))

	_subtractProjectedVector(X, numpy.array(vData), 'row')

	ret = X.todense()
	exp = [[6,6,6], [-10,0,0], [0,0,0], [0,9,9]]
	numpy.testing.assert_array_equal(ret, exp)
	assert not numpy.any(X.data == 0)


def test_subtractProjectedVector_Uncompressed():
	"""
	_subtractProjectedVector - correct for handmade data, uncompressed axis

	"""
	Xdata = [[1,1,1], [10,0,0], [0,10,0], [0,10,10]]
	vData = [-5, 20, 10, 1]

	X = scipy.sparse.csc_matrix(numpy.array(Xdata))

	_subtractProjectedVector(X, numpy.array(vData), 'row')
	ret = X.todense()
	exp = [[6,6,6], [-10,0,0], [0,0,0], [0,9,9]]
	numpy.testing.assert_array_equal(ret, exp)
	assert not numpy.any(X.data == 0)


#####################
### Benchmarking? ###
#####################


######################
### Demonstration? ###
######################

if __name__ == "__main__":
	pass
