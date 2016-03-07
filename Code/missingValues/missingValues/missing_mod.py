def covarianceOfRows_mod(X):
	"""
	MODIFIED to standardise colums rather than rows

	Returns a covariance matrix calculated using missing-aware operations
	where we take the rows of the sparse matrix X to be variables and
	columns to be observations. The 0 values in X are taken to be
	missing. This is done roughly via a vectorized version of the
	two-pass algorithm for calculating covariance.

	"""
	# get mean of each column as vector?
	columnMeans = meanOfColumns(X)

	# subtract means from X
	XMeanSub = X.copy()
	_subtractProjectedVector(XMeanSub, columnMeans, 'column')

	#### divide by standard deviation of each feature
	XScaled

	####

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