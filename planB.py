# Importing wals
import wals
# Importing tensorflow
import tensorflow as tf
# Importing some more libraries
import pandas as pd
import numpy as np

# ratio of train set size to test set size
TEST_SET_RATIO = 5

# Create test and train sets
def create_test_and_train_sets():
	# reading the ratings data
	print('\nReading ratings data...')
	ratings_df = pd.read_csv(
		'dataset/ratings.dat',
		sep='::',
		names=['user_id', 'movie_id', 'rating', 'timestamp'],
		header=None,
		dtype={
			'user_id': np.int32,
			'movie_id': np.int32,
			'rating': np.float32,
			'timestamp': np.int32,
		})

	np_users = ratings_df.user_id.as_matrix()
	np_movies = ratings_df.movie_id.as_matrix()
	unique_users = np.unique(np_users)
	unique_movies = np.unique(np_movies)

	n_users = unique_users.shape[0]
	n_movies = unique_movies.shape[0]

	# make indexes for users and items if necessary
	max_user = unique_users[-1]
	max_movie = unique_movies[-1]
	if n_users != max_user or n_movies != max_movie:
		# make an array of 0-indexed unique user ids corresponding to the dataset
		# stack of user ids
		z = np.zeros(max_user+1, dtype=int)
		z[unique_users] = np.arange(n_users)
		u_r = z[np_users]

		# make an array of 0-indexed unique item ids corresponding to the dataset
		# stack of item ids
		z = np.zeros(max_movie+1, dtype=int)
		z[unique_movies] = np.arange(n_movies)
		m_r = z[np_movies]

		# construct the ratings set from the three stacks
		np_ratings = ratings_df.rating.as_matrix()
		ratings = np.zeros((np_ratings.shape[0], 3), dtype=object)
		ratings[:, 0] = u_r
		ratings[:, 1] = m_r
		ratings[:, 2] = np_ratings
	else:
		ratings = ratings_df.as_matrix(['user_id', 'movie_id', 'rating'])
		# deal with 1-based user indices
		ratings[:, 0] -= 1
		ratings[:, 1] -= 1

	print('\nCreating train and test sets...')
	tr_sparse, test_sparse = _create_sparse_train_and_test(ratings, n_users, n_movies)

	return ratings[:, 0], ratings[:, 1], tr_sparse, test_sparse

# Given ratings, create sparse matrices for train and test sets.
def _create_sparse_train_and_test(ratings, n_users, n_movies):
	# pick a random test set of entries, sorted ascending
	test_set_size = len(ratings) / TEST_SET_RATIO
	test_set_idx = np.random.choice(xrange(len(ratings)),size=test_set_size, replace=False)
	test_set_idx = sorted(test_set_idx)

	# sift ratings into train and test sets
	ts_ratings = ratings[test_set_idx]
	tr_ratings = np.delete(ratings, test_set_idx, axis=0)

	# create training and test matrices as coo_matrix's
	u_tr, m_tr, r_tr = zip(*tr_ratings)
	tr_sparse = coo_matrix((r_tr, (u_tr, m_tr)), shape=(n_users, n_movies))

	u_ts, m_ts, r_ts = zip(*ts_ratings)
	test_sparse = coo_matrix((r_ts, (u_ts, m_ts)), shape=(n_users, n_movies))

	return tr_sparse, test_sparse

# Instantiate WALS model and use "simple_train" to factorize the matrix.
def train_model(args, tr_sparse):
	dim = args['latent_factors']
	num_iters = args['num_iters']
	reg = args['regularization']
	unobs = args['unobs_weight']
	wt_type = args['wt_type']
	feature_wt_exp = args['feature_wt_exp']
	obs_wt = args['feature_wt_factor']

	# generate model
	print('\nGenerating model...')
	input_tensor, row_factor, col_factor, model = wals.wals_model(tr_sparse, dim, reg, unobs, args['weights'], wt_type, feature_wt_exp, obs_wt)

	# factorize matrix
	print('\nTraining model...')
	session = wals.simple_train(model, input_tensor, num_iters)

	# evaluate output factor matrices
	output_row = row_factor.eval(session=session)
	output_col = col_factor.eval(session=session)

	# close the training session now that we've evaluated the output
	session.close()

	print('\nTrain finished.')
	return output_row, output_col
