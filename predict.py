import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd

CONST_USER_INDEX = 150

# Load model
print('\nLoading model...')
cwd = os.getcwd()
path = os.path.join(cwd, 'model')
movies = pd.read_csv('dataset/movies.dat', sep="::", header = None, engine='python')[1]

def rating_to_tuple(ratings):
	return [(i,x) for i,x in enumerate(np.round(ratings)[0])]

def unseen_movies(rating_tuple):
	return [t for t in rating_tuple if t[1] == 0.0]

def normalize_recommendation(recommendation):
	return np.round(recommendation * 5)

def sort_rated_movie_tuples(rated_movies_tuples):
	tuple_array = np.array(rated_movies_tuples, dtype=[('index', int), ('rating', float)])
	tuple_array.sort(order='rating')
	return tuple_array[::-1]

def filter_movies_rating_greater_equal(tuple_array, threshold):
	return [t for t in tuple_array if t[1] >= threshold]

def set_movies_description(tuple_array):
	return list(map(lambda a: (movies[a[0]], a[1]), tuple_array))

def random_n_recommendations_for(n, recommendations, unseen_movies):
	unseen_movies_ids = [i for (i, x) in unseen_movies]
	unseen_recommendation_tuples = [i for i in recommendations if i[0] in unseen_movies_ids]

	filtered_list = filter_movies_rating_greater_equal(unseen_recommendation_tuples, 3)
	return set_movies_description(sort_rated_movie_tuples(random.sample(filtered_list, n)))

with tf.Session() as sess:
	saver = tf.train.import_meta_graph(path + '/model-1000.meta')
	saver.restore(sess,tf.train.latest_checkpoint(path))
	graph = tf.get_default_graph()
	# restore layers
	input_layer = graph.get_tensor_by_name('input_layer:0')
	output_layer = graph.get_tensor_by_name('output_layer:0')
	# pick a user
	X_test = graph.get_tensor_by_name('X_test:0')
	X_test_np = X_test.eval()
	
	#get the predicted ratings for sample user ratings
	sample_user_ratings = X_test_np[CONST_USER_INDEX]
	sample_user_pred = normalize_recommendation(sess.run(output_layer, feed_dict={input_layer:[sample_user_ratings]}))

	unseen_movie_list = unseen_movies(rating_to_tuple([sample_user_ratings]))
	recommendations_for_random = random_n_recommendations_for(20, rating_to_tuple(sample_user_pred), unseen_movie_list)
	print('SORTED RECOMMENDATIONS FOR SAMPLE USER')
	print(recommendations_for_random)
	
	#get the predicted ratings for user random ratings
	print('\nGet predicted ratings for user:')
	random_user_ratings = np.random.randint(0, 5, (3706,)) / 5
	random_user_pred = normalize_recommendation(sess.run(output_layer, feed_dict={input_layer:[random_user_ratings]}))

	unseen_movie_list = unseen_movies(rating_to_tuple([random_user_ratings]))
	recommendations_for_random = random_n_recommendations_for(20, rating_to_tuple(random_user_pred), unseen_movie_list)
	print('SORTED RECOMMENDATIONS FOR RANDOM USER')
	print(recommendations_for_random)