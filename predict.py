import os
import tensorflow as tf
import numpy as np
import random

# Load model
print('\nLoading model...')
cwd = os.getcwd()
path = os.path.join(cwd, 'model')

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

def random_n_recommendations_for(n, recommendations, unseen_movies):
	unseen_movies_ids = [i for (i, x) in unseen_movies]
	unseen_recommendation_tuples = [i for i in recommendations if i[0] in unseen_movies_ids]
	print('UNSEEN RECOMMENDATIONS COUNT')
	print(len(unseen_recommendation_tuples))
	print('UNSEEN COUNT')
	print(len(unseen_movie_list))

	filtered_list = filter_movies_rating_greater_equal(unseen_recommendation_tuples, 3)
	print(filtered_list)
	return sort_rated_movie_tuples(random.sample(filtered_list, n))

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
	sample_user = X_test_np[150]
	#get the predicted ratings
	print('\nGet predicted ratings for user:')
	sample2 = np.random.randint(0, 5, (3706,)) / 5
	user_pred = normalize_recommendation(sess.run(output_layer, feed_dict={input_layer:[sample_user]}))
	user_pred2 = normalize_recommendation(sess.run(output_layer, feed_dict={input_layer:[sample2]}))

	unseen_movie_list = unseen_movies(rating_to_tuple([sample2]))

	recommendations = random_n_recommendations_for(20, rating_to_tuple(user_pred2), unseen_movie_list)
	print('SORTED RECOMMENDATIONS')
	print(recommendations)


	# print(np.max(user_pred))
	# print(np.mean(user_pred))
	# print(np.sort((np.round(-user_pred))))
	# print('pred 2: ')
	# print(np.max(user_pred2))
	# print(np.mean(user_pred2))
	# print(np.sort((np.round(-user_pred2))))
	# print(rating_to_tuple(user_pred2))
	# tuple_array = np.array(rating_to_tuple(user_pred2), dtype=[('index', int), ('rating', float)])
	# tuple_array.sort(order='rating')
	# print(tuple_array[::-1])
	