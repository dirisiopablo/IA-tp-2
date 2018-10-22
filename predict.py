import os
import tensorflow as tf
import numpy as np

# Load model
print('\nLoading model...')
cwd = os.getcwd()
path = os.path.join(cwd, 'model')
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
	sample2 = np.random.randint(0, 1, (3706,))
	user_pred = sess.run(output_layer, feed_dict={input_layer:[sample_user]})
	user_pred2 = sess.run(output_layer, feed_dict={input_layer:[sample2]})

	print(np.round(user_pred * 5))
	print(np.max(user_pred2) * 5)
	print(np.mean(user_pred2) * 5)
	print('pred 2: ')
	print(np.max(user_pred2 * 5))
	print(np.mean(user_pred2) * 5)
	print(np.round(user_pred2 * 5))