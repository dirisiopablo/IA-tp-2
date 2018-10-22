import os
import tensorflow as tf

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
	sample_user = X_test_np[98,:]
	#get the predicted ratings
	print('\nGet predicted ratings for user:')
	user_pred = sess.run(output_layer, feed_dict={input_layer:[sample_user]})
	print(user_pred)