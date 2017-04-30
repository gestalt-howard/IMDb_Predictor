import tensorflow as tf
import numpy as np
import sys
import os

#set a path to parent folder that holds the database, the input/output names, and name of neural net to be saved
#creates name for checkpoints and prediction files
if (len(sys.argv) == 5):
	parent_path = sys.argv[1]
	input_name = sys.argv[2]
	output_name = sys.argv[3]
	network_name = sys.argv[4]
else:
	parent_path = "C:/Documents/movie_review_proj/"
	input_name = "movie_input_vectors.npy"
	output_name = "movie_output_vectors.npy"
	network_name = "movie_fc_nn_model"

input_path = parent_path + input_name
output_path = parent_path + output_name
model_path = parent_path + network_name + "/"
network_path = model_path + network_name
checkpoint_path = model_path + "checkpoint"
prediction_path = model_path + "predictions.csv"

#load the input and output/label arrays
input_array = np.load(input_path)
label_array = np.load(output_path)

#check if the path to the neural net exists (create if not)
#check if a checkpoint exists (if neural net already ran before)
#open file to write predictions
start_epoch = 0
network_exist = 0
if not os.path.exists(model_path):
	os.mkdir(model_path)
pred_file = open(prediction_path, 'w')
pred_file.write("test_score,prediction\n")

if os.path.exists(checkpoint_path):
	network_exist = 1
	
#splice the input and output/label arrays into training and testing sets based on a given ratio (set at .8)
train_percent = .8
num_items = input_array.shape[0]
cutoff_idx = round(num_items * train_percent)
train_input_array = input_array[:cutoff_idx]
train_label_array = label_array[:cutoff_idx]
test_input_array = input_array[cutoff_idx+1:]
test_label_array = label_array[cutoff_idx+1:]

#set the learning rate, number of epochs, batch size, and display flag
learning_rate = 0.01
if network_exist == 0:
	training_epochs = 1000
else:
	training_epochs = 100
batch_size = 10
display_step = 1

#set number of hidden layer nodes for layers 1 and 2
#set number of input and number of output
n_hidden_1 = 256
n_hidden_2 = 256
n_input = len(input_array[0])
n_classes = len(label_array[0])

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_rate = tf.placeholder(tf.float32)

# Create model
def fully_connected_model(x, weights, biases, keep_rate):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	drop_out = tf.nn.dropout(layer_2, keep_rate)
	out_layer = tf.matmul(drop_out, weights['out']) + biases['out']
	return out_layer

# Store layers weight & bias
weights = {'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), 'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))}
biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])), 'b2': tf.Variable(tf.random_normal([n_hidden_2])), 'out': tf.Variable(tf.random_normal([n_classes]))}

# Construct model
pred = fully_connected_model(x, weights, biases, keep_rate)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

Saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	if (network_exist == 1):
		Saver.restore(sess, network_path)
		print("Model restored from file: %s" % network_path)
    # Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(train_input_array.shape[0]/batch_size)
        # Loop over all batches
		for i in range(total_batch):
			batch_beg = batch_size*i
			batch_end = batch_size*(i+1)
			if (batch_end >= train_input_array.shape[0]):
				batch_end = train_input_array.shape[0]
			batch_x = train_input_array[batch_beg:batch_end]
			batch_y = train_label_array[batch_beg:batch_end]
            # Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_rate: .8})
            # Compute average loss
			avg_cost += c / total_batch
        # Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
		Saver.save(sess, network_path)
	print("Optimization Finished!")

    # Test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	prediction_errors = tf.abs(tf.subtract(tf.argmax(pred, 1), tf.argmax(y, 1)))
    # Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	mean_error = tf.reduce_mean(tf.cast(prediction_errors, "float"))
	print("Accuracy:", accuracy.eval({x: test_input_array, y: test_label_array, keep_rate: 1}))
	print("Mean error:", mean_error.eval({x: test_input_array, y: test_label_array, keep_rate: 1}))
	
	#make and save predictions
	pred_y = sess.run(pred, feed_dict={x: test_input_array, keep_rate: 1})
	for idx in range(test_label_array.shape[0]):
		prediction = np.argmax(pred_y[idx])
		true_label = np.argmax(test_label_array[idx])
		pred_file.write('{:0.2f},{:0.2f}\n'.format(float(true_label),float(prediction)))