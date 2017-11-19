import tensorflow as tf
import numpy as np

#Start interactive session
sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Initial parameters
width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

#Input and output
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

#Converting images of the data set to tensors
x_image = tf.reshape(x, [-1,28,28,1])  

#Convolutional Layer 1
#Defining kernel weight and bias
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

#Convolve with weight tensor and add biases
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

#Apply the ReLU activation Function
h_conv1 = tf.nn.relu(convolve1)

#Apply the max pooling
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

#Convolutional Layer 2
#Weights and Biases of kernels
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

#Convolve image with weight tensor and add biases
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2

#Apply the ReLU activation Function
h_conv2 = tf.nn.relu(convolve2)

#Apply the max pooling
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

#Fully Connected Layer
#Flattening Second Layer
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

#Weights and Biases between layer 2 and 3
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

#Matrix Multiplication (applying weights and biases)
fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1

#Apply the ReLU activation Function
h_fc1 = tf.nn.relu(fcl)

#Dropout Layer, Optional phase for reducing overfitting
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer (Softmax Layer)
#Weights and Biases
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

#Matrix Multiplication (applying weights and biases)
fc=tf.matmul(layer_drop, W_fc2) + b_fc2

#Apply the Softmax activation Function
y_CNN= tf.nn.softmax(fc)

#Define functions and train the model
#Define the loss function
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

#Define the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Define prediction
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))

#Define accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Run session, train
sess.run(tf.global_variables_initializer())

for i in range(900):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
#Evaluate the model
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

sess.close() #finish the session
