#! /usr/bin/env python
'''
This is a deep CNN that trains and evaluates on the ARCENE dataset which main objective is to
distinguish cancer versus normal patterns from massspectrometric data. The Python programming language
is used in combination with TensorFlow, which uses a highly efficient C++ backend for computations.
'''


#Modules
import pdb
import tensorflow as tf
import numpy as np
import csv

#Start ineractive session in tf
sess = tf.InteractiveSession()

#Read files
def file_reader(file_path):
    '''Input = file path (str)
       Output = numpy array of items in files
    '''
    
    data = []
    with open(file_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\n')
        for row in reader:
            for x in row:
                x=x.split(' ')
                example = []
                for item in x:
                    if item:
                        item = int(item) #convert to int
                        example.append(item)
                data.append(example)
        data = np.asarray(data)
    return data

#Create one hot vectors
def create_one_hot(array):
    '''Creates a one hot vector from an array of +/- 1 labels
    '''
    one_hot = []
    for item in array:
        if item == -1:
            one_hot.append(0)
        else:
            one_hot.append(1)
            
    depth = 2
    array = tf.one_hot(one_hot, depth)
    return array

    
#Lists with data lists
#train data
train_data = file_reader('arcene_train.data')
#train labels
train_labels = create_one_hot(file_reader('arcene_train.labels'))
y_ = train_labels #used when training
#validation data
valid_data = file_reader('arcene_valid.data')
#validation labels
valid_labels = create_one_hot(file_reader('arcene_valid.labels'))

pdb.set_trace()
#Placeholders
#None indicates a variable batch size
x = tf.placeholder(tf.float32, shape=[100, 10000]) #features

#Weights and biases
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution and maxpooling
def conv2d(x, W): 
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#First convolutional layer
#Dimension 1&2 = patch size, 3 = input channels, 4 = number of output channels
#32 features for each 5x5 patch
W_conv1 = weight_variable([5, 5, 1, 32]) #no overlaps - better with?
#bias vector, one component for each output channel
b_conv1 = bias_variable([32])

#Reshape x to a 4d tensor to apply layer
#-1 indicates a dimension computed by tf
#[batch, height, width, channel]
x_image = tf.reshape(x, [-1,100,100,1]) #effectively a 100x100 image

#Convolve x_image with the weight tensor,
#add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #Reduces image to 50x50

#Second convolutional layer - reduces image to 25x25
#64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #Reduces image to 25x25

#Densely connected layer
#Add a fully connected layer (1024 neurons) to allow processing
#of the entire image
W_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*64])#reshape into a batch of vectors
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout - reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
#Training using step 1e-4
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#Correct predictions and accuracy (should be done on validation set)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Initalize variables and run
sess.run(tf.global_variables_initializer())
for i in range(1000): 
  if i%10 == 0: #print training and validation accuracy every 10th step
    train_accuracy = accuracy.eval(feed_dict={
        x:train_data , keep_prob: 1.0}) #keep_prob: 1.0 specifies no drop out
    print("step %d, training accuracy %g"%(i, train_accuracy))
    
  #make sure the training labels are used when training 
  train_step.run(feed_dict={x:train_data, keep_prob: 0.5}) #have to rely on random subset (50 %) of neuron output --> has to be robust

#Validate model
y_ = valid_labels #set validation labels to validate
print("validation accuracy %g"%accuracy.eval(feed_dict={
        x: valid_data, keep_prob: 1.0}))

