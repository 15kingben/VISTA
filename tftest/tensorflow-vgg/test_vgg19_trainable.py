"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import numpy as np

import vgg19_trainable as vgg19
import utils

import os
import os.path
import sys
import random

# OUTPUT_DIM = 1000
OUTPUT_DIM = 1

if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = None

if target != None:
    print("target = " + target)
    NPY_PATH = './vgg16-' + target + '.npy'
else:
    print("target = drinking")
    NPY_PATH = './vgg16-drinking.npy'

testdir = '../../Scraping/flickr/mmfeat/images_2/hot dog/'


flickr_base = '../../Scraping/flickr/mmfeat/images_2/'


flickrn_folders = ['galaparty/', 'corporateparty/' ]

positive_example_folders = [testdir]

negative_example_folders = [flickr_base + i for i in flickrn_folders]




print(positive_example_folders)

all_positives = np.array([])

for f in positive_example_folders:
    ls = os.listdir(f)[:-2] 
    all_positives = np.append(all_positives, [f + i for i in ls])

all_negatives = np.array([])

for f in negative_example_folders:
    ls = os.listdir(f)[:-2] 
    all_negatives = np.append(all_negatives, [f + i for i in ls])




val_set = np.concatenate([all_positives[:] , all_negatives[0:len(all_positives)]])
val_set = np.array([utils.load_image(i) for i in val_set])
val_set_labels = np.concatenate( [ [ [0] for i in range(len(all_positives))]  , [ [1] for i in range(len(all_positives))] ])


c = list(zip(val_set, val_set_labels))

random.shuffle(c)

val_set, val_set_labels = zip(*c)



batch_size = 16
training_iters = 1000000 # 100000
display_step = 20
validate_step = 50
save_copy_step = 400

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, OUTPUT_DIM])
train_mode = tf.placeholder(tf.bool)



vgg = vgg19.Vgg19(vgg19_npy_path= (NPY_PATH if os.path.isfile(NPY_PATH) else 'vgg16.npy'), output_path="./vgg16-drinking.npy")
vgg.build(images, train_mode)



learning_rate = .0001

# Define loss and Optimizer
# cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=true_out, logits=vgg.prob))

# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_out, logits=vgg.prob))

prediction = tf.sigmoid(vgg.prob)
predicted_class = tf.greater(prediction,0.5)
correct = tf.equal(predicted_class, tf.equal(true_out,1.0))
accuracy = tf.reduce_mean( tf.cast(correct, 'float') )


train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Evaluate model
# correct_pred = tf.equal(tf.argmax(vgg.prob, 1) , tf.argmax(true_out, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.device('/gpu:1'):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.initialize_all_variables())


               
    tloss = 0.0
    tlabels = np.array([[1]])
    tpc = np.array([[True]])
    tpred = np.array([[.1]])
    tacc = 0.0
    tcpred = np.array([[True]])


    count = 0.0
    for i in range(len(val_set) / batch_size):
        
        count+=1.0
        start = i*batch_size
        finish = min((i+1)*batch_size, len(val_set))

        imgs = val_set[start:finish]
        labels = val_set_labels[start:finish]

        loss, acc, cpred, pc, pred = sess.run([cost, accuracy, correct, predicted_class,vgg.prob], feed_dict={images: imgs,
                                                     true_out: labels,
                                                      train_mode: False})
        tloss += loss
        tacc += acc
        tcpred = np.concatenate([tcpred, cpred])
        tlabels = np.concatenate([tlabels, labels])
        tpred = np.concatenate([tpred, pred])
        tpc  = np.concatenate([tpc, pc])
    print(len(val_set))
    print(val_set_labels)
    print( tpc)
    tacc /= count
    tloss /= count

    print('validation set')
    # print(tlabels)
    # print(tpc)
    # print(tpred)
    print("Accuracy: "  + str(tacc))
    print("Loss: " + str(tloss))



    print(", Minibatch Loss= " + \
          "{:.6f}".format(tloss) + ", Training Accuracy= " + \
          "{:.5f}".format(tacc))
