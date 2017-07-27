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




targets = ['keg', 'party']
# [ 'chug',  'underagedrinking']
# beerfunnel, kegstand, passed, 
# frat, shotgunbeer, flipcup
# 'beerbong', 'beerpong', 'collegeparty',
# keg
# 'party',


# pseudocode
# for target in target:
#   load all paths from dir 
#   for each image, 
#   run it through the classifier
#   log probability in text file in corresponding directory





flickr_base = '../../Scraping/flickr/mmfeat/images_2/'
insta_base = '../../Scraping/instagram-scraper/instagram_scraper/instagram/'



instap_folders = ["beerbong/", "collegeparty/", "frathouse/", "kegstand/", "shotgunbeer/", "beerfunnel/" ,  "flipcup/",  "kegger/" ,   "partypeople/"]
flickrp_folders = ['beerbong/' , 'beerpong/', 'chuggingalcohol/', 'chuggingbeer/', 'drunkfrat/', 'fratpartydrunk/', 'kegstand/' , 'passedoutdrunk/', 'shotgunbeer/', 'underagedrinking/', 'overfittest/']



flickrn_folders = ['galaparty/', 'corporateparty/' ]




for target in targets:
    instap_folders = ["beerbong/", "collegeparty/", "frathouse/", "kegstand/", "shotgunbeer/", "beerfunnel/" ,  "flipcup/",  "kegger/" ,   "partypeople/"]
    flickrp_folders = ['beerbong/' , 'beerpong/', 'chuggingalcohol/', 'chuggingbeer/', 'drunkfrat/', 'fratpartydrunk/', 'kegstand/' , 'passedoutdrunk/', 'shotgunbeer/', 'underagedrinking/', 'overfittest/']
    instap_folders.remove('kegstand/')
    instap_folders.remove('collegeparty/')
    flickrp_folders = []



    if target != None:
        nipf = []
        nfpf = []
        for i in instap_folders:
            if target in i:
                print(i)
                nipf.append(i)
        for i in flickrp_folders:
            if target in i:
                nfpf.append(i)
        instap_folders = nipf
        flickrp_folders = nfpf

    print("target = " + target)
    NPY_PATH = './' + target + '/vgg16-' + target + '.npy'


    positive_example_folders = [insta_base + i for i in instap_folders]  
    positive_example_folders += [flickr_base + i for i in flickrp_folders]


    positive_example_folders = [insta_base + i for i in instap_folders]  
    positive_example_folders += [flickr_base + i for i in flickrp_folders]



    negative_example_folders = [flickr_base + i for i in flickrn_folders]



    print(positive_example_folders)

    all_positives = np.array([])

    for f in positive_example_folders:
        ls = os.listdir(f)[:] 
        all_positives = np.append(all_positives, [f + i for i in ls])

    all_negatives = np.array([])

    for f in negative_example_folders:
        ls = os.listdir(f)[:] 
        all_negatives = np.append(all_negatives, [f + i for i in ls])


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


    with tf.device('/gpu:0'):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.initialize_all_variables())


        for path in all_positives:
            print(path)
            img = utils.load_image(path)
            # label = [[1]]
            img = img.reshape((1,224,224,3))
            
            prob, pred = sess.run([vgg.prob, prediction], feed_dict={images: img,
                                                         # true_out: label,
                                                          train_mode: False})



            txt = '/'.join(path.split('/')[:-1]) + '/' + "probs.txt"
            if 'kegstand' not in txt and 'college' not in txt and 'frat' not in txt and 'dinner' not in txt and 'gala' not in txt:
                print(path, float(pred))
                with open(txt, 'a') as f:
                    f.write(path.split('/')[-1] +  " : " + str(float(pred)) +  '\n')
        sess.close()
