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

import pickle


keyword_targets = ['beerbong beerfunnel', 'beerpong',  'shotgunbeer', 'flipcup', \
    'keg', 'passed', 'chug',  'gala corporate']

    # 'frat','fratparty collegeparty partypeople',

# ommitting underage drinking b/c of bad results

# OUTPUT_DIM = 1000
OUTPUT_DIM = 2
VALIDATE = True
TRAIN = True
OVERSAMPLE = -1 
TRIAL = 10 
# if len(sys.argv) > 1:
#     target = sys.argv[1]
# else:
#     target = None

# if target != None:
#     print("target = " + target)
#     NPY_PATH = './' + target + '/vgg16-' + target + '.npy'
# else:
#     print("target = drinking")
#     NPY_PATH = './drinking/vgg16-drinking.npy'

target = None
NPY_PATH = './drinkingbin/vgg16-drinkingbin' + str(TRIAL) + '.npy'


flickr_base = '../../Scraping/flickr/mmfeat/images_2/'
insta_base = '../../Scraping/instagram-scraper/instagram_scraper/instagram/'



instap_folders = ["beerbong/", "kegstand/", "shotgunbeer/", "beerfunnel/" ,  "flipcup/",  "kegger/" ]
flickrp_folders = ['beerbong/' , 'beerpong/', 'chuggingalcohol/', 'chuggingbeer/', 'kegstand/' , 'passedoutdrunk/', 'shotgunbeer/', ]
 # "collegeparty/", "frathouse/",   "partypeople/"
# 'underagedrinking/', 'drunkfrat/', 'fratpartydrunk/'

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


target = 'drinking_bin' + str(TRIAL)


flickrn_folders = [ 'picture/', 'jpg/', 'imgnet/']
flickrn_folders += ['galaparty/']
# 'fieldday/', 'outdoorconcert/', 'galaparty', 'corporateparty'
positive_example_folders = [insta_base + i for i in instap_folders]
positive_example_folders += [flickr_base + i for i in flickrp_folders]



negative_example_folders = [flickr_base + i for i in flickrn_folders]


print(positive_example_folders)

all_positives = np.array([])

for f in positive_example_folders:
    ls2 = []
    ls = np.array(os.listdir(f)[:])
    for i in range(ls.shape[0]):
        if i < ls.shape[0] and ls[i][-3:] == 'txt':
            pass
        else:
            ls2+=[ls[i]]
    
    all_positives = np.append(all_positives, [f + i for i in ls2])

all_negatives = np.array([])






for f in negative_example_folders:
    count = 0
    ls2 = []
    ls = np.array(os.listdir(f)[:])
    for i in range(ls.shape[0]):            
        if i < ls.shape[0] and (ls[i][-3:] == 'txt' or count > (all_positives.shape[0] / len(negative_example_folders) )):
            pass
        else:
            ls2 += [ls[i]]
        count += 1

    all_negatives = np.append(all_negatives, [f + i for i in ls2])

print(all_negatives.shape, all_positives.shape)


# all_positives = np.concatenate([all_positives, all_negatives])
all_labels = None

# all_positives = all_positives[::20]
pos_labels = None
neg_labels = None

target_counts = {}
for t in keyword_targets:
    target_counts[t] = 0


for img in all_positives:
    label = [1]
    if pos_labels is None:
        pos_labels = label
    else:
        pos_labels = pos_labels + label

for img in all_negatives:
    label = [0]
    if neg_labels is None:
        neg_labels = label
    else:
        neg_labels = neg_labels + label

pos_labels = np.array(pos_labels)
neg_labels = np.array(neg_labels)

print(pos_labels, neg_labels)
all_labels = np.concatenate([pos_labels, neg_labels])
all_positives = np.concatenate([all_positives, all_negatives])

import random

# all_labels = np.array(all_labels)

print(all_labels)
# c = list(zip(all_positives, all_labels))
# print(c)
if not os.path.exists("shufflebin" + str(TRIAL) + ".pickle"):
    shuffle = np.arange(len(all_labels))
    np.random.shuffle(shuffle)
    with open('shufflebin' + str(TRIAL) + '.pickle', 'wb') as f:
        pickle.dump(shuffle, f, protocol = pickle.HIGHEST_PROTOCOL)
else:
    with open('shufflebin' + str(TRIAL) + '.pickle', 'rb') as f:
        shuffle = pickle.load(f) 

all_labels = all_labels[shuffle]
all_positives = all_positives[shuffle]

# all_positives, all_labels = zip(*c)

# all_positives = np.array(all_positives)
# all_labels = np.array(all_labels)

val_set_size = len(all_positives) / 10

if VALIDATE:
    val_set = all_positives[-1 * val_set_size:]
    val_set_labels = all_labels[-1 * val_set_size:]

    all_positives = all_positives[:-1* val_set_size]
    all_labels = all_labels[:-1* val_set_size]


def load_val_batch(start, finish):


    imgs = np.array([utils.load_image(i) for i in val_set[start:finish]])
    labels = np.array([i for i in val_set_labels[start:finish]])
    return imgs, labels



def load_batch(batch_size = 256):
    idxs = np.random.choice(all_positives.shape[0], size = batch_size, replace = False)
    img_paths = [all_positives[i] for i in idxs ]
    imgs = [utils.load_image(i) for i in img_paths]
    labels = [all_labels[i] for i in idxs]
    return imgs, labels


# img1 = utils.load_image("./test_data/tiger.jpeg")
# img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger
# batch1 = img1.reshape((1, 224, 224, 3))









# with tf.device('/gpu:0'):
#     img_count = 256

#     sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))




#     # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
#     print(vgg.get_var_count())


#     # test classification
#     prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
#     utils.print_prob(prob[0], './synset.txt')

#     # simple 1-step training
#     cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
#     train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
#     sess.run(train, feed_dict={images: batch1, true_out: [imgs_true_result], train_mode: True})

#     # test classification again, should have a higher probability about tiger
#     prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
#     utils.print_prob(prob[0], './synset.txt')

#     # test save
#     vgg.save_npy(sess, './vgg19-drinking-save.npy')


batch_size = 16
training_iters = 44000 # 100000
display_step = 20
validate_step = 200
save_copy_step = 400
max_accuracy = 0
iters_before_no_new_max = 0
new_max_iters_cutoff = 4


with tf.device('/gpu:1'):



    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.int64, [None])
    train_mode = tf.placeholder(tf.bool)



    vgg = vgg19.Vgg19(vgg19_npy_path= (NPY_PATH if os.path.isfile(NPY_PATH) else 'vgg16.npy'), output_path="./vgg16-drinking.npy", output_dim = OUTPUT_DIM)
    vgg.build(images, train_mode)



    learning_rate = .0001

    # Define loss and Optimizer

    # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_out, logits=vgg.prob))

    prediction = vgg.prob
    predicted_class = tf.argmax(prediction, 1)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_out, logits=prediction))
    correct = tf.equal(predicted_class, true_out)
    accuracy = tf.reduce_mean( tf.cast(correct, 'float') )


    # predicted_class = tf.greater(prediction,0.5)
    # actual_label = tf.argmax(true_out, 1)
    # correct = tf.equal(predicted_class, tf.argmax(true_out, 1))
    # accuracy = tf.reduce_mean( tf.cast(correct, 'float') )


    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


    # Evaluate model
    # correct_pred = tf.equal(tf.argmax(vgg.prob, 1) , tf.argmax(true_out, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.initialize_all_variables())



    step = 1
    # Keep training until reach max iterations
    if TRAIN:
        while step * batch_size < training_iters:
            imgs, labels = load_batch(batch_size)
            batch = np.array(imgs).reshape((-1, 224, 224, 3))
            # Run optimization op (backprop)        
            sess.run(train, feed_dict={images: batch, true_out: labels, train_mode: True})

            if step % display_step == 0:
                # prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
                loss, acc, cpred, pc,pred = sess.run([cost, accuracy, correct, predicted_class,vgg.prob], feed_dict={images: imgs,
                                                                 true_out: labels,
                                                                  train_mode: False})
                print(labels)
                print(pc)
                print(pred)
                print(acc, cpred)
                print(loss)

                
            if VALIDATE:
                if step % validate_step == 0:
                    tloss = 0.0
                    tlabels = None
                    tpc = None
                    tpred = None
                    tacc = 0.0
                    tcpred = None

                    count = 0.0
                    for i in range(len(val_set) / batch_size):
                        
                        count+=1.0
                        start = i*batch_size
                        finish = min((i+1)*batch_size, len(val_set))

                        imgs, labels = load_val_batch(start, finish)

                        # imgs = val_set[start:finish]
                        # labels = val_set_labels[start:finish]

                        loss, acc, cpred, pc, pred = sess.run([cost, accuracy, correct, predicted_class,vgg.prob], feed_dict={images: imgs,
                                                                     true_out: labels,
                                                                      train_mode: False})
                        tloss += loss
                        tacc += acc
                        print(tcpred, cpred)
                        if tcpred is None:
                            tcpred = cpred
                        else:
                            tcpred = np.concatenate([tcpred, cpred])
                        if tlabels is None:
                            tlabels = labels
                        else:
                            tlabels = np.concatenate([tlabels, labels])
                        if tpred is None:
                            tpred = pred
                        else:
                            tpred = np.concatenate([tpred, pred])
                        if tpc is None:
                            tpc = pc
                        else:
                            tpc  = np.concatenate([tpc, pc])
                    print(len(val_set))

                    tacc /= count
                    tloss /= count

                    print('validation set')
                    # print(tlabels)
                    # print(tpc)
                    # print(tpred)
                    print("Accuracy: "  + str(tacc))
                    print("Loss: " + str(tloss))


                    if tacc > max_accuracy:
                        max_accuracy = tacc
                        iters_before_no_new_max = 0
                    else:
                        iters_before_no_new_max += 1

                    if iters_before_no_new_max > new_max_iters_cutoff:
                        break

                    
                    vgg.save_npy(sess, NPY_PATH )
                    if step % save_copy_step  == 0:
                        vgg.save_npy(sess, NPY_PATH[:-4] + '_1_' + str(step/save_copy_step) + '.npy')
                    

                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(tloss) + ", Training Accuracy= " + \
                          "{:.5f}".format(tacc))


            step += 1


print("Optimization Finished")




tloss = 0.0
tlabels = None
# tpc = np.array([True for i in range(OUTPUT_DIM)])
tpc = None
tpred = None
tacc = 0.0
tcpred = None
tal = None

count = 0.0
for i in range(len(val_set) / batch_size):
    
    count+=1.0
    start = i*batch_size
    finish = min((i+1)*batch_size, len(val_set))

    imgs, labels = load_val_batch(start, finish)

    # imgs = val_set[start:finish]
    # labels = val_set_labels[start:finish]

    loss, acc, cpred, pc, pred = sess.run([cost, accuracy, correct, predicted_class,vgg.prob], feed_dict={images: imgs,
                                                 true_out: labels,
                                                  train_mode: False})
    tloss += loss
    tacc += acc
    if tal is None:
        tal = labels 
    else:
        tal = np.concatenate([tal, labels])

    if tcpred is None:
        tcpred = cpred
    else:
        tcpred = np.concatenate([tcpred, cpred])
    if tlabels is None:
        tlabels = labels
    else:
        tlabels = np.concatenate([tlabels, labels])
    if tpred is None:
        tpred = pred
    else:
        tpred = np.concatenate([tpred, pred])
    if tpc is None:
        tpc = pc
    else:
        tpc  = np.concatenate([tpc, pc])




print(len(val_set))

tacc /= count
tloss /= count

print('validation set')
# print(tlabels)
# print(tpc)
# print(tpred)
print("Accuracy: "  + str(tacc))
print("Loss: " + str(tloss))

tpc = tpc.astype(int)
print(tal)
print(tpc)
print(tal.shape, tpc.shape)
print(type(tal), type(tpc))
# confusion = tf.contrib.metrics.confusion_matrix( labels = tal , predictions = tpc)
# print(confusion)


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(tal, tpc))
print(confusion_matrix(tal, tpc))
print(len(val_set))
print(len(all_positives))
ipc = tpc.astype(int)
# sess = tf.Session()
# with sess.as_default(): 
#    print(confusion.eval())
# print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={images: val_set,
#                                       true_out : val_set_labels,
#                                       train_mode : False}))
