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


from skimage.io import imshow
from matplotlib import pyplot as plt


keyword_targets = ['beerbong beerfunnel', 'beerpong', 'shotgunbeer', 'flipcup', \
    'keg', 'passed', 'chug', \
    'gala corporate']


# OUTPUT_DIM = 1000
OUTPUT_DIM = len(keyword_targets)
VALIDATE = True
TRAIN = True 

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
NPY_PATH = './drinkingoh_boosted2/vgg16-drinkingoh-boosted.npy'


flickr_base = '../../Scraping/flickr/mmfeat/images_2/'
insta_base = '../../Scraping/instagram-scraper/instagram_scraper/instagram/'



instap_folders = ["beerbong/",  "kegstand/", "shotgunbeer/", "beerfunnel/" ,  "flipcup/",  "kegger/" ]
flickrp_folders = ['beerbong/' , 'beerpong/', 'chuggingalcohol/', 'chuggingbeer/',  'kegstand/' , 'passedoutdrunk/', 'shotgunbeer/' ]

# "frathouse/", "partypeople/" , 'drunkfrat/', 'fratpartydrunk/', 'underagedrinking/', "collegeparty/", 


confs = {}
for f in instap_folders:
    count = 0
    with open(insta_base + f + 'probs.txt', 'r') as probs:
        for l in probs.readlines():
            l = l.split(':')
            confs[insta_base + f + l[0].strip()] = float(l[1].strip())
            count+=1
    print(f, count)

for f in flickrp_folders:
    count = 0
    with open(flickr_base + f + 'probs.txt', 'r') as probs:
        for l in probs.readlines():
            l = l.split(':')
            confs[flickr_base + f + l[0].strip()] = float(l[1].strip())
            count += 1
    print(f, count)

print(len(confs.values()))
x = np.array(confs.values())
print(len(x[x > .9]))
print(len(x[x > .8]))
print(len(x[x > .7]))
print(len(x[x > .6]))
print(len(x[x > .5]))
print(len(x[x > .4]))
print(len(x[x > .3]))
print(len(x[x > .2]))
print(len(x[x > .1]))
print(len(x[x > .05]))
print(len(x[x > .03]))
print(len(x[x > .01]))
# for i in confs:
#     if confs[i] < .9:
#         if 'chug' not in i:
#             continue
#         print(i.split('/')[-2])
#         im = utils.load_image(i)
#         imshow(i)
#         plt.show()
#         # t = raw_input()



THRESHOLD = .01

all_positives = np.array([])


for i in confs:
    if confs[i] > THRESHOLD:
        all_positives = np.append(all_positives, i)





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


target = 'drinking_oh_boosted2'


flickrn_folders = ['galaparty/', 'corporateparty/' ]

# positive_example_folders = [insta_base + i for i in instap_folders]  
# positive_example_folders += [flickr_base + i for i in flickrp_folders]



negative_example_folders = [flickr_base + i for i in flickrn_folders]


# print(positive_example_folders)


# for f in positive_example_folders:
#     ls = np.array(os.listdir(f)[:])
#     for i in range(ls.shape[0]):
#         if i < ls.shape[0] and ls[i][-3:] == 'txt':
#             ls = np.delete(ls, i)

#     all_positives = np.append(all_positives, [f + i for i in ls])

all_negatives = np.array([])



neg_limit = 10000
num_neg_folders = len(negative_example_folders)

for f in negative_example_folders:
    ls = np.array(os.listdir(f)[:])
    count = 0
    for i in range(ls.shape[0]):
	if count > neg_limit / num_neg_folders:
	    break
	count+=1 
        if i < ls.shape[0] and ls[i][-3:] == 'txt':
            ls = np.delete(ls ,i)
    all_negatives = np.append(all_negatives, [f + i for i in ls])


all_positives = np.concatenate([all_positives, all_negatives])
all_labels = None

# all_positives = all_positives[::20]


target_counts = {}
for t in keyword_targets:
    target_counts[t] = 0


for img in all_positives:
    folder = img.split('/')[-2]
    label = None
    for targ in keyword_targets:
        for t in targ.split():
            if t in folder:
                target_counts[targ] += 1
                if label == None:
                    label = [1 if i == keyword_targets.index(targ) else 0 for i in range(len(keyword_targets))]
                else:
                    label[keyword_targets.index(targ)] = 1

    assert label != None
    if all_labels == None:
        all_labels = [label]
    else:
        all_labels = all_labels + [label]

for k in target_counts:
    print(k, target_counts[k])


import random

all_labels = np.array(all_labels)

# print(all_labels)
# c = list(zip(all_positives, all_labels))
# print(c)
if not os.path.exists("shuffle_boosted2.pickle"):
    shuffle = np.arange(len(all_labels))
    np.random.shuffle(shuffle)
    with open('shuffle_boosted2.pickle', 'wb') as f:
        pickle.dump(shuffle, f, protocol = pickle.HIGHEST_PROTOCOL)
else:
    with open('shuffle_boosted2.pickle', 'rb') as f:
        shuffle = pickle.load(f) 

all_labels = all_labels[shuffle]
all_positives = all_positives[shuffle]

# all_positives, all_labels = zip(*c)

# all_positives = np.array(all_positives)
# all_labels = np.array(all_labels)

# print(all_labels)


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


batch_size = 12
training_iters = 44000 # 100000
display_step = 20
validate_step = 200
save_copy_step = 400
max_accuracy = 0
iters_before_no_new_max = 0
new_max_iters_cutoff = 4

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, OUTPUT_DIM])
train_mode = tf.placeholder(tf.bool)



vgg = vgg19.Vgg19(vgg19_npy_path= (NPY_PATH if os.path.isfile(NPY_PATH) else 'vgg16.npy'), output_path="poop", output_dim = OUTPUT_DIM)
vgg.build(images, train_mode)



learning_rate = .0001

# Define loss and Optimizer

# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_out, logits=vgg.prob))

prediction = tf.nn.softmax(vgg.prob)
predicted_class = tf.argmax(vgg.prob, 1)
cost = tf.reduce_sum((prediction - true_out) ** 2)

# predicted_class = tf.greater(prediction,0.5)
actual_label = tf.argmax(true_out, 1)
correct = tf.equal(predicted_class, tf.argmax(true_out, 1))
accuracy = tf.reduce_mean( tf.cast(correct, 'float') )


train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Evaluate model
# correct_pred = tf.equal(tf.argmax(vgg.prob, 1) , tf.argmax(true_out, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.device('/gpu:1'):
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
                    tlabels = np.array([[1 for i in  range(OUTPUT_DIM)]])
                    tpc = np.array([True for i in range(OUTPUT_DIM)])
                    tpred = np.array([[.1 for i in range(OUTPUT_DIM)]])
                    tacc = 0.0
                    tcpred = np.array([True for i in range(OUTPUT_DIM)])


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
                        tcpred = np.concatenate([tcpred, cpred])
                        tlabels = np.concatenate([tlabels, labels])
                        tpred = np.concatenate([tpred, pred])
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
tlabels = np.array([[1 for i in  range(OUTPUT_DIM)]])
# tpc = np.array([True for i in range(OUTPUT_DIM)])
tpc = None
tpred = np.array([[.1 for i in range(OUTPUT_DIM)]])
tacc = 0.0
tcpred = np.array([True for i in range(OUTPUT_DIM)])
tal = None

count = 0.0
for i in range(len(val_set) / batch_size):
    
    count+=1.0
    start = i*batch_size
    finish = min((i+1)*batch_size, len(val_set))

    imgs, labels = load_val_batch(start, finish)

    # imgs = val_set[start:finish]
    # labels = val_set_labels[start:finish]

    loss, acc, cpred, pc, pred, al = sess.run([cost, accuracy, correct, predicted_class,vgg.prob, actual_label], feed_dict={images: imgs,
                                                 true_out: labels,
                                                  train_mode: False})
    tloss += loss
    tacc += acc
    if tal is None:
        tal = al
    else:
        tal = np.concatenate([tal, al])

    tcpred = np.concatenate([tcpred, cpred])
    tlabels = np.concatenate([tlabels, labels])
    tpred = np.concatenate([tpred, pred])
    if tpc is None:
        tpc = pc
    else:
        tpc = np.concatenate([tpc, pc])




print(len(val_set))

tacc /= count
tloss /= count

print('validation set')
# print(tlabels)
# print(tpc)
# print(tpred)
print("Accuracy: "  + str(tacc))
print("Loss: " + str(tloss))

print(tal)
print(tpc)
print(tal.shape, tpc.shape)

print(type(tal), type(tpc))
confusion = tf.contrib.metrics.confusion_matrix( predictions = tpc,labels = tal) 
print(confusion)


from sklearn.metrics import classification_report

print(classification_report(tal, tpc, target_names=keyword_targets))


sess = tf.Session()
with sess.as_default(): 
    print(confusion.eval())
# print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={images: val_set,
#                                       true_out : val_set_labels,
#                                       train_mode : False}))
