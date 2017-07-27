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


# OUTPUT_DIM = 1000
OUTPUT_DIM = 1
VALIDATE = True

if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = None

if target != None:
    print("target = " + target)
    NPY_PATH = './' + target + '/vgg16-' + target + '.npy'
else:
    print("target = drinking")
    NPY_PATH = './drinking/vgg16-drinking.npy'

flickr_base = '../../Scraping/flickr/mmfeat/images_2/'
insta_base = '../../Scraping/instagram-scraper/instagram_scraper/instagram/'



instap_folders = ["beerbong/", "kegstand/", "shotgunbeer/", "beerfunnel/" ,  "flipcup/",  "kegger/" ]
flickrp_folders = ['beerbong/' , 'beerpong/', 'chuggingalcohol/', 'chuggingbeer/', 'kegstand/' , 'passedoutdrunk/', 'shotgunbeer/' ]
# "collegeparty/",,   "partypeople/" , "frathouse/", 
# , 'drunkfrat/', 'fratpartydrunk/', 'underagedrinking/', 'overfittest/'

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




flickrn_folders = ['galaparty/', 'corporateparty/' ]

positive_example_folders = [insta_base + i for i in instap_folders]  
positive_example_folders += [flickr_base + i for i in flickrp_folders]



negative_example_folders = [flickr_base + i for i in flickrn_folders]


print(positive_example_folders)

all_positives = np.array([])

for f in positive_example_folders:
    ls = np.array(os.listdir(f)[:])
    for i in range(ls.shape[0]):
        if i < ls.shape[0] and ls[i][-3:] == 'txt':
            ls = np.delete(ls, i)
            

    all_positives = np.append(all_positives, [f + i for i in ls])

all_negatives = np.array([])

for f in negative_example_folders:
    ls = np.array(os.listdir(f)[:])
    for i in range(ls.shape[0]):
        if i < ls.shape[0] and ls[i][-3:] == 'txt':
            ls = np.delete(ls ,i)
    all_negatives = np.append(all_negatives, [f + i for i in ls])


val_set_size = len(all_positives) / 10

if VALIDATE:
    val_set = np.concatenate([all_positives[-1 * val_set_size:] , all_negatives[-1* val_set_size:]])
    val_set_labels = np.concatenate( [ [ [0] for i in range(val_set_size)]  , [ [1] for i in range(val_set_size)] ])

    all_positives = all_positives[:-1* val_set_size]
    all_negatives = all_negatives[:-1* val_set_size]


def load_val_batch(start, finish):


    imgs = np.array([utils.load_image(i) for i in val_set[start:finish]])
    labels = np.array([i for i in val_set_labels[start:finish]])
    return imgs, labels



def load_batch(batch_size = 256):
    img_paths = np.concatenate( [np.random.choice(all_positives, batch_size/2, replace = False) , np.random.choice(all_negatives, batch_size/2, replace = False) ])
    imgs = [utils.load_image(i) for i in img_paths]
    labels = np.concatenate( [ [ [0] for i in range(batch_size/2)]  , [ [1] for i in range(batch_size/2)] ])
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


    step = 1
    # Keep training until reach max iterations
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

                    imgs, labels = load_val_batch(start, finish)

                    # imgs = val_set[start:finish]
                    # labels = val_set_labels[start:finish]

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

print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={images: val_set,
                                      true_out : val_set_labels,
                                      train_mode : False}))