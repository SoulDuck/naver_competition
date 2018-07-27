from dnn import affine , logits , algorithm
import tensorflow as tf
from DataProvider import DataProvider
from utils import next_batch , normalize ,cls2onehot
import csv
import numpy as np
n_classes =2
x_ = tf.placeholder(tf.float32 , [None , 9 ] , 'x_')
y_ = tf.placeholder(tf.int32 , [None , n_classes] , 'y_')



lr_ = tf.placeholder(tf.float32  , name='lr_')
out_chs=[64,128]
layer = x_
for i,out_ch in enumerate(out_chs):
    layer=affine('fc{}'.format(i) , layer , out_ch)
logits_=logits('logits' , layer , n_classes )
pred,pred_cls , cost_op, train_op,correct_pred ,accuracy_op = \
    algorithm(logits_ , y_ , learning_rate=lr_ , optimizer='adam'  , use_l2_loss=None)


init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess=tf.Session()
sess.run(init)
max_iter = 10000

# Data Provider
dataprovider = DataProvider('./wbc_10fold.txt')



print ' ##### 10 Fold Training Start ######'
result = open('./result.csv','w')
writer=csv.writer(result)



for i in range(1,10+1):
    print ' ##### {} ######'.format(i)
    dataprovider.read_wbc_10_fold('\t' , i ) #
    train_data , val_data , test_data =normalize(dataprovider.train , dataprovider.val , dataprovider.test)
    train_lab, val_lab, test_lab=map(lambda labels : cls2onehot(labels , 2) ,\
                                     [dataprovider.train_label , dataprovider.val_label, dataprovider.test_label] )
    for step in range(max_iter):
        # Validation
        if step % 100 ==0:
            feed_dict={x_ : val_data  , y_: val_lab }
            val_acc , val_cost = sess.run([accuracy_op ,cost_op ] , feed_dict)
            #print val_acc , val_cost
        # Batch
        batch_xs , batch_ys = next_batch(train_data, train_lab, 30)
        # Training

        feed_dict = {x_: batch_xs, y_: batch_ys, lr_: 0.0005}
        _, train_cost, train_acc = sess.run([train_op, cost_op, accuracy_op], feed_dict)
    feed_dict = {x_: test_data, y_: test_lab}
    test_acc , test_cost = sess.run([accuracy_op ,cost_op ] , feed_dict)
    print 'Train ACC : ', train_acc
    print 'Train LOSS :', train_cost
    print 'Val ACC : ', val_acc
    print 'Val LOSS :', val_cost
    print 'Test ACC : ',test_acc
    print 'Test LOSS :', test_cost
    writer.writerow([train_acc, train_cost , val_acc ,val_cost , test_acc, test_cost ])

result.close()







