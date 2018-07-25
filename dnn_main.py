from dnn import affine , logits , algorithm
import tensorflow as tf
from DataProvider import DataProvider
from utils import next_batch
import numpy as np
n_classes =2
x_ = tf.placeholder(tf.float32 , [None , 9 ] , 'x_')
y_ = tf.placeholder(tf.int32 , [None , n_classes] , 'y_')



lr_ = tf.placeholder(tf.float32  , name='lr_')
out_chs=[48,120]
layer = x_
for i,out_ch in enumerate(out_chs):
    layer=affine('fc{}'.format(i) , layer , out_ch)
logits_=logits('logits' , layer , n_classes )
pred,pred_cls , cost_op, train_op,correct_pred ,accuracy_op = \
    algorithm(logits_ , y_ , learning_rate=lr_ , optimizer='adam'  , use_l2_loss=None)


init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess=tf.Session()
sess.run(init)
max_iter = 100000

# Data Provider
dataprovider = DataProvider()
print np.shape(dataprovider.wbc_y_test )
for step in range(max_iter):
    # Validation
    if step % 100 ==0:
        feed_dict={x_ : dataprovider.wbc_x_test  , y_: dataprovider.wbc_y_test }
        val_acc , val_cost = sess.run([accuracy_op ,cost_op ] , feed_dict)
        print val_acc , val_cost

    # Batch
    batch_xs , batch_ys = next_batch(dataprovider.wbc_x_train, dataprovider.wbc_y_train, 30)


    # Training
    feed_dict = {x_: batch_xs, y_: batch_ys, lr_: 0.001}
    _, train_cost=sess.run([train_op , cost_op], feed_dict)








