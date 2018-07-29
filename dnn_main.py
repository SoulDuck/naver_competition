from dnn import affine , logits , algorithm
import tensorflow as tf
from DataProvider import DataProvider
from utils import next_batch , normalize ,cls2onehot
import csv
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import plot_scatter

# Data Provider
dataprovider = DataProvider('./wbc_10fold.txt')
print ' ##### 10 Fold Training Start ######'
result = open('./result.csv','w')
writer=csv.writer(result)
save_root_dir = './save_model'

for i in range(1,11):
    max_acc = 0
    train_acc = 0
    train_cost = 10000
    # INIT MODEL
    n_classes = 2
    x_ = tf.placeholder(tf.float32, [None, 9], 'x_')
    y_ = tf.placeholder(tf.int32, [None, n_classes], 'y_')
    lr_ = tf.placeholder(tf.float32, name='lr_')
    out_chs = [10, 20]
    layer = x_
    for ind, out_ch in enumerate(out_chs):
        layer = affine('fc{}'.format(ind), layer, out_ch)
    top_layer = tf.identity(layer , 'top_layer')
    logits_ = logits('logits', layer, n_classes)
    pred, pred_cls, cost_op, train_op, correct_pred, accuracy_op = \
        algorithm(logits_, y_, learning_rate=lr_, optimizer='sgd', use_l2_loss=None)

    saver = tf.train.Saver()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    max_iter = 30000

    print ' ##### {} ######'.format(i)
    dataprovider.read_wbc_10_fold('\t' , i ) #

    # Log Writer Setting
    logdir=os.path.join('logs' , str(i))
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    log_writer = tf.summary.FileWriter(logdir)
    log_writer.add_graph(tf.get_default_graph())

    # Save Model Setting
    save_dir = os.path.join(save_root_dir, str(i))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    train_data , val_data , test_data = np.asarray(dataprovider.train)/10. , np.asarray(dataprovider.val)/10. , np.asarray(dataprovider.test)/10.
    train_lab, val_lab, test_lab=map(lambda labels : cls2onehot(labels , 2) ,\
                                     [dataprovider.train_label , dataprovider.val_label, dataprovider.test_label] )

    for step in range(max_iter):
        # Validation
        if step % 100 ==0:
            # Validation
            feed_dict={x_ : val_data  , y_: val_lab }
            val_acc , val_cost = sess.run([accuracy_op ,cost_op ] , feed_dict)
            # Write validation log
            prefix = "Validation"
            summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(val_cost)),
                                        tf.Summary.Value(tag='accuracy_{}'.format(prefix),
                                                         simple_value=float(val_acc))])
            log_writer.add_summary(summary, step)
            # Model Save
            save_path=os.path.join(save_dir , 'model')
            if val_acc > max_acc:
                max_acc = val_acc
                print 'Max ACC : {}'.format(max_acc)
                saver.save(sess , save_path ,step )
                prefix = 'Validation'
                # Test
                feed_dict = {x_: test_data, y_: test_lab}
                test_acc, test_cost = sess.run([accuracy_op, cost_op], feed_dict)
                print 'Test ACC : ', test_acc
                print 'Test LOSS :', test_cost
                writer.writerow([i, train_acc, train_cost, val_acc, val_cost, test_acc, test_cost])
                test_acc, test_cost = sess.run([accuracy_op, cost_op], feed_dict)
                print 'PCA'
                feed_dict = {x_: np.vstack([train_data, val_data]) , y_: np.vstack([train_lab, val_lab])}
                top_layer_values = np.squeeze(sess.run([top_layer ] , feed_dict))
                pca = PCA(n_components=2)
                reduced_top_layer_values = pca.fit_transform(top_layer_values)
                print 'TSNE'
                cls=np.argmax(np.vstack([train_lab, val_lab]) ,axis =1 )
                tsne = TSNE(n_components=2)
                transfer_values_reduced = tsne.fit_transform(reduced_top_layer_values )
                tsne_savepath = os.path.join('./logs' , str(i) , '{}.png'.format(step))
                plot_scatter(transfer_values_reduced, cls , 2 , savepath= tsne_savepath)


        # Training
        # Get random Batch
        batch_xs , batch_ys = next_batch(train_data, train_lab, 30)
        feed_dict = {x_: batch_xs, y_: batch_ys, lr_: 0.01}
        _, train_cost, train_acc = sess.run([train_op, cost_op, accuracy_op], feed_dict)
        prefix = 'Train'
        summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(val_cost)),
                                    tf.Summary.Value(tag='accuracy_{}'.format(prefix),
                                                     simple_value=float(val_acc))])
        log_writer.add_summary(summary, step)

    print 'RESET GRAPH'
    tf.reset_default_graph()
result.close()







