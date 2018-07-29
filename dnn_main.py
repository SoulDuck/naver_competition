from dnn import affine , logits , algorithm
import tensorflow as tf
from DataProvider import DataProvider
from utils import next_batch , normalize ,cls2onehot
import csv
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import cohen_kappa_score
from utils import plot_scatter , get_spec_sens , plotROC , balanced_accuracy


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
    train_sens, train_spec, train_cohen, train_b_acc, train_auc=0,0,0,0,0
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
    pred_op , pred_cls, cost_op, train_op, correct_pred, accuracy_op = \
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
            val_acc , val_cost , val_preds  = sess.run([accuracy_op ,cost_op , pred_op ] , feed_dict)
            # Write validation log
            val_sens , val_spec = get_spec_sens(val_preds[:,1] , np.argmax(val_lab , axis =1) ,  0.5)
            val_cohen = cohen_kappa_score(y1=np.argmax(val_preds, axis=1), y2=np.argmax(val_lab, axis=1))
            val_b_acc = balanced_accuracy(val_preds[:, 1], np.argmax(val_lab, axis=1))
            val_auc = plotROC(predStrength=val_preds[:, 1], labels=np.argmax(val_lab, axis=1), prefix='Validation ROC Curve',
                    savepath='./logs/{}/validation_plot_{}.png'.format(i, step))


            prefix = "Validation"
            summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(val_cost)),
                                        tf.Summary.Value(tag='accuracy_{}'.format(prefix), simple_value=float(val_acc)),
                                        tf.Summary.Value(tag='sensitivity_{}'.format(prefix), simple_value=float(val_sens)),
                                        tf.Summary.Value(tag='specifity_{}'.format(prefix), simple_value=float(val_spec)),
                                        tf.Summary.Value(tag='cohen_{}'.format(prefix),simple_value=float(val_cohen)),
                                        tf.Summary.Value(tag='balance_accuracy_{}'.format(prefix),simple_value=float(val_b_acc)),
                                        tf.Summary.Value(tag='balance_accuracy_{}'.format(prefix),simple_value=float(val_auc)),
                                        ])

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
                test_acc, test_cost , test_preds = sess.run([accuracy_op, cost_op , pred_op], feed_dict)
                test_sens, test_spec = get_spec_sens(test_preds[:, 1], np.argmax(test_lab, axis=1), 0.5)
                test_cohen = cohen_kappa_score(y1=np.argmax(test_preds, axis=1), y2=np.argmax(test_lab, axis=1))
                test_b_acc = balanced_accuracy(test_preds[:, 1] ,np.argmax(test_lab, axis=1) )
                test_auc = plotROC(predStrength=test_preds[:, 1], labels=np.argmax(test_lab, axis=1), prefix='Test ROC Curve',
                        savepath='./logs/{}/test_plot_{}.png'.format(i,step))

                print 'Test ACC : ', test_acc
                print 'Test LOSS :', test_cost
                print 'Test Sensitiivty : ', test_sens
                print 'Test Specifity :', test_spec
                print 'Test cohen :', test_cohen
                print 'Test balanced accuracy :' ,test_b_acc
                print 'Test AUC :', test_auc

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


                writer.writerow([i, train_acc, train_cost, train_sens , train_spec , train_cohen , train_b_acc , train_auc ,
                                 val_acc, val_cost, val_sens , val_spec , val_cohen , val_b_acc , val_auc,
                                 test_acc ,test_cost , test_sens , test_spec , test_cohen , test_b_acc , test_auc])

        # Training
        # Get random Batch
        batch_xs , batch_ys = next_batch(train_data, train_lab, 30)
        feed_dict = {x_: batch_xs, y_: batch_ys, lr_: 0.01}
        _, train_cost, train_acc , train_preds = sess.run([train_op, cost_op, accuracy_op, pred_op], feed_dict)

        prefix = 'Train'
        summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(train_cost)),
                                    tf.Summary.Value(tag='accuracy_{}'.format(prefix), simple_value=float(train_acc)),
                                    ])
        log_writer.add_summary(summary, step)
    print 'RESET GRAPH'
    tf.reset_default_graph()
result.close()


"""
feed_dict={x_ : val_data  , y_: val_lab }
val_acc , val_cost , val_preds  = sess.run([accuracy_op ,cost_op , pred_op ] , feed_dict)
# Write validation log
val_sens , val_spec = get_spec_sens(val_preds[:,1] , np.argmax(val_lab , axis =1) ,  0.5)
val_cohen = cohen_kappa_score(y1=np.argmax(val_preds, axis=1), y2=np.argmax(val_lab, axis=1))
val_b_acc = balanced_accuracy(val_preds[:, 1], np.argmax(val_lab, axis=1))
val_auc = plotROC(predStrength=val_preds[:, 1], labels=np.argmax(val_lab, axis=1), prefix='Validation ROC Curve',
    savepath='./logs/{}/validation_plot_{}.png'.format(i, step))


prefix = "Validation"
summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(val_cost)),
                        tf.Summary.Value(tag='accuracy_{}'.format(prefix), simple_value=float(val_acc)),
                        tf.Summary.Value(tag='sensitivity_{}'.format(prefix), simple_value=float(val_sens)),
                        tf.Summary.Value(tag='specifity_{}'.format(prefix), simple_value=float(val_spec)),
                        tf.Summary.Value(tag='cohen_{}'.format(prefix),simple_value=float(val_cohen)),
                        tf.Summary.Value(tag='balance_accuracy_{}'.format(prefix),simple_value=float(val_b_acc)),
                        tf.Summary.Value(tag='balance_accuracy_{}'.format(prefix),simple_value=float(val_auc)),
                        ])



"""







