import numpy as np
import random
def next_batch(x_data , y_data , batch_size):
    indices=random.sample(range(len(x_data))  , batch_size)
    return x_data[indices] , y_data[indices]

def cls2onehot(cls, depth):
    debug_flag=False
    if not type(cls).__module__ == np.__name__:
        cls=np.asarray(cls)
    cls=cls.astype(np.int32)
    debug_flag = False
    labels = np.zeros([len(cls), depth] , dtype=np.int32)
    for i, ind in enumerate(cls):
        labels[i][ind:ind + 1] = 1
    if __debug__ == debug_flag:
        print '#### data.py | cls2onehot() ####'
        print 'show sample cls and converted labels'
        print cls[:10]
        print labels[:10]
        print cls[-10:]
        print labels[-10:]
    return labels

def normalize(train_data , val_data ,test_data):
    train_data, val_data, test_data=map(np.asarray ,[train_data ,val_data , test_data])
    train_h, train_w = np.shape(train_data)
    val_h, val_w = np.shape(val_data)
    test_h, test_w = np.shape(test_data)

    ret_train_mat = np.zeros([train_h, train_w])
    ret_val_mat = np.zeros([val_h, val_w])
    ret_test_mat = np.zeros([test_h, test_w])

    for i in range(train_w):

        tmp_max = np.max(train_data[:, i])
        tmp_min = np.min(train_data[:, i])

        normal_train = (train_data[:,i] - tmp_max) /(tmp_max - tmp_min)
        normal_val = (val_data[:,i]- tmp_max) / (tmp_max - tmp_min)
        normal_test= (test_data[:,i] - tmp_max) / (tmp_max - tmp_min)
        ret_train_mat[:, i] = normal_train

        ret_val_mat[:, i] = normal_val
        ret_test_mat[:, i] = normal_test

    return ret_train_mat , ret_val_mat, ret_test_mat