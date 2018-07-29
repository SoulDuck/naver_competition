import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , cohen_kappa_score

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


def plot_scatter (values, cls , n_classes , savepath):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, n_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    if not savepath == None:
        plt.savefig(savepath)
    plt.show()


def get_confmat( pred_cls , cls):
    cm = confusion_matrix(cls,pred_cls)
    return cm


def get_spec_sens( pred_cls , labels , cutoff):
    pred_cls=np.asarray(pred_cls)

    indices = pred_cls > cutoff
    rev_indices = pred_cls < cutoff

    pred_cls[indices]=1
    pred_cls[rev_indices] = 0

    cm=get_confmat(pred_cls , labels )
    sensitivity = cm[0, 0] / float(cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / float(cm[1, 0] + cm[1, 1])
    print'Sensitivity : {}'.format(sensitivity)
    print'Specificity : {}'.format(specificity)
    return sensitivity , specificity

def balanced_accuracy(pred_cls , labels):
    cm = get_confmat(pred_cls, labels)
    sensitivity = cm[0, 0] / float(cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / float(cm[1, 0] + cm[1, 1])

    b_acc  =(sensitivity + specificity )/ 2
    return b_acc




def plotROC(predStrength, labels , prefix , savepath):
    debug_flag = False
    assert np.ndim(predStrength) == np.ndim(labels)
    if np.ndim(predStrength) == 2:
        predStrength = np.argmax(predStrength, axis=1)
        labels = np.argmax(labels, axis=1)

    # how to input?

    cursor = (1.0, 1.0)  # initial cursor
    ySum = 0.0  # for AUC curve
    n_pos = np.sum(np.array(labels) == 1)
    n_neg = len(labels) - n_pos
    print n_pos
    print n_neg
    y_step = 1 / float(n_pos)
    x_step = 1 / float(n_neg)
    n_est_pos = 0
    sortedIndices = np.argsort(predStrength, axis=0)
    fig = plt.figure(figsize=(10,10))
    fig.clf()
    ax = plt.subplot(1, 1, 1)
    if __debug__ == debug_flag:
        print 'labels', labels[:10]
        print 'predStrength', predStrength.T[:10]
        print 'sortedIndices', sortedIndices.T[:10]
        print  sortedIndices.tolist()[:10]
    for ind in sortedIndices.tolist():
        print ind
        if labels[ind] == 1.0:
            DelX = 0;
            DelY = y_step
        else:
            DelX = x_step;
            DelY = 0
            ySum += cursor[1]
        ax.plot([cursor[0], cursor[0] - DelX], [cursor[1], cursor[1] - DelY])
        cursor = (cursor[0] - DelX, cursor[1] - DelY)
        if __debug__ == debug_flag:
            print 'label', labels[ind]
            print 'delX',
            print 'sortedIndices', sortedIndices.T
            print 'DelX:', DelX, 'DelY:', DelY
            print 'cursor[0]-DelX :', cursor[0], 'cursor[1]-DelY :', cursor[1]
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for {}'.format(prefix))

    ax.axis([0, 1, 0, 1])
    if __debug__ == debug_flag:
        print '# of True :', n_pos
        print '# of False :', n_neg
    plt.savefig(savepath)
    # plt.show()
    print 'The Area Under Curve is :', ySum * x_step
    return ySum * x_step




if __name__ == '__main__':
    plot_scatter(np.asarray([[1.0,1.0] , [1.0,2.0] ,[1.0,3.0]])   , np.asarray([0,1,1]) ,2  ,None )
    # Kohen Kappa Score
    preds = [0 ,1 , 1]
    trues = [0,0,1]
    print cohen_kappa_score(preds , trues )