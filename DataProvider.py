import numpy as np
from utils import cls2onehot
import random
class DataProvider(object):
    def __init__(self ,f_path):
        self.f_path = f_path

    def read_wbc_10_fold(self , delimeter , fold_num):
        self.f = open(self.f_path)
        print fold_num
        lines = self.f.readlines()
        # ['SID', 'C_thick', 'C_size', 'C_shape', 'MA', 'SEC_size', 'BN', 'BC', 'NN', 'Mit', 'Class', 'group\n']
        self.val=[]
        self.val_label=[]
        self.train=[]
        self.train_label=[]
        self.test=[]
        self.test_label = []
        self.question=[]
        for line in lines[1:]:
            elements = line.split(delimeter)[1:]
            if '?' in elements:
                continue
            data=map(int ,elements[1:-2])
            label=int(elements[-2])/2-1

            if 'train{}'.format(fold_num) in elements[-1]:
                self.val.append(data)
                self.val_label.append(label)
            elif 'test' in elements[-1]:
                self.test.append(data)
                self.test_label.append(label)
            elif '?' in elements[-1]:
                self.question.append((elements))
            else:
                self.train.append(data )
                self.train_label.append(label)

        print '# Train {}'.format(np.shape(self.train))
        print '# Train Label {}'.format(len(self.train_label))
        print '# Val {}'.format(np.shape(self.val))
        print '# Val Label {}'.format(len(self.val_label))
        print '# Test {}'.format(np.shape(self.test))
        print '# Test Label {}'.format(len(self.test_label))
        print '# Question {}'.format(len(self.question))

        self.f.close()

    def get_all_train(self , delimeter ):
        self.f = open(self.f_path)
        lines = self.f.readlines()
        datum = []
        labels = []
        for line in lines[1:]:
            elements = line.split(delimeter)[1:]
            if '?' in elements:
                continue
            data=map(int ,elements[1:-2])
            label=int(elements[-2])/2-1
            datum.append(data)
            labels.append(label)
        datum, labels=map( np.asarray ,  [datum , labels])
        return datum  , labels






if '__main__' == __name__:
    dataprovider = DataProvider('wbc_10fold.txt')
    dataprovider.read_wbc_10_fold('\t', 2 )
    datum, labels = dataprovider.get_all_train('\t')





