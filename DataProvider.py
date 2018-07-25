import numpy as np
from utils import cls2onehot
import random
class DataProvider(object):
    def __init__(self):
        f = open('wbc.csv', 'r')
        lines = f.readlines()
        self.wbc_data=[]
        for line in lines[2:]:
            if '?' in line:
                print line
                continue
            else:
                elements = map(lambda ele : int(ele.strip()) , line.split(',')[1:])
                self.wbc_data.append(elements)

        self.wbc_data=np.asarray(self.wbc_data)
        self.wbc_x = self.wbc_data[: ,1:-1]
        self.wbc_y = self.wbc_data[: ,-1:]

        self.n_test = 60
        indices=random.sample(range(len(self.wbc_x)) , len(self.wbc_x))
        self.test_indices = indices[:120]
        self.train_indices = indices[120:]

        self.wbc_x_train = self.wbc_x[self.train_indices]
        self.wbc_x_test = self.wbc_x[self.test_indices]

        #onehot encoding
        indices=np.where([self.wbc_y ==2])[1]
        self.wbc_y[indices]=0
        indices = np.where([self.wbc_y == 4])[1]
        self.wbc_y[indices]=1
        self.wbc_y= self.wbc_y.reshape(-1)
        self.wbc_y= cls2onehot(self.wbc_y , 2)

        self.wbc_y_train = self.wbc_y[self.train_indices]
        self.wbc_y_test = self.wbc_y[self.test_indices]


        #Normalize
        tmp_mat_train = []
        tmp_mat_test = []
        for i in range(9):
            max_ = np.max(self.wbc_x_train[:,i])
            min_ = np.min(self.wbc_x_train[:, i])
            tmp_mat_train.append((self.wbc_x_train[:, i] - min_) / float((max_ - min_)))
            tmp_mat_test.append((self.wbc_x_test[:, i] - min_) / float((max_ - min_)))



        self.wbc_x_train=np.vstack(tmp_mat_train)

        self.wbc_x_train=np.transpose(self.wbc_x_train , [1,0])

        self.wbc_x_test=np.vstack(tmp_mat_test)
        self.wbc_x_test=np.transpose(self.wbc_x_test , [1,0])




        print np.shape(self.wbc_x_train)
        print np.shape(self.wbc_x_test)
        print np.shape(self.wbc_y_train)
        print np.shape(self.wbc_y_test)












if '__main__' == __name__:
    dataprovider = DataProvider()



