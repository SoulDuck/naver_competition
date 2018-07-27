from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from DataProvider import DataProvider
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import mglearn
import numpy as np

X,y=make_moons(n_samples=100 , noise=0.25 , random_state=3)
X_train , X_test , y_train , y_test=train_test_split(X,y,stratify=y, random_state=42)
print np.shape(X_test)
print np.shape(X_train)


dataprovider = DataProvider('wbc_10fold.txt')
dataprovider.read_wbc_10_fold('\t', 1)
train_datum= dataprovider.train
train_labels =dataprovider.train_label
train_labels =np.reshape(train_labels , [-1])
train_labels =np.reshape(train_labels , [-1])
print np.shape(y_train)
print type(y_train)
print np.shape(train_labels)
print type(train_labels)

train_datum=np.asarray(train_datum)
print np.shape(train_labels)
mlp=MLPClassifier(solver='lbfgs' , random_state=0 ,hidden_layer_sizes=[10,10])
mlp.fit(train_datum  , train_labels)

mglearn.discrete_scatter(train_datum[:,0] , train_datum[:,1] , train_labels)
plt.show()

datum, labels = dataprovider.get_all_train('\t')
