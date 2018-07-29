import numpy as np
import matplotlib.pyplot as plt

from DataProvider import DataProvider
dataprovider = DataProvider('./wbc_10fold.txt')
datum , labels =dataprovider.get_all_train('\t')


class EDA(object):
    def __init__(self):

        plt.scatter(x ,y )






mal_indices=np.where([labels ==1])[1]
# Benign
bgn_indices=np.where([labels ==0])[1]

plt.boxplot(datum[be_indices])
plt.ylabel('scale')
plt.xlabel('feature')
plt.show()


plt.boxplot(datum[mal_indices])
plt.ylabel('scale')
plt.xlabel('feature')
plt.show()