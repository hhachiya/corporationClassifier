import os
import numpy as np
import pickle
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

if __name__ == "__main__":

    data = []
    dataN = 5
    dataType = 2
    data_name = ["train loss","train confusion matrix","train auc","train precision","test loss","test confusion matrix","test auc"]

    with open("../data/out/log/biternion_test_log.pickle","rb") as f:
        for i in range(dataN*dataType):
            data.append(pickle.load(f))

    ite = len(data[0])

    for i in range(len(data_name)):
        plt.close()
        #pdb.set_trace()
        if i == 1 or i == 4:
            continue
        plt.plot(range(ite),data[i])
        plt.xlabel("iteration")
        plt.ylabel(data_name[i])
        plt.savefig("../data/out/{0}.png".format(data_name[i]))        


