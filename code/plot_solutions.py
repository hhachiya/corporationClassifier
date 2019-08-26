import os
import numpy as np
import pickle
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

if __name__ == "__main__":

    data = []
    dataN = 3
    dataType = 2
    data_name = ["train loss","train confusion matrix","train auc","test loss","test confusion matrix","test auc"]

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
        if data_name[i] == "train loss" or data_name[i] == "test loss":
            plt.ylim([80,0])
        else:
            plt.ylim([1,0])
        plt.xlabel("iteration")
        plt.ylabel(data_name[i])
        plt.savefig("../data/out/{0}.png".format(data_name[i]))        


