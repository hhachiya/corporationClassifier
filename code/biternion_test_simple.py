# -*- coding: utf-8 -*-
#実写でトレーニングし、CGの特徴量を可視化。
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.examples.tutorials.mnist import input_data
# from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import math, os
import pickle
import pdb
import input_data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import cv2
import sys
from matplotlib.colors import LinearSegmentedColormap
from pylab import rcParams
import time

#===========================
# レイヤーの関数
def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

# 1D convolution layer
def conv1d_relu(inputs, w, b, stride):
	# tf.nn.conv1d(input,filter,strides,padding)
	#filter: [kernel, output_depth, input_depth]
	# padding='SAME' はゼロパティングしている
	conv = tf.nn.conv1d(inputs, w, stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 1D deconvolution
def conv1d_t_relu(inputs, w, b, output_shape, stride):
	conv = nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 2D convolution
def conv2d_relu(inputs, w, b, stride):
	# tf.nn.conv2d(input,filter,strides,padding)
	# filter: [kernel, output_depth, input_depth]
	# input 4次元([batch, in_height, in_width, in_channels])のテンソルを渡す
	# filter 畳込みでinputテンソルとの積和に使用するweightにあたる
	# stride （=１画素ずつではなく、数画素ずつフィルタの適用範囲を計算するための値)を指定
	# ただし指定は[1, stride, stride, 1]と先頭と最後は１固定とする
	conv = tf.nn.conv2d(inputs, w, strides=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

def conv2d_bn_relu(inputs, w, b, stride,pad='SAME',isTrain=True):
    conv = tf.nn.conv2d(inputs, w, strides=stride, padding=pad) + b
    conv = tf.layers.batch_normalization(conv, training=isTrain, trainable=isTrain)
    conv = tf.nn.relu(conv)
    return conv

def conv2d_bn_pool2x2_relu(inputs, w, b, stride,pad='SAME',isTrain=True):
    conv = tf.nn.conv2d(inputs, w, strides=stride, padding=pad) + b
    conv = tf.layers.batch_normalization(conv, training=isTrain, trainable=isTrain)
    conv = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding="SAME")
    conv = tf.nn.relu(conv)
    return conv

# 2D deconvolution layer
def conv2d_t_sigmoid(inputs, w, b, output_shape, stride):
    conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
    conv = tf.nn.sigmoid(conv)
    return conv

# 2D deconvolution layer
def conv2d_t_relu(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

def conv2d_t_tanh(inputs, w, b,output_shape,stride):
    conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
    conv = tf.nn.tanh(conv)
    return conv

# 2D deconvolution layer
def conv2d_t_bn_relu(inputs, w, b, output_shape, stride,isTrain=True):
    conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
    conv = tf.layers.batch_normalization(conv, training=isTrain, trainable=isTrain)
    conv = tf.nn.relu(conv)
    return conv


# 2D deconvolution layer
def conv2d_t(inputs, w, b, output_shape, stride):
    conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
    return conv

# fc layer with ReLU
def fc_relu(inputs, w, b, rate=0.0):
    fc = tf.matmul(inputs, w) + b
    fc = tf.nn.dropout(fc, rate=rate)
    fc = tf.nn.relu(fc)
    return fc

# fc layer
def fc(inputs, w, b, rate=0.0):
    fc = tf.matmul(inputs, w) + b
    fc = tf.nn.dropout(fc, rate=rate)
    return fc

# fc layer with sigmoid
def fc_sigmoid(inputs, w, b, rate=0.0):
    fc = tf.matmul(inputs, w) + b
    fc = tf.nn.dropout(fc, rate=rate)
    fc = tf.nn.sigmoid(fc)
    return fc

def flatten(inputs):
    size = np.prod(inputs.get_shape().as_list()[1:])
    vec = tf.reshape(inputs, [-1, size])
    return vec

#=========================================================================

def baseCNN(x,reuse=False,isTrain=True,rates=[0.0,0.0]):
    chan = [3,24,24,48,48,64,64]
    layerNum = len(chan)-1
    f_size = 3

    with tf.variable_scope('baseCNN') as scope:
        if reuse:
            scope.reuse_variables()

        W = [weight_variable("convW{}".format(i),[f_size,f_size,chan[i],chan[i+1]]) for i in range(layerNum)]
        B = [bias_variable("convB{}".format(i),[chan[i+1]]) for i in range(layerNum)]

        #pdb.set_trace()
        # 34 -> 32
        conv = conv2d_bn_relu(x,W[0],B[0],[1,1,1,1],pad='VALID',isTrain=isTrain)
        # 32 -> 30 -> 15
        conv = conv2d_bn_pool2x2_relu(conv,W[1],B[1],[1,1,1,1],pad='VALID',isTrain=isTrain)
        # 15 -> 13
        conv = conv2d_bn_relu(conv,W[2],B[2],[1,1,1,1],pad='VALID',isTrain=isTrain)
        # 13 -> 11 -> 6
        conv = conv2d_bn_pool2x2_relu(conv,W[3],B[3],[1,1,1,1],pad='VALID',isTrain=isTrain)
        # 6 -> 4
        conv = conv2d_bn_relu(conv,W[4],B[4],[1,1,1,1],pad='VALID',isTrain=isTrain)
        # 4 -> 2
        conv = conv2d_bn_relu(conv,W[5],B[5],[1,1,1,1],pad='VALID',isTrain=isTrain)

        fc1 = flatten(conv)
        fc1 = tf.nn.dropout(fc1,rate=rates[0])

        fcW1 = weight_variable("fcW1",[2*2*64,128])
        fcB1 = bias_variable("fcB1",[128])
        fc1 = fc_relu(fc1,fcW1,fcB1,rates[1])

        fcW2 = weight_variable("fcW2",[128,2])
        fcB2 = bias_variable("fcB2",[2])
        fc2 = fc(fc1,fcW2,fcB2)

        # 単位ベクトルに正規化
        fc2 = vec_norm(fc2)

        return fc2

#========================================

# y : numpy,shape=[N,2]
# t : numpy,shape=[N,1]
def MAE(y,t):
    degy = vec2deg(y)
    errs = np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(t-degy)),np.cos(np.deg2rad(t-degy)))))
    mean_errs = np.mean(errs)
    std_errs = np.std(errs)
    return mean_errs,std_errs

# vec : numpy,shape=[N,2]
def vec2deg(vec):
    # arctan2は[-pi,pi]の範囲で値を返す
    res = (np.rad2deg(np.arctan2(vec[:,1],vec[:,0])) + 360) % 360
    return np.reshape(res,[-1,1])

def tf_deg2rad(degs):
    return degs*(np.pi/180)

def deg2vec(degs):
    rads = tf_deg2rad(degs)
    return tf.concat([tf.cos(rads),tf.sin(rads)],axis=1)

def vec_norm(vector):
    l2norm = tf.sqrt(tf.reduce_sum(tf.square(vector),axis=1,keepdims=True))
    return vector/l2norm

# y,t :　shape=[N,2]
def CosBiternion(y,t):
    return tf.reduce_mean(1 - tf.reduce_sum(y*t,axis=1,keepdims=True))

def VMBiternion(y,t,kappa=1.0):
    res = tf.reduce_mean(1 - tf.exp(kappa*(tf.reduce_sum(y*t,axis=1) - 1)))
    return res


#shape = [w,h]
#augNum > 1 なら yにラベルを入れる
def crop(x,dataNum,out_shape,y = None,augNum=1,xshape=[50,50],isTrain=True):
    w = out_shape[0]
    h = out_shape[1]
    if isTrain:
        res = []
        for i in range(dataNum):
            res_x = tf.gather_nd(x,indices=[i])
            # ランダムクロップ（sess.runするたびに変化）
            res_x = [tf.image.random_crop(res_x,[h,w,3]) for _ in range(augNum)]

            res.extend(res_x)

        #pdb.set_trace()

        res = tf.stack(res,axis=0)

        # データ数が増加しないならresのみ、増加するなら拡張後のラベルも
        if augNum == 1:
            return res
        elif augNum > 1:
            new_label = tf.reshape(tf.tile(y,[1,augNum]),[dataNum*augNum,1])
            return res, new_label
        else:
            print("augNum is wrong")
            return

    else:
        xw = xshape[0]
        xh = xshape[1]
        normw = (w-1)/(xw-1)
        normh = (h-1)/(xh-1)
        inds = [i for i in range(dataNum)]

        # top_left (shape=[N,h,w,3])
        tl_x = tf.image.crop_and_resize(x,[[0,0,normh,normw]]*dataNum,inds,[h,w])
        # top_right
        tr_x = tf.image.crop_and_resize(x,[[0,1-normw,normh,1]]*dataNum,inds,[h,w])
        # center
        ce_x = tf.image.crop_and_resize(x,[[(1-normh)/2,(1-normw)/2,(1+normh)/2,(1+normw)/2]]*dataNum,inds,[h,w])
        # bottom_left
        bl_x = tf.image.crop_and_resize(x,[[1-normh,0,1,normw]]*dataNum,inds,[h,w])
        # bottom_right
        br_x = tf.image.crop_and_resize(x,[[1-normh,1-normw,1,1]]*dataNum,inds,[h,w])

        return [tl_x,tr_x,ce_x,bl_x,br_x]

def ensemble_biternion(vecs):
    #pdb.set_trace()
    crop_num = len(vecs)
    sum_v = vecs[0]
    for i in range(crop_num-1):
        sum_v = sum_v + vecs[i+1]

    return sum_v/crop_num

def show(los,me,se,isWhat):
    if isWhat=="train":
        print("#{}(epoch{}))".format(ite,epo))
        print("< train >")
    elif isWhat=="test":
        print("<<< test >>>")
    elif isWhat=="valid":
        print("< validate >")
    print("  lossReg={}, MAE={}±{}".format(los,me,se))
    return

def next_batch(num,data,labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]

    return data_shuffle, labels_shuffle


if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=config)

    #Epoch数
    nEpo = 50
    # plotする画像
    plotNum = 25
    plotWid = 5
    augNum = 3
    # バッチデータ数
    batchSize = 300
    croped_batchSize = batchSize*augNum

    #======================================
    # データ読み込み
    realData = pickle.load(open("data/realData4reg.pickle","rb"))

    # validation
    valid_num = realData.validate.num_examples
    valid_x = realData.validate.images
    valid_label = np.reshape(realData.validate.labels,[valid_num,1])

    # test
    test_num = realData.test.num_examples
    test_x = realData.test.images
    test_label = np.reshape(realData.test.labels,[test_num,1])
    #======================================
    ## placeholder
    #pdb.set_trace()
    x_train = tf.placeholder(tf.float32,shape=[None,50,50,3])
    x_label = tf.placeholder(tf.float32,shape=[None,1])
    x_crop_train, x_aug_label = crop(x_train,batchSize,[34,34],y=x_label,augNum=augNum)
    x_vec_label = deg2vec(x_aug_label)

    x_valid = tf.placeholder(tf.float32,shape=[None,50,50,3])
    ##------ 注意:x_valid_cropsはlist --------------------
    x_valid_crops = crop(x_valid,valid_num,[34,34],isTrain=False)
    x_valid_label = tf.placeholder(tf.float32,shape=[None,1])
    x_valid_vec_label = deg2vec(x_valid_label)

    x_test = tf.placeholder(tf.float32,shape=[None,50,50,3])
    ##------ 注意:x_test_cropsはlist --------------------
    x_test_crops = crop(x_test,test_num,[34,34],isTrain=False)
    x_test_label = tf.placeholder(tf.float32,shape=[None,1])
    x_test_vec_label = deg2vec(x_test_label)
    #======================================
    #--------------------------------------
    ## build model
    train_pred = baseCNN(x_crop_train,rates=[0.2,0.5])

    valid_preds = [baseCNN(valid,reuse=True,isTrain=False) for valid in x_valid_crops]
    test_preds = [baseCNN(test,reuse=True,isTrain=False) for test in x_test_crops]

    valid_pred = ensemble_biternion(valid_preds)
    test_pred = ensemble_biternion(test_preds)
    #--------------------------------------
    ## loss function
    #pdb.set_trace()

    lossReg = VMBiternion(train_pred,x_vec_label)
    valid_lossReg = VMBiternion(valid_pred,x_valid_vec_label)
    test_lossReg = VMBiternion(test_pred,x_test_vec_label)

    #--------------------------------------
    ## trainer & vars
    #pdb.set_trace()

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="baseCNN")
    with tf.control_dependencies(extra_update_ops):
        regVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="baseCNN")
        trainerReg = tf.train.AdamOptimizer(1e-3).minimize(lossReg, var_list=regVars)
    #trainerReg = tf.train.AdadeltaOptimizer(1e-3).minimize(lossReg, var_list=regVars)

    #--------------------------------------
    #======================================
    # 保存用
    tra_loss_list = []
    tra_merr_list = []
    tra_serr_list = []
    tra_preds_list = []
    tra_label_list = []
    val_loss_list = []
    val_merr_list = []
    val_serr_list = []
    val_preds_list = []
    tes_loss_list = []
    tes_merr_list = []
    tes_serr_list = []
    tes_preds_list = []
    #======================================
    # 初期化
    sess.run(tf.global_variables_initializer())
    ite = 0
    isStop = False
    #======================================

    pdb.set_trace()

    while not isStop:
        ite = ite + 1

        #-----------------
        ## バッチの取得
        #pdb.set_trace()
        batch_x,batch_label = realData.train.next_batch(batchSize)
        batch_label = np.reshape(batch_label,[batchSize,1])
        crop_batch_label = np.tile(batch_label,[1,augNum])
        crop_batch_label = np.reshape(crop_batch_label,[croped_batchSize,1])
        epo = realData.train.epochs_completed
        #-----------------

        # training
        _,lossReg_value,pred_value = sess.run([trainerReg,lossReg,train_pred],feed_dict={x_train:batch_x, x_label:batch_label})

        merr,serr = MAE(pred_value,crop_batch_label)

        # 保存
        tra_loss_list.append(lossReg_value)
        tra_merr_list.append(merr)
        tra_serr_list.append(serr)
        tra_preds_list.append(pred_value)
        tra_label_list.append(crop_batch_label)

        # test
        test_pred_value, test_lossReg_value = sess.run([test_pred,test_lossReg],feed_dict={x_test:test_x, x_test_label:test_label})
        test_merr, test_serr = MAE(test_pred_value, test_label)
        # validation
        valid_pred_value, valid_lossReg_value = sess.run([valid_pred,valid_lossReg],feed_dict={x_valid:valid_x, x_valid_label:valid_label})
        valid_merr, valid_serr = MAE(valid_pred_value,valid_label)

        # 保存
        val_loss_list.append(valid_lossReg_value)
        val_merr_list.append(valid_merr)
        val_serr_list.append(valid_serr)
        val_preds_list.append(valid_pred_value)

        tes_loss_list.append(test_lossReg_value)
        tes_merr_list.append(test_merr)
        tes_serr_list.append(test_serr)
        tes_preds_list.append(test_pred_value)

        if (ite%100==0):
            show(lossReg_value,merr,serr,"train")
            show(valid_lossReg_value, valid_merr, valid_serr, "valid")
            show(test_lossReg_value, test_merr, test_serr,"test")

        if epo==nEpo:
            isStop = True


    with open("log/biternion_test_log.pickle","wb") as f:
        pickle.dump(tra_loss_list,f)
        pickle.dump(tra_merr_list,f)
        pickle.dump(tra_serr_list,f)
        pickle.dump(tra_preds_list,f)
        pickle.dump(tra_label_list,f)

        pickle.dump(val_loss_list,f)
        pickle.dump(val_merr_list,f)
        pickle.dump(val_serr_list,f)
        pickle.dump(val_preds_list,f)
        pickle.dump(valid_label,f)

        pickle.dump(tes_loss_list,f)
        pickle.dump(tes_merr_list,f)
        pickle.dump(tes_serr_list,f)
        pickle.dump(tes_preds_list,f)
        pickle.dump(test_label,f)
