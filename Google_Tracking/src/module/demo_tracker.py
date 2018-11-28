import numpy as np
import tensorflow as tf
import cv2
import time
import os
import sys
import matplotlib.pyplot as plt

import src.net_core.colornet_3D as colornet
import dataset_utils.datasetUtils as datasetUtils
########################################################################################################################
class Google_Tracker(object):
    def __init__(self,
                 name='tracker',
                 imgSize=(224, 224),
                 batchSize=32,
                 learningRate=0.0001,
                 coreActivation=tf.nn.relu,
                 lastActivation=tf.nn.relu,
                 consecutiveFrame=4,
                 centers=np.load('/ssdubuntu/color/center_256_MiniBatchKmeans.npy'),
                 weight_flag=True
                 ):
        self._name = name
        self._imgSize = imgSize
        self._batchSize = batchSize
        self._lr = learningRate
        self._coreAct = coreActivation
        self._lastAct = lastActivation
        self._consecutiveFrame = consecutiveFrame
        self._centers = centers
        self._binNum = 16
        self._img_reduce_factor = 8
        self.variables = None
        self.update_ops = None
        self._inputImgs = None
        self._optimizer = None
        self._loss = None
        self._dist = np.load('/ssdubuntu/color/distribution.npy')
        self._weight_flag = weight_flag

        self._refineWeight()
        self._buildNetwork()

        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        self._sess = tf.Session()
        self._sess.run(init)

    def _refineWeight(self):
        prob_lambda = 0.5
        uni_dist = np.ones_like(self._dist, dtype=np.float32)
        uni_dist = uni_dist / np.sum(uni_dist)
        final_prob = prob_lambda * self._dist + (1.0 - prob_lambda) * uni_dist
        self._weight = 1.0 / final_prob
        self._weight = self._weight / np.sum(self._weight)

    def _buildNetwork(self):
        print 'build Network...'
        self._inputImgs = tf.placeholder(tf.float32, shape=[None,
                                                            self._consecutiveFrame,
                                                            self._imgSize[0],
                                                            self._imgSize[1],
                                                            1])

        self._colorGT = tf.placeholder(tf.float32, shape=[None,
                                                          self._consecutiveFrame,
                                                          self._imgSize[0] / self._img_reduce_factor,
                                                          self._imgSize[1] / self._img_reduce_factor,
                                                          self._binNum])

        self._colorizor = colornet.CN_Colorize(name=self._name + '_Tracker',
                                               frameNum=self._consecutiveFrame,
                                               trainable=True,
                                               bnPhase=False,
                                               coreActivation=tf.nn.relu,
                                               lastLayerActivation=tf.nn.relu,
                                               lastLayerPooling=None)

        self._NetworkOutput = self._colorizor(self._inputImgs)
        # print self._NetworkOutput.get_shape()
        print 'build Done!'

    def fit(self, batchDict):
        feed_dict = {
            self._inputImgs: batchDict['InputImages']
        }

        net_out = \
            self._sess.run(self._NetworkOutput,
                           feed_dict=feed_dict)

        return net_out

    def restoreTrackerCore(self, restorePath='./'):
        CorePath = os.path.join(restorePath, self._name + '_trackerCore.ckpt')
        self._colorizor.coreSaver.restore(self._sess, CorePath)

    def restoreTrackerLastLayer(self, restorePath='./'):
        LastPath = os.path.join(restorePath, self._name + '_trackerLastLayer.ckpt')
        self._colorizor.colorizorSaver.restore(self._sess, LastPath)

    def restoreNetworks(self, restorePath='./'):
        self.restoreTrackerCore(restorePath)
        self.restoreTrackerLastLayer(restorePath)

# sample = Google_Tracker()