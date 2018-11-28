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
        self._createLoss()
        self._setOptimizer()
        self._createEvaluation()

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
                                               bnPhase=True,
                                               coreActivation=tf.nn.relu,
                                               lastLayerActivation=tf.nn.relu,
                                               lastLayerPooling=None)

        self._NetworkOutput = self._colorizor(self._inputImgs)
        # print self._NetworkOutput.get_shape()
        print 'build Done!'

    def _createLoss(self):
        print 'create loss...'
        #####################################################
        # get similarity matrix (frame 1&2, 2&3, 3&4)
        left_term = tf.cast(self._NetworkOutput[:, :-1, ...], tf.float32)
        right_term = tf.cast(tf.tile(self._NetworkOutput[:, -1:, ...], [1, self._consecutiveFrame - 1, 1, 1, 1]),
                             tf.float32)
        # print left_term.get_shape()
        # print right_term.get_shape()

        term_shape = left_term.get_shape().as_list()
        # left_term = tf.reshape(left_term, [-1] + [term_shape[1]] + [term_shape[2] * term_shape[3]] + [term_shape[-1]])
        # right_term = tf.reshape(right_term, [-1] + [term_shape[1]] + [term_shape[2] * term_shape[3]] + [term_shape[-1]])
        left_term = tf.reshape(left_term, [-1] + [term_shape[2] * term_shape[3]] + [term_shape[-1]])
        right_term = tf.reshape(right_term, [-1] + [term_shape[2] * term_shape[3]] + [term_shape[-1]])

        print left_term.get_shape()
        print right_term.get_shape()

        feature_prod = tf.matmul(left_term, tf.transpose(right_term, perm=[0, 2, 1]))
        feature_prod = tf.nn.softmax(feature_prod, 1)
        print feature_prod.get_shape()

        # left_tile = tf.tile(left_term, [1, right_term.get_shape()[1], 1])
        # right_tile = tf.tile(right_term[..., None, :], [1, 1, left_term.get_shape()[1], 1])
        # right_tile_shape = right_tile.get_shape().as_list()
        # right_tile = tf.reshape(right_tile, [-1] + left_tile.get_shape().as_list()[1:])
        #
        # print left_tile.get_shape()
        # print right_tile.get_shape()
        #
        # prod = tf.reduce_sum(left_tile * right_tile, -1)
        #
        # # exponential
        # prod = tf.exp(prod)
        #
        # # print prod.get_shape()
        # prod_shape = prod.get_shape().as_list()
        # prod = tf.reshape(prod, [-1] + [int(np.sqrt(prod_shape[-1]))] * 2)
        # prod = prod / tf.reduce_sum(prod, -1, keep_dims=True)
        # prod_shape = prod.get_shape().as_list()
        # print prod.get_shape()

        #
        # # tile prod
        # prod_tile = tf.tile(prod[..., None, :], [1, 1, self._binNum, 1])
        # print prod_tile.get_shape()

        print '*****************************************************'
        ref_colorGT = self._colorGT[:, :3, ...]
        tar_colorGT = tf.tile(self._colorGT[:, -1:, ...], [1, self._consecutiveFrame - 1, 1, 1, 1])

        ref_colorGT_reshape = tf.reshape(ref_colorGT, [-1] + [ref_colorGT.get_shape()[-3] * ref_colorGT.get_shape()[-2]]
                                         + [ref_colorGT.get_shape()[-1]])
        tar_colorGT_reshape = tf.reshape(tar_colorGT, [-1] + [tar_colorGT.get_shape()[-3] * tar_colorGT.get_shape()[-2]]
                                         + [tar_colorGT.get_shape()[-1]])
        print ref_colorGT_reshape.get_shape()
        print tar_colorGT_reshape.get_shape()

        pred_color = tf.matmul(tf.transpose(feature_prod, perm=[0, 2, 1]), ref_colorGT_reshape)
        pred_color = tf.nn.softmax(pred_color, -1)
        print pred_color.get_shape()

        self._colorPred = tf.reshape(pred_color, [-1] + [self._consecutiveFrame - 1]
                                     + self._colorGT.get_shape().as_list()[2:])
        print self._colorPred.get_shape()
        max_cls = tf.argmax(self._colorPred, -1)
        print max_cls.get_shape()

        self._loss_sum_over_q = -1.0 * tf.reduce_sum(tar_colorGT * tf.log(self._colorPred + 1e-9), -1)
        # # self._loss_sum_over_q = tf.reshape(self._loss_sum_over_q, [-1] + [self._consecutiveFrame - 1]
        # #                                    + self._loss_sum_over_q.get_shape().as_list()[1:])
        print self._loss_sum_over_q.get_shape()
        mask = tf.cast(tf.one_hot(max_cls, depth=self._binNum), tf.float32)
        # # print mask.get_shape()
        weight_mat = tf.convert_to_tensor(self._weight, tf.float32)
        weight_mat = tf.tile(weight_mat[None, None, None, None, ...], [tf.shape(mask)[0],
                                                                       self._consecutiveFrame - 1,
                                                                       self._imgSize[0] / self._img_reduce_factor,
                                                                       self._imgSize[1] / self._img_reduce_factor,
                                                                       1])
        # print weight_mat.get_shape()
        mask = tf.reduce_sum(mask * weight_mat, -1)
        # print mask.get_shape()
        if self._weight_flag:
            self._loss = tf.reduce_mean(
                tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(mask * self._loss_sum_over_q, -1), -1), -1), -1)
        else:
            self._loss = tf.reduce_mean(
                tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(self._loss_sum_over_q, -1), -1), -1), -1)
        print self._loss.get_shape()

        print 'loss done!'

    def _setOptimizer(self):
        print 'set optimizer...'
        self._lr = tf.placeholder(tf.float32, shape=[])

        with tf.control_dependencies(self._colorizor.allUpdate_ops):
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(
                self._loss,
                var_list=self._colorizor.allVariables
                # var_list=self._colorizor.coreVariables
            )
        print 'optimizer done!'


    def _createEvaluation(self):
        print 'create evaluation...'
        # eval_mask = tf.cast(tf.equal(tf.argmax(self._colorGT[:, -1, ...], -1),
        #                              tf.argmax(self._colorPred[:, -1, ...], -1)), tf.float32)
        # eval_mask_acc = tf.reduce_sum(tf.reduce_sum(eval_mask, -1), -1) / (self._imgSize[0] * self._imgSize[1] / 16.0)
        # self._avg_acc = tf.reduce_mean(eval_mask_acc, -1)

        pred_flat = tf.reshape(self._colorPred[:, -1, ...], [-1, self._binNum])
        GT_flat = tf.reshape(self._colorGT[:, -1, ...], [-1, self._binNum])

        pred_ind = tf.argmax(pred_flat, -1)
        gt_ind = tf.argmax(GT_flat, -1)
        equality = tf.equal(pred_ind, gt_ind)
        self._acc = tf.reduce_mean(tf.cast(equality, tf.float32))
        self._top5Acc = tf.reduce_mean(
            tf.cast(
                tf.nn.in_top_k(
                    predictions=pred_flat,
                    targets=gt_ind,
                    k=5),
                tf.float32))

        print 'eval done!'

    def fit(self, batchDict):
        feed_dict = {
            self._inputImgs: batchDict['InputImages'],
            self._colorGT: batchDict['OutputImages'],
            self._lr: batchDict['LearningRate']
        }

        opt, loss, acc, top5_acc, pred_color, gt_color = \
            self._sess.run([self._optimizer,
                            self._loss,
                            self._acc,
                            self._top5Acc,
                            self._colorPred,
                            self._colorGT],
                           feed_dict=feed_dict)

        print ("loss is {:f}".format(loss))
        print ("acc is {:f}%".format(acc * 100))
        print ("top5 acc is {:f}%".format(top5_acc * 100))
        # print (np.sum(gt_color, -1))
        # print (np.sum(pred_color, -1))

        return loss, acc, top5_acc

    def saveTrackerCore(self, savePath='./'):
        CorePath = os.path.join(savePath, self._name + '_trackerCore.ckpt')
        self._colorizor.coreSaver.save(self._sess, CorePath)

    def saveTrackerLastLayer(self, savePath='./'):
        LastPath = os.path.join(savePath, self._name + '_trackerLastLayer.ckpt')
        self._colorizor.colorizorSaver.save(self._sess, LastPath)

    def saveNetworks(self, savePath='./'):
        self.saveTrackerCore(savePath)
        self.saveTrackerLastLayer(savePath)

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