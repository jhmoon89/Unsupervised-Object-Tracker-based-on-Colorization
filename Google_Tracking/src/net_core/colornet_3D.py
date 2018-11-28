import tensorflow as tf
import numpy as np
import cv2
import time
import os
import sys

class ColorNet_core(object):
    def __init__(self, name='colornet_core', trainable=True, bnPhase=True, reuse=False, activation=tf.nn.elu):
        self._reuse = tf.AUTO_REUSE
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._activation = activation
        self._name = name
        self.variables = None
        self.update_ops = None
        self.saver = None

        # print('init func')

    def _conv(self, inputs, filters, kernel_size, strides=1, dilations=1, batch_norm_flag=False):
        # print inputs.get_shape()
        hidden = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
            dilation_rate=dilations, activation=None, trainable=self._trainable, use_bias=False,

        )
        if batch_norm_flag:
            hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        hidden = self._activation(hidden)
        # print hidden.get_shape()
        return hidden

    def _conv_trans(self, inputs, filters, kernel_size, strides=1, batch_norm_flag=False):
        hidden = tf.layers.conv2d_transpose(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
            activation=None, trainable=self._trainable, use_bias=False,

        )
        if batch_norm_flag:
            hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        hidden = self._activation(hidden)
        # print hidden.get_shape()
        return hidden

    def _maxpool(self, inputs, pool_size=(2, 2), strides=2, padding='same'):
        hidden = tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides, padding=padding)
        return hidden

    def __call__(self, InputImgs):
        # print self._nameScope

        InputImgs = tf.reshape(InputImgs, [-1] + InputImgs.get_shape().as_list()[2:])

        with tf.variable_scope(self._name, reuse=self._reuse):
            h11 = self._conv(inputs=InputImgs, filters=64, kernel_size=3)
            h12 = self._conv(inputs=h11, filters=64, kernel_size=3, strides=2, batch_norm_flag=True)

            h21 = self._conv(inputs=h12, filters=128, kernel_size=3)
            h22 = self._conv(inputs=h21, filters=128, kernel_size=3, strides=2, batch_norm_flag=True)

            h31 = self._conv(inputs=h22, filters=256, kernel_size=3)
            h32 = self._conv(inputs=h31, filters=256, kernel_size=3)
            h33 = self._conv(inputs=h32, filters=256, kernel_size=3, strides=2, batch_norm_flag=True)

            h41 = self._conv(inputs=h33, filters=256, kernel_size=3)
            h42 = self._conv(inputs=h41, filters=256, kernel_size=3)
            h43 = self._conv(inputs=h42, filters=256, kernel_size=3, strides=1, batch_norm_flag=True)

        # self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._name)
        self.saver = tf.train.Saver(var_list=self.variables)
        outputs = h43

        return outputs

class CN_Colorize(object):
    def __init__(self, name='cn', frameNum=4, trainable=True,
                 bnPhase=True, coreActivation=tf.nn.elu,
                 lastLayerActivation=tf.nn.relu,
                 lastLayerPooling=None):
        self._name = name
        self._frameNum = frameNum
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._reuse = tf.AUTO_REUSE
        self._coreActivation = coreActivation
        self._lastActivation = lastLayerActivation
        self._lastPool = lastLayerPooling
        self.variables = None
        self.update_ops = None
        self.saver = None
        self._CNet_core = None

        # print 'init'

    def _conv_3D(self, inputs, filters, kernel_size, dilation, strides=1, batch_norm_flag=True):
        # print inputs.get_shape()
        hidden = tf.layers.conv3d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
            dilation_rate=dilation, activation=None, trainable=self._trainable, use_bias=False,
        )
        if batch_norm_flag:
            hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        hidden = self._lastActivation(hidden)
        # print hidden.get_shape()
        return hidden

    def _conv_trans_3D(self, inputs, filters, kernel_size, strides=1, batch_norm_flag=True):
        hidden = tf.layers.conv3d_transpose(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
            activation=None, trainable=self._trainable, use_bias=False,

        )
        if batch_norm_flag:
            hidden = tf.layers.batch_normalization(hidden, training=self._bnPhase, trainable=self._trainable)
        hidden = self._lastActivation(hidden)
        # print hidden.get_shape()
        return hidden

    def _maxpool_3D(self, inputs, pool_size=(2, 2, 2), strides=2, padding='same'):
        hidden = tf.layers.max_pooling3d(inputs=inputs, pool_size=pool_size, strides=strides, padding=padding)
        return hidden

    def __call__(self, InputImgs):
        # print self._nameScope
        self._CNet_core = ColorNet_core(name=self._name+"_CNCore", trainable=self._trainable,
                                        bnPhase=self._bnPhase, reuse=self._reuse, activation=self._coreActivation)

        hidden = self._CNet_core(InputImgs)
        hidden = tf.reshape(hidden, [-1] + [self._frameNum] + hidden.get_shape().as_list()[1:])

        with tf.variable_scope(self._name+'_Detection', reuse=self._reuse):
            h1 = self._conv_3D(inputs=hidden, filters=128, kernel_size=(1, 3, 3), dilation=(1, 1, 1))
            h2 = self._conv_3D(inputs=h1, filters=128, kernel_size=(3, 1, 1), dilation=(1, 1, 1))
            h3 = self._conv_3D(inputs=h2, filters=128, kernel_size=(1, 3, 3), dilation=(1, 2, 2))
            h4 = self._conv_3D(inputs=h3, filters=128, kernel_size=(3, 1, 1), dilation=(1, 1, 1))
            h5 = self._conv_3D(inputs=h4, filters=128, kernel_size=(1, 3, 3), dilation=(1, 4, 4))
            h6 = self._conv_3D(inputs=h5, filters=128, kernel_size=(3, 1, 1), dilation=(1, 1, 1))
            h7 = self._conv_3D(inputs=h6, filters=128, kernel_size=(1, 3, 3), dilation=(1, 8, 8))
            h8 = self._conv_3D(inputs=h7, filters=128, kernel_size=(3, 1, 1), dilation=(1, 1, 1))
            h9 = self._conv_3D(inputs=h8, filters=128, kernel_size=(1, 3, 3), dilation=(1, 16, 16))
            h10 = self._conv_3D(inputs=h9, filters=128, kernel_size=(3, 1, 1), dilation=(1, 1, 1))
            h11 = self._conv_3D(inputs=h10, filters=64, kernel_size=(1, 1, 1), dilation=(1, 1, 1))

            output = h11
            # output = self._conv_3D(inputs=h11, filters=self._outputDim, kernel_size=(3, 3, 3), dilation=(1, 1, 1))
        # self._reuse = True
        self.variables = [self._CNet_core.variables,
                          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name+"_Detection")]
        self.update_ops = [self._CNet_core.update_ops,
                           tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._name+"_Detection")]
        self.allVariables = self._CNet_core.variables + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                          scope=self._name+"_Detection")
        self.allUpdate_ops = self._CNet_core.update_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                                            scope=self._name+"_Detection")
        self.coreVariables = self._CNet_core.variables
        self.colorizorVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name+"_Detection")
        self.coreSaver = tf.train.Saver(var_list=self.coreVariables,
                                        max_to_keep=4, keep_checkpoint_every_n_hours=2)
        self.colorizorSaver = tf.train.Saver(var_list=self.colorizorVariables,
                                             max_to_keep=4, keep_checkpoint_every_n_hours=2)

        # output = tf.reshape(output, [-1] + output.get_shape().as_list()[2:])
        return output


# x = tf.placeholder(tf.float32, [None, 4, 256, 256, 1], 'input')
# y = CN_Colorize()
# out = y(x)
# print out.get_shape()
