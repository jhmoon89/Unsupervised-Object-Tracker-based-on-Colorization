import os, cv2
import numpy as np
import sys
from PIL import Image
from xml.etree.cElementTree import parse
import xml.etree.ElementTree as ET
# import dataset_utils.datasetUtils as datasetUtils
import time
from scipy import ndimage as ndimage

#################################################
import dataset_utils.datasetUtils as datasetUtils

class KITTI_Dataset(object):
    def __init__(self, dataPath='/hdd/data/KITTI', consecutiveLength=4):
        # print os.listdir(dataPath)
        self._gamut = np.load("/ssdubuntu/color/gamut.npy")
        self._binNum = 16
        self._img_reduce_factor = 8
        # self._tree = None
        self._dataPath = dataPath
        # self._classNum = classNum
        self._epoch = 0
        self._dataStart = 0
        self._dataLength = 0
        self._minLength = 40
        self._dataPointPathList = []
        self._subPathList = []
        self._classIdxConverter = None
        self._imageSize = (224, 224)
        self._consecutiveLength = consecutiveLength
        self._loadDataPointPath()
        # self._dataShuffle()

    def setImageSize(self, size=(224, 224)):
        self._imageSize = (size[0], size[1])

    def img_crop(self, img):
        H, W, _ = img.shape
        crop_range_start = int(W / 6)
        crop_range_end = int(W / 6 * 5)
        return img[:, crop_range_start:crop_range_end, ...].astype('uint8')

    def index_to_onehot_bin(self, np_file):
        return (np.arange(self._binNum) == np_file[..., None]).astype('float32')

    def _loadDataPointPath(self):
        print 'load data point path...'
        trainPath = os.path.join(self._dataPath, 'train')
        subtrainPathList = os.listdir(trainPath)
        for subtrainPath in subtrainPathList:
            subsubPath2 = os.path.join(trainPath, subtrainPath, "image_02", "data")
            subsubPath3 = os.path.join(trainPath, subtrainPath, "image_03", "data")
            self._subPathList.append(subsubPath2)
            # self._subPathList.append(subsubPath3)

        self._dataLength = len(self._subPathList)
        print 'load done!'

    def _dataShuffle(self):
        self._dataStart = 0
        np.random.shuffle(self._subPathList)
        print "KITTI shuffle done!\n"

    def newEpoch(self):
        self._epoch += 1
        self._dataStart = 0
        self._dataShuffle()

    # def getNextBatch_Random(self, batchSize=32):
    #     if self._dataStart + batchSize >= self._dataLength:
    #         print 'new epoch'
    #         self.newEpoch()
    #     dataStart = self._dataStart
    #     dataEnd = dataStart + batchSize
    #     self._dataStart = self._dataStart + batchSize
    #
    #     dataPathTemp = self._subPathList[dataStart:dataEnd]
    #
    #     Img_List_Batch = np.zeros([batchSize,
    #                                self._consecutiveLength,
    #                                self._imageSize[0],
    #                                self._imageSize[1],
    #                                1], dtype=np.float32)
    #     Q_List_Batch_small = np.zeros([batchSize,
    #                                    self._consecutiveLength,
    #                                    self._imageSize[0] / self._img_reduce_factor,
    #                                    self._imageSize[1] / self._img_reduce_factor,
    #                                    self._binNum], dtype=np.float32)
    #
    #     for i in range(len(dataPathTemp)):
    #         path = dataPathTemp[i]
    #         img_path_list = os.listdir(path)
    #         np.random.shuffle(img_path_list)
    #         for j in range(self._consecutiveLength):
    #             ###################################################################################
    #             # load image
    #             img_name = os.path.join(path, img_path_list[j])
    #             img_RGB = cv2.imread(img_name, cv2.IMREAD_COLOR)
    #             ####################################################
    #             # add salt & pepper noise
    #             if (np.random.rand() >= 0.5 and j != self._consecutiveLength - 1):
    #                 img_RGB = datasetUtils.noisy("s&p", img_RGB)
    #             ####################################################
    #             # add Gaussian noise
    #             if (np.random.rand() >= 0.5 and j != self._consecutiveLength - 1):
    #                 img_RGB = cv2.GaussianBlur(img_RGB, (5, 5), 0)
    #                 # img_BGR = datasetUtils.noisy("gauss", img_BGR)
    #             ####################################################
    #             # # add flipping
    #             # if (np.random.rand() >= 0.5 and j != self._consecutiveLength - 1):
    #             #     img_RGB = cv2.flip(img_RGB, flipCode=np.random.randint(-1, 2))
    #             ####################################################
    #             img_RGB = cv2.resize(img_RGB, self._imageSize)
    #             img_RGB_small = cv2.resize(img_RGB, (self._imageSize[0] / self._img_reduce_factor,
    #                                                  self._imageSize[1] / self._img_reduce_factor))
    #             img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2Lab)
    #             img_Lab_small = cv2.cvtColor(img_RGB_small, cv2.COLOR_BGR2Lab)
    #             ###################################################################################
    #             # assign
    #             Q_List_Batch_small[i, j, ...] = bin_img_smooth
    #             Img_List_Batch[i, j, ..., :3] = img_Lab
    #             ########################################
    #             # 4th channel L
    #             Img_List_Batch[i, j, ..., -1] = img_Lab[..., 0]
    #             ########################################
    #             if j == self._consecutiveLength - 1:
    #                 Img_List_Batch[i, j, ..., :3] = np.tile(img_Lab[..., 0:1], [1, 1, 3])
    #
    #     final_batchData = {
    #         'Paths': dataPathTemp,
    #         'InputImages': Img_List_Batch,
    #         'OutputImages': Q_List_Batch_small[:, -1, ...]
    #     }
    #
    #     return final_batchData


    def getNextBatch_RandStepSeq(self, batchSize=32, max_step=30):
        if self._dataStart + batchSize >= self._dataLength:
            print 'new epoch'
            self.newEpoch()
        dataStart = self._dataStart
        dataEnd = dataStart + batchSize
        self._dataStart = self._dataStart + batchSize

        dataPathTemp = self._subPathList[dataStart:dataEnd]

        Img_List_Batch = np.zeros([batchSize,
                                   self._consecutiveLength,
                                   self._imageSize[0],
                                   self._imageSize[1],
                                   1], dtype=np.float32)
        L_List_Batch_small = np.zeros([batchSize,
                                       self._consecutiveLength,
                                       self._imageSize[0] / self._img_reduce_factor,
                                       self._imageSize[1] / self._img_reduce_factor,
                                       1], dtype=np.float32)
        Q_List_Batch_small = np.zeros([batchSize,
                                       self._consecutiveLength,
                                       self._imageSize[0] / self._img_reduce_factor,
                                       self._imageSize[1] / self._img_reduce_factor,
                                       self._binNum], dtype=np.float32)

        for i in range(len(dataPathTemp)):
            path = dataPathTemp[i]
            img_path_list = os.listdir(path)
            img_path_list.sort(key=datasetUtils.natural_keys)
            rand_int = np.random.randint(1, max_step + 1, (self._consecutiveLength - 1,))
            rand_int[::-1].sort()
            rand_int = np.append(rand_int, 0)
            rand_tar = np.random.randint(max_step, len(img_path_list))
            for j in range(self._consecutiveLength):
                ###################################################################################
                # load image
                img_name = os.path.join(path, img_path_list[rand_tar - rand_int[j]])
                img_RGB = cv2.imread(img_name, cv2.IMREAD_COLOR)
                # crop image
                img_RGB = self.img_crop(img_RGB)
                ####################################################
                # add salt & pepper noise
                if (np.random.rand() >= 0.5 and j != self._consecutiveLength - 1):
                    img_RGB = datasetUtils.noisy("s&p", img_RGB)
                ####################################################
                # add Gaussian noise
                if (np.random.rand() >= 0.5 and j != self._consecutiveLength - 1):
                    img_RGB = cv2.GaussianBlur(img_RGB, (5, 5), 0)
                    # img_BGR = datasetUtils.noisy("gauss", img_BGR)
                ####################################################
                # # add flipping
                # if (np.random.rand() >= 0.5 and j != self._consecutiveLength - 1):
                #     img_RGB = cv2.flip(img_RGB, flipCode=np.random.randint(-1, 2))
                ####################################################
                img_RGB = cv2.resize(img_RGB, self._imageSize)
                img_RGB_small = cv2.resize(img_RGB, (self._imageSize[0] / self._img_reduce_factor,
                                                     self._imageSize[1] / self._img_reduce_factor))
                img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2Lab)
                img_Lab_small = cv2.cvtColor(img_RGB_small, cv2.COLOR_BGR2Lab)
                ###################################################################################
                # assign
                Q_List_Batch_small[i, j, ...] = datasetUtils.img_to_quant(img_RGB_small)
                # smoothing
                Q_List_Batch_small[i, j, ...] += np.ones((self._binNum,), dtype=np.float32) / self._binNum
                Q_List_Batch_small[i, j, ...] /= np.sum(Q_List_Batch_small[i, j, ...], -1, keepdims=True)
                ########################################
                # img
                Img_List_Batch[i, j, ...] = img_Lab[..., 0:1] / 255.0 * 2 - 1
                ########################################
                # L list
                L_List_Batch_small[i, j, ...] = img_Lab_small[..., 0:1]

        final_batchData = {
            'Paths': dataPathTemp,
            'InputImages': Img_List_Batch,
            'Small_Inputs': L_List_Batch_small,
            'OutputImages': Q_List_Batch_small
        }

        return final_batchData


#####################################################################################################
# # test
# vid_data = KITTI_Dataset(consecutiveLength=4)
# print vid_data._dataLength
# sample = vid_data.getNextBatch_RandStepSeq(3)
# input = sample['InputImages']
# output = sample['OutputImages']
# small_L = sample['Small_Inputs']
#
# print input.shape
# print output.shape
#
# # print output
# # print np.max(output, -1)
# # print np.array_equal(np.sum(output, -1), np.ones_like(np.sum(output, -1)))
#
# for i in range(input.shape[0]):
#     for j in range(4):
#         cv2.imshow('input', ((input[i, j, ...] + 1) / 2.0 * 255).astype('uint8'))
#         L = small_L[i, j, ...].astype('uint8')
#         ab = output[i, j, ...]
#         # print ab.shape
#         # print L.shape
#         cv2.imshow('output', datasetUtils.quant_to_img(ab, L[..., 0]))
#         cv2.waitKey(0)
# cv2.destroyAllWindows()
#####################################################################################################
