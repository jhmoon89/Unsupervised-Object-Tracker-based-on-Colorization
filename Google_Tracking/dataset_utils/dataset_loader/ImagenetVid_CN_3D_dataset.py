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

def imageResize(imagePath, imageSize, bbox):
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if bbox != None:
        imageBbox = image[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
        if len(imageBbox) == 0 or len(imageBbox[0]) == 0:
            imageResult = image
        else:
            imageResult = imageBbox
    else:
        imageResult = image
    imageResult = datasetUtils.imgAug(imageResult)
    imageResult = cv2.resize(imageResult, imageSize)
    return imageResult


class imagenetVidDataset(object):
    def __init__(self, dataPath, consecutiveLength=4, classNum=31):
        self._classes = ['__background__',  # always index 0
                         'airplane', 'antelope', 'bear', 'bicycle',
                         'bird', 'bus', 'car', 'cattle',
                         'dog', 'domestic_cat', 'elephant', 'fox',
                         'giant_panda', 'hamster', 'horse', 'lion',
                         'lizard', 'monkey', 'motorcycle', 'rabbit',
                         'red_panda', 'sheep', 'snake', 'squirrel',
                         'tiger', 'train', 'turtle', 'watercraft',
                         'whale', 'zebra']
        self._classes_map = ['__background__',  # always index 0
                             'n02691156', 'n02419796', 'n02131653', 'n02834778',
                             'n01503061', 'n02924116', 'n02958343', 'n02402425',
                             'n02084071', 'n02121808', 'n02503517', 'n02118333',
                             'n02510455', 'n02342885', 'n02374451', 'n02129165',
                             'n01674464', 'n02484322', 'n03790512', 'n02324045',
                             'n02509815', 'n02411705', 'n01726692', 'n02355227',
                             'n02129604', 'n04468005', 'n01662784', 'n04530566',
                             'n02062744', 'n02391049']
        # self._gamut = np.load("/ssdubuntu/color/gamut_cell16.npy")
        self._binNum = 16
        self._img_reduce_factor = 8
        # self._tree = None
        self._dataPath = dataPath
        self._classNum = classNum
        self._epoch = 0
        self._dataStart = 0
        self._dataLength = 0
        self._minLength = 40
        self._dataPointPathList = []
        self._classIdxConverter = None
        self._imageSize = (224, 224)
        self._consecutiveLength = consecutiveLength
        self._loadDataPointPath()
        self._dataShuffle()

    def setImageSize(self, size=(224, 224)):
        self._imageSize = (size[0], size[1])

    def index_to_onehot_bin(self, np_file):
        return (np.arange(self._binNum) == np_file[..., None]).astype('float32')

    def _loadDataPointPath(self):
        print 'load data point path...'
        trainPath = os.path.join(self._dataPath, 'Data')
        trainPath = os.path.join(trainPath, 'VID')
        trainPath = os.path.join(trainPath, 'train')
        subtrainPathList = os.listdir(trainPath)
        subtrainPathList.sort(key=datasetUtils.natural_keys)
        # print subtrainPathList
        semi_finalPath_list = []
        subsubtrainPathList = []
        for subtrainpath in subtrainPathList:
            sub_sub_train_path = os.listdir(os.path.join(trainPath, subtrainpath))
            sub_sub_train_path.sort(key=datasetUtils.natural_keys)
            subsubtrainPathList.append(sub_sub_train_path)
            for k in sub_sub_train_path:
                self._singlevideoPathList = [] # temporary
                semi_finalPath = os.path.join(trainPath, subtrainpath, k)
                imgPathList = os.listdir(semi_finalPath)
                imgPathList.sort(key=datasetUtils.natural_keys)
                if len(imgPathList) < self._minLength:
                    continue
                semi_finalPath_list.append(semi_finalPath)
                for img in range(len(imgPathList) - self._consecutiveLength):
                    finalPath = os.path.join(semi_finalPath, imgPathList[img])
                    self._dataPointPathList.append(finalPath)
                    self._singlevideoPathList.append(finalPath)
                    # print self._dataPointPathList[-1]
                    #         print subsubtrainPathList[0]
                    #         print len(subsubtrainPathList)
                    #         print len(subsubtrainPathList[3])

        self._dataLength = len(self._dataPointPathList)
        self._single_dataLength = len(self._singlevideoPathList)
        self._semi_finalPathList = semi_finalPath_list
        self._semi_final_dataLength = len(self._semi_finalPathList)

        print 'load done!'

    def _dataShuffle(self):
        # 'data list shuffle...'
        self._dataStart = 0
        np.random.shuffle(self._dataPointPathList)
        np.random.shuffle(self._semi_finalPathList)
        print "ImgVID shuffle done!\n"
        # print len(self._dataPointPathList)
        # print self._dataPointPathList[0]

    def showImage(self, path, ind):
        index = str(ind).zfill(6)
        img_name = path + '/' + index + '.JPEG'

        # print img_name
        img = cv2.imread(img_name)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawRect(self, img, box):
        img = cv2.rectangle(img, (box[1], box[3]), (box[0], box[2]), (255, 0, 0), 3)
        return img

    def getConsecutiveImages(self, path, startind):

        # change from BGR to Lab and get (L, ab) channels
        # img_ab_List = np.zeros([self._consecutiveLength, self._imageSize[0], self._imageSize[1], 2])
        input_img_List = np.zeros([self._consecutiveLength,
                                   self._imageSize[0],
                                   self._imageSize[1],
                                   1], dtype=np.float32)
        L_List_small = np.zeros([self._consecutiveLength,
                                 self._imageSize[0] / self._img_reduce_factor,
                                 self._imageSize[1] / self._img_reduce_factor,
                                 1], dtype=np.float32)
        Q_List_small = np.zeros([self._consecutiveLength,
                                 self._imageSize[0] / self._img_reduce_factor,
                                 self._imageSize[1] / self._img_reduce_factor,
                                 self._binNum], dtype=np.float32)
        cls_List = np.zeros([self._consecutiveLength,
                             self._classNum], dtype=np.float32)

        # For loop for getting concurrent images
        for i in range(self._consecutiveLength):
            # current frame
            index = str(startind + i).zfill(6)
            img_name = path + '/' + index + '.JPEG'
            img_BGR = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img_BGR = cv2.resize(img_BGR, self._imageSize)
            img_BGR_small = cv2.resize(img_BGR, (self._imageSize[0] / self._img_reduce_factor,
                                                 self._imageSize[1] / self._img_reduce_factor))
            img_Lab_small = cv2.cvtColor(img_BGR_small, cv2.COLOR_BGR2Lab)
            ####################################################
            # add salt & pepper noise
            if (np.random.rand() >= 0.5 and i != self._consecutiveLength - 1):
                img_BGR = datasetUtils.noisy("s&p", img_BGR)
            ####################################################
            # add Gaussian noise
            if (np.random.rand() >= 0.5 and i != self._consecutiveLength - 1):
                img_BGR = cv2.GaussianBlur(img_BGR, (5, 5), 0)
                # img_BGR = datasetUtils.noisy("gauss", img_BGR)
            ####################################################
            # # add flipping
            # if (np.random.rand() >= 0.5 and i != self._consecutiveLength - 1):
            #     img_BGR = cv2.flip(img_BGR, flipCode=np.random.randint(-1, 2))
            ####################################################
            # get Lab image
            img_Lab = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2Lab).astype('float32')
            img_BGR = img_BGR.astype('float32')

            # load xml
            xml_path = img_name.replace('Data', 'Annotations')
            xml_path = xml_path.replace('JPEG', 'xml')
            xml_file = parse(xml_path)
            note = xml_file.getroot()
            obj_exist = note.find('object')
            if obj_exist != None:
                first_obj = obj_exist.findtext('name')
                first_obj_ind = int(self._classes_map.index(first_obj))
                # print self._classes[first_obj_ind]
                cls_List[i, first_obj_ind] = 1.0

            ####################################################
            # assign BGR to first 3 channels
            # input_img_List[i, ...] = img_BGR[..., 0:1
            ####################################################
            # assign L to input
            input_img_List[i, ...] = (img_Lab[..., 0:1] / 255.0) * 2 - 1
            ####################################################
            # assign quant img
            Q_List_small[i, ...] = datasetUtils.img_to_quant(img_BGR_small)

            # smoothing
            Q_List_small[i, ...] += np.ones((self._binNum,), dtype=np.float32) / self._binNum
            Q_List_small[i, ...] /= np.sum(Q_List_small[i, ...], -1, keepdims=True)
            ####################################################
            # assign small L
            L_List_small[i, ...] = img_Lab_small[..., 0:1]

        return input_img_List, Q_List_small, cls_List, L_List_small

    def newEpoch(self):
        self._epoch += 1
        self._dataStart = 0
        self._dataShuffle()

    # def convert_Lab_to_bin(self, Img_List_Batch, gamut_tile, batchnum):
    #     gamut_tile_tile = np.tile(np.expand_dims(gamut_tile, 0), [batchnum, 1, 1, 1, 1, 1])
    #     Img_List_Batch_tile = np.tile(np.expand_dims(Img_List_Batch[..., 1:].astype('uint8') / 8, -2),
    #                                   [1, 1, 1, 1, self._binNum, 1])
    #     Q_List_Batch = np.equal(gamut_tile_tile, Img_List_Batch_tile)
    #     Q_List_Batch = np.all(Q_List_Batch, -1).astype('float32')
    #     L_List_Batch = Img_List_Batch[..., 0:1]
    #     # toc = time.time()
    #     # print toc-tic
    #     Input_Batch = np.concatenate([L_List_Batch[:, 1:, ...], Q_List_Batch[:, :-1, ...]], -1)
    #     Input_Batch = Input_Batch.reshape(tuple([-1]) + Input_Batch.shape[2:])
    #     Output_Batch = Q_List_Batch[:, 1:, ...]
    #     Output_Batch = Output_Batch.reshape(tuple([-1]) + Output_Batch.shape[2:])
    #
    #     return Input_Batch, Output_Batch


    def getNextBatch(self, batchSize=32):

        verystartTime = time.time()
        if self._dataStart + batchSize >= self._dataLength:
        # if self._dataStart + batchSize >= self._single_dataLength:
            print 'new epoch'
            self.newEpoch()
        dataStart = self._dataStart
        dataEnd = dataStart + batchSize
        self._dataStart = self._dataStart + batchSize

        # print dataStart, dataEnd, self._dataLength

        # Getting Batch
        dataPathTemp = self._dataPointPathList[dataStart:dataEnd]
        # dataPathTemp = self._singlevideoPathList[dataStart:dataEnd] # temp: test for one video

        Img_List_Batch = np.zeros([batchSize,
                                  self._consecutiveLength,
                                  self._imageSize[0],
                                  self._imageSize[1],
                                  1], dtype=np.float32)
        Q_List_Batch_small = np.zeros([batchSize,
                                       self._consecutiveLength,
                                       self._imageSize[0] / self._img_reduce_factor,
                                       self._imageSize[1] / self._img_reduce_factor,
                                       self._binNum], dtype=np.float32)
        L_List_Batch_small = np.zeros([batchSize,
                                       self._consecutiveLength,
                                       self._imageSize[0] / self._img_reduce_factor,
                                       self._imageSize[1] / self._img_reduce_factor,
                                       1], dtype=np.float32)
        cls_List_Batch = np.zeros([batchSize,
                                   self._consecutiveLength,
                                   self._classNum], dtype=np.float32)

        for i in range(len(dataPathTemp)):
            path = dataPathTemp[i]
            parent_path = os.path.abspath(os.path.join(path, '..'))  # get the parent path
            ind = int(path.split('/')[-1].replace('.JPEG', ''))  # get the image index

            # then obtain consecutive images
            Img_List_Batch[i, ...], Q_List_Batch_small[i, ...], \
            cls_List_Batch[i, ...], L_List_Batch_small[i, ...] = \
                self.getConsecutiveImages(parent_path, ind)

        final_batchData = {
            'Paths': dataPathTemp,
            'InputImages': Img_List_Batch,
            'Small_Inputs': L_List_Batch_small,
            'OutputClasses': cls_List_Batch,
            'OutputImages': Q_List_Batch_small
        }

        return final_batchData

    # def getNextBatch_Random(self, batchSize=32):
    #
    #     if self._dataStart + batchSize >= self._semi_final_dataLength:
    #         # if self._dataStart + batchSize >= self._single_dataLength:
    #         print 'new epoch'
    #         self.newEpoch()
    #     dataStart = self._dataStart
    #     dataEnd = dataStart + batchSize
    #     self._dataStart = self._dataStart + batchSize
    #
    #     # Getting Batch
    #     dataPathTemp = self._semi_finalPathList[dataStart:dataEnd]
    #
    #     Img_List_Batch = np.zeros([batchSize,
    #                                self._consecutiveLength,
    #                                self._imageSize[0],
    #                                self._imageSize[1],
    #                                4], dtype=np.float32)
    #     Q_List_Batch_small = np.zeros([batchSize,
    #                                    self._consecutiveLength,
    #                                    self._imageSize[0] / 4,
    #                                    self._imageSize[1] / 4,
    #                                    self._binNum], dtype=np.float32)
    #     cls_List_Batch = np.zeros([batchSize,
    #                                self._consecutiveLength,
    #                                self._classNum], dtype=np.float32)
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
    #             # add flipping
    #             if (np.random.rand() >= 0.5 and j != self._consecutiveLength - 1):
    #                 img_RGB = cv2.flip(img_RGB, flipCode=np.random.randint(-1, 2))
    #             ####################################################
    #             img_RGB = cv2.resize(img_RGB, self._imageSize)
    #             img_RGB_small = cv2.resize(img_RGB, (self._imageSize[0] / 4, self._imageSize[1] / 4))
    #             img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2Lab)
    #             img_Lab_small = cv2.cvtColor(img_RGB_small, cv2.COLOR_BGR2Lab)
    #             ###################################################################################
    #             # load numpy
    #             numpy_name = img_name.replace("/ssdubuntu/data/ILSVRC/Data/VID/train", "/ssdubuntu/Bin_Data")
    #             numpy_name = numpy_name.replace(".JPEG", ".npy")
    #             numpy_name_small = numpy_name.replace(".npy", "_small.npy")
    #             numpy_small = np.load(numpy_name_small)
    #
    #             # get gt one-hot bin image
    #             bin_img_onehot = self.index_to_onehot_bin(numpy_small)
    #
    #             # get smoothed one-hot vector
    #             bin_img_smooth = bin_img_onehot + np.ones_like(bin_img_onehot) * 0.0001
    #             bin_img_smooth = bin_img_smooth / np.sum(bin_img_smooth, -1)[..., None]
    #             ###################################################################################
    #             # load xml
    #             xml_path = img_name.replace('Data', 'Annotations')
    #             xml_path = xml_path.replace('JPEG', 'xml')
    #             xml_file = parse(xml_path)
    #             note = xml_file.getroot()
    #             obj_exist = note.find('object')
    #             if obj_exist != None:
    #                 first_obj = obj_exist.findtext('name')
    #                 first_obj_ind = int(self._classes_map.index(first_obj))
    #                 # print self._classes[first_obj_ind]
    #                 cls_List_Batch[i, j, first_obj_ind] = 1.0
    #             ###################################################################################
    #             # assign
    #             Q_List_Batch_small[i, j, ...] = bin_img_smooth
    #             Img_List_Batch[i, j, ..., :3] = img_Lab
    #             ########################################
    #             # # 4th channel L
    #             # Img_List_Batch[i, j, ..., -1] = img_Lab[..., 0]
    #             ########################################
    #             # 4th channel Canny
    #             Img_List_Batch[i, j, ..., -1] = cv2.Canny(img_Lab[..., 0].astype('uint8'), 50, 100)
    #             ########################################
    #             if j == self._consecutiveLength - 1:
    #                 Img_List_Batch[i, j, ..., :3] = np.tile(img_Lab[..., 0:1], [1, 1, 3])
    #
    #     #         cv2.imshow('k', img_RGB)
    #     #         cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     final_batchData = {
    #         'Paths': dataPathTemp,
    #         'InputImages': Img_List_Batch,
    #         'OutputClasses': cls_List_Batch[:, -1, ...],
    #         'OutputImages': Q_List_Batch_small[:, -1, ...]
    #     }
    #
    #     return final_batchData
    #
    def getNextBatch_RandStepSeq(self, batchSize=32, max_step=30):
        assert max_step <= self._minLength

        if self._dataStart + batchSize >= self._semi_final_dataLength:
            # if self._dataStart + batchSize >= self._single_dataLength:
            print 'new epoch'
            self.newEpoch()
        dataStart = self._dataStart
        dataEnd = dataStart + batchSize
        self._dataStart = self._dataStart + batchSize

        # Getting Batch
        dataPathTemp = self._semi_finalPathList[dataStart:dataEnd]

        Img_List_Batch = np.zeros([batchSize,
                                   self._consecutiveLength,
                                   self._imageSize[0],
                                   self._imageSize[1],
                                   1], dtype=np.float32)
        Q_List_Batch_small = np.zeros([batchSize,
                                       self._consecutiveLength,
                                       self._imageSize[0] / self._img_reduce_factor,
                                       self._imageSize[1] / self._img_reduce_factor,
                                       self._binNum], dtype=np.float32)
        L_List_Batch_small = np.zeros([batchSize,
                                       self._consecutiveLength,
                                       self._imageSize[0] / self._img_reduce_factor,
                                       self._imageSize[1] / self._img_reduce_factor,
                                       1], dtype=np.float32)
        cls_List_Batch = np.zeros([batchSize,
                                   self._consecutiveLength,
                                   self._classNum], dtype=np.float32)

        for i in range(len(dataPathTemp)):
            path = dataPathTemp[i]
            img_path_list = os.listdir(path)
            img_path_list.sort(key=datasetUtils.natural_keys)
            rand_int = np.random.randint(1, max_step + 1, (self._consecutiveLength - 1,))
            rand_int[::-1].sort()
            rand_int = np.append(rand_int, 0)
            # print max_step, len(img_path_list)
            rand_tar = np.random.randint(max_step - 1, len(img_path_list))
            for j in range(self._consecutiveLength):
                ###################################################################################
                # load image
                img_name = os.path.join(path, img_path_list[rand_tar - rand_int[j]])
                img_RGB = cv2.imread(img_name, cv2.IMREAD_COLOR)
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
                # load xml
                xml_path = img_name.replace('Data', 'Annotations')
                xml_path = xml_path.replace('JPEG', 'xml')
                xml_file = parse(xml_path)
                note = xml_file.getroot()
                obj_exist = note.find('object')
                if obj_exist != None:
                    first_obj = obj_exist.findtext('name')
                    first_obj_ind = int(self._classes_map.index(first_obj))
                    # print self._classes[first_obj_ind]
                    cls_List_Batch[i, j, first_obj_ind] = 1.0
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

        # cv2.imshow('k', img_RGB)
        #         cv2.waitKey(0)
        # cv2.destroyAllWindows()
        final_batchData = {
            'Paths': dataPathTemp,
            'InputImages': Img_List_Batch,
            'Small_Inputs': L_List_Batch_small,
            'OutputClasses': cls_List_Batch,
            'OutputImages': Q_List_Batch_small
        }

        return final_batchData










#####################################################################################################
# data_path = '/ssdubuntu/data/ILSVRC'
# vid_data = imagenetVidDataset(data_path, consecutiveLength=4)
# # #
# sample = vid_data.getNextBatch_RandStepSeq(8)
# # sample = vid_data.getNextBatch(8)
# input = sample.get('InputImages')
# output = sample.get('OutputImages')
# small_L = sample['Small_Inputs']
# # print input.shape
# # print output.shape
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



