import numpy as np
import time, sys
import tensorflow as tf
import dataset_utils.dataset_loader.ImagenetVid_CN_3D_dataset as ImagenetVid_CN_dataset
import dataset_utils.dataset_loader.KITTI_dataset as KITTI_dataset
import src.module.tracker as tracker
import math
import datetime
import dataset_utils.datasetUtils as datasetUtils
import matplotlib.pyplot as plt
import cv2

CNConfig = {
    'classDim': 31,
    'consecutiveFrame': 4
}


def trainTracker(
        CNConfig, batchSize=8, training_epoch=10,
        learningRate=0.001,
        savePath=None, restorePath=None
):

    ##################################################################
    # load Imagenet VID dataset
    Imagenet_VID_datasetPath = '/ssdubuntu/data/ILSVRC'
    ImagenetVID_Dataset = ImagenetVid_CN_dataset.imagenetVidDataset(Imagenet_VID_datasetPath,
                                                                    consecutiveLength=CNConfig['consecutiveFrame'],
                                                                    classNum=CNConfig['classDim'])
    ##################################################################
    # load KITTI dataset
    KITTI_datasetPath = '/hdd/data/KITTI'
    KITTI_Dataset = KITTI_dataset.KITTI_Dataset(dataPath=KITTI_datasetPath,
                                                consecutiveLength=CNConfig['consecutiveFrame'])
    ##################################################################
    ImagenetVID_Dataset.setImageSize((224, 224))
    KITTI_Dataset.setImageSize((224, 224))
    model = tracker.Google_Tracker(consecutiveFrame=CNConfig['consecutiveFrame'])
    # dataset = ImagenetVID_Dataset

    if restorePath != None:
        print 'restore weights...'
        # ### only restore core
        # model.restoreTrackerCore(restorePath)

        ### restore every weights
        model.restoreNetworks(restorePath)

    loss = 0.0
    acc = 0.0
    top5acc = 0.0
    epoch = 0
    iteration = 0
    run_time = 0.0
    if learningRate == None:
        learningRate = 0.001

    veryStart = time.time()
    print 'start training...'

    # while epoch < training_epoch:
    #     iteration = 0
    #     for cursor in range(max_iter):
    #         start = time.time()
    #         iteration = cursor
    #         batchData = dataset.getNextBatch(batchSize=batchSize)
    #         batchData['LearningRate'] = learningRate
    #         epochCurr = dataset._epoch
    #         dataStart = dataset._dataStart
    #         dataLength = dataset._dataLength
    #         if epochCurr != epoch:
    #             epoch = epochCurr
    #             break
    #
    #         lossTemp, accTemp = model.fit(batchData)
    #
    #         end = time.time()
    #         loss = float(loss * iteration + lossTemp) / float(iteration + 1.0)
    #         acc = float(acc * iteration + accTemp) / float(iteration + 1.0)
    #         run_time = (run_time * iteration + (end - start)) / float(iteration + 1.0)
    #
    #         sys.stdout.write(
    #             "Epoch:{:03d} iter:{:05d} runtime:{:.3f} ".format(int(epoch + 1), int(iteration + 1), run_time))
    #         sys.stdout.write("cur/tot:{:07d}/{:07d} ".format(dataStart, dataLength))
    #         # sys.stdout.write("Current Loss={:.6f} ".format(lossTemp))
    #         sys.stdout.write("Average Loss={:.6f} ".format(loss))
    #         sys.stdout.write("Average Acc={:.6f}% ".format(acc * 100))
    #         sys.stdout.write("\n")
    #         sys.stdout.flush()
    #
    #         if math.isnan(loss):
    #             break
    #
    #         if cursor != 0 and cursor % 2000 == 0:
    #             model.saveNetworks(savePath)
    #
    #     if math.isnan(loss):
    #         break
    #
    #     if savePath != None:
    #         print 'save model...'
    #         model.saveNetworks(savePath)
    #
    #     dataset.newEpoch()
    #     epoch += 1
    ###############################################################################################
    while epoch < training_epoch:

        start = time.time()
        #########################################################
        # # for seq input
        # batchData = dataset.getNextBatch(batchSize=batchSize)
        # dataLength = dataset._dataLength
        #########################################################
        # for rand input
        # ImgVID_batchData = ImagenetVID_Dataset.getNextBatch_Random(batchSize=batchSize / 2)
        ImgVID_batchData = ImagenetVID_Dataset.getNextBatch_RandStepSeq(batchSize=batchSize / 2, max_step=10)
        ImgVID_dataLength = ImagenetVID_Dataset._semi_final_dataLength
        #########################################################
        # for KITTI
        # KITTI_batchData = KITTI_Dataset.getNextBatch_Random(batchSize=batchSize / 2)
        KITTI_batchData = KITTI_Dataset.getNextBatch_RandStepSeq(batchSize=batchSize / 2, max_step=10)
        KITTI_dataLength = KITTI_Dataset._dataLength
        #########################################################
        # merge batch of two datasets
        batchData =\
            {'InputImages': np.concatenate([ImgVID_batchData['InputImages'], KITTI_batchData['InputImages']], 0),
             'OutputImages': np.concatenate([ImgVID_batchData['OutputImages'], KITTI_batchData['OutputImages']], 0)
             }
        #########################################################
        batchData['LearningRate'] = learningRate
        epochCurr = ImagenetVID_Dataset._epoch
        dataStart = ImagenetVID_Dataset._dataStart

        dataLength = ImgVID_dataLength

        if epochCurr != epoch or ((iteration + 1) % 1000 == 0 and (iteration + 1) != 1):
        # if (iteration + 1) % 1000 == 0 and (iteration + 1) != 1:
            print ''
            # iteration = 0
            # loss = loss * 0.0
            # run_time = 0.0
            if savePath != None:
                print 'save model...'
                model.saveNetworks(savePath)
        epoch = epochCurr

        lossTemp, accTemp, top5accTemp = model.fit(batchData)

        end = time.time()
        loss = float(loss * iteration + lossTemp) / float(iteration + 1.0)
        acc = float(acc * iteration + accTemp) / float(iteration + 1.0)
        top5acc = float(top5acc * iteration + top5accTemp) / float(iteration + 1.0)

        run_time = (run_time * iteration + (end - start)) / float(iteration + 1.0)

        sys.stdout.write(
            "Epoch:{:03d} iter:{:05d} runtime:{:.3f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:07d}/{:07d} ".format(dataStart, dataLength))
        sys.stdout.write("Average Loss={:.6f} ".format(loss))
        sys.stdout.write("Average Acc={:.6f}% ".format(acc * 100))
        sys.stdout.write("Average Top5 Acc={:.6f}% ".format(top5acc * 100))
        sys.stdout.write("\n")
        sys.stdout.flush()

        iteration = iteration + 1.0

    veryEnd = time.time()
    sys.stdout.write("total training time:" + str(datetime.timedelta(seconds=veryEnd - veryStart)))


if __name__ == "__main__":
    sys.exit(trainTracker(CNConfig=CNConfig
        , batchSize=32
        , training_epoch=30000
        , learningRate=1e-5
        # , savePath='/ssdubuntu/ColorNet_Weights/2frames/180928/1.lr1e-3epoch1000'
        # , restorePath='/ssdubuntu/ColorNet_Weights/2frames/180928/1.lr1e-3epoch1000'
        # , savePath='/ssdubuntu/ColorNet_Weights/2frames/181001/2.lr1e-5epoch1000kernel1'
        # , restorePath='/ssdubuntu/ColorNet_Weights/2frames/181001/2.lr1e-5epoch1000kernel1'
        , savePath='/ssdubuntu/Google_Tracker_Weights/4frames/181110_RandStepSeq_noflip_VIDKITTI/2.lr1e-5'
        , restorePath='/ssdubuntu/Google_Tracker_Weights/4frames/181110_RandStepSeq_noflip_VIDKITTI/1.lr1e-4'
    ))

