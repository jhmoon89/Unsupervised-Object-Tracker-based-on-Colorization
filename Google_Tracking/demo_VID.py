import numpy as np
import time, sys
import tensorflow as tf
import dataset_utils.dataset_loader.ImagenetVid_CN_3D_dataset as ImagenetVid_CN_dataset
import dataset_utils.dataset_loader.KITTI_dataset as KITTI_dataset
import src.module.demo_tracker as tracker
import math
import datetime
import dataset_utils.datasetUtils as datasetUtils
import matplotlib.pyplot as plt
import cv2
import os

CNConfig = {
    'classDim': 31,
    'consecutiveFrame': 4
}


# cropping function for KITTI dataset
def img_crop(img):
    H, W, _ = img.shape
    crop_range_start = int(W / 6)
    crop_range_end = int(W / 6 * 5)
    return img[:, crop_range_start:crop_range_end, ...].astype('uint8')

def get_IOU(img1, img2):
    assert np.array_equal(img1.shape, img2.shape)
    H, W, quant_channel = img1.shape

    # black color : index 8
    # non-zero means something is segmented
    mask1 = (np.argmax(img1, -1) != 8)
    mask2 = (np.argmax(img2, -1) != 8)

    mask_union = np.logical_or(mask1, mask2).astype('float32')
    mask_intersection = np.logical_and(mask1, mask2).astype('float32')

    if np.sum(mask_union) == 0:
        iou = 0
    else:
        iou = np.sum(mask_intersection) / np.sum(mask_union)
    return iou

def Run_DAVIS(index, weight_restore_path='/ssdubuntu/Google_Tracker_Weights/4frames/'
                                         '181110_RandStepSeq_noflip_VIDKITTI/2.lr1e-5',
              display_image=True, vid_save=False, eval=False, propagate=False, rand_ref=False, img_save=False,
              img_save_prefix=''):
    txt_file = open('/hdd/data/DAVIS/DAVIS/ImageSets/2017/train.txt', 'r')
    vid_list = txt_file.readlines()

    # # random index
    # index = np.random.randint(0, len(vid_list))
    # # fix index
    # index = 17
    # # find index by search
    # key_word = 'walking'
    # index = vid_list.index(key_word + '\n')

    obj_cls = vid_list[index].replace('\n', '')
    ###########################################
    # video options
    if vid_save:
        fps = 20.0
        size = (224 * 2, 224)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_name = '/hdd/figures/DAVIS_GOOGLE/' + obj_cls + '.avi'
        out_vid = cv2.VideoWriter(video_name, fourcc, fps, size)
    ###########################################
    img_path = '/hdd/data/DAVIS/DAVIS/JPEGImages/Full-Resolution/' + obj_cls
    seg_path = '/hdd/data/DAVIS/DAVIS/Annotations_semantics/Full-Resolution/' + obj_cls
    seg_path2 = '/hdd/data/DAVIS/DAVIS/Annotations/Full-Resolution/' + obj_cls

    img_list = os.listdir(img_path)
    img_list.sort(key=datasetUtils.natural_keys)
    seg_list = os.listdir(seg_path)
    seg_list.sort(key=datasetUtils.natural_keys)

    img_num = len(img_list)

    images = []
    images_small = []
    segs_up = []
    segs = []
    Q = []

    for i in range(img_num):
        img = cv2.resize(cv2.imread(os.path.join(img_path, img_list[i])), (224, 224))
        img_small = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2Lab)
        images.append(img)
        images_small.append(img_small)
        seg_up = cv2.resize(cv2.imread(os.path.join(seg_path2, seg_list[i])), (224, 224))
        seg = cv2.resize(cv2.imread(os.path.join(seg_path2, seg_list[i])), (28, 28))
        q = datasetUtils.img_to_quant(seg)
        segs.append(seg)
        segs_up.append(seg_up)
        Q.append(q)

    images = np.array(images)
    segs = np.array(segs)
    images_small = np.array(images_small)
    Q = np.array(Q)

    # print images.shape
    # print segs.shape

    model = tracker.Google_Tracker(consecutiveFrame=CNConfig['consecutiveFrame'])
    model.restoreNetworks(restorePath=weight_restore_path)

    iou_list = np.zeros((img_num - 4,))

    if propagate:
        print 'propagate outputs...'
        next_ref = None
        for i in range(img_num - 4):
            current_batch = images[i:i + 4, ..., 0:1]
            batch = {'InputImages': (current_batch[None, ...] / 255.0 * 2 - 1).astype('float32')}
            out = np.array(model.fit(batch))
            # print out.shape
            ref = out[0, 2, ...].reshape([-1, 64])
            tar = out[0, 3, ...].reshape([-1, 64])
            A = np.matmul(ref, np.transpose(tar)).astype('float64')
            A = np.exp(A)
            A = A / np.sum(A, axis=0, keepdims=True)

            if i  ==0 :
                seg_ref = Q[i + 2, ...].reshape([-1, 16])
            else:
                seg_ref = next_ref
            seg_tar_pred = np.matmul(np.transpose(A), seg_ref)
            seg_tar_pred = seg_tar_pred.reshape([28, 28, 16])
            next_ref = seg_tar_pred.reshape([-1, 16])
            # mask_tar_pred = np.matmul(np.transpose(A), mask_ref).reshape([28, 28])
            tar_RGB = datasetUtils.quant_to_img(seg_tar_pred, images_small[i + 2, ...][..., 0])
            tar_Lab_up = cv2.cvtColor(cv2.resize(tar_RGB, (224, 224)), cv2.COLOR_BGR2Lab)
            tar_Lab_up[..., 0] = images[i + 3, ..., 0]
            tar_RGB_up = cv2.cvtColor(tar_Lab_up.astype('uint8'), cv2.COLOR_Lab2BGR)

            final = np.concatenate([tar_RGB_up.astype('uint8'),
                                    cv2.cvtColor(images[i + 3, ...], cv2.COLOR_Lab2BGR),
                                    segs_up[i + 3]], 1)

            if eval:
                # mask = np.equal(np.argmax(seg_tar_pred, -1), np.argmax(Q[i + 3, ...], -1)).astype('float')
                iou_list[i] = get_IOU(seg_tar_pred, Q[i + 3, ...])
            if display_image:
                cv2.imshow('k', final)
                # cv2.imshow('k', np.concatenate([(mask_tar_pred > 0.5).astype('uint8'), mask_ref.reshape([28, 28])], 1))
                cv2.waitKey(30)
            if vid_save:
                out_vid.write(final)

        if vid_save:
            out_vid.release()
        if display_image:
            cv2.destroyAllWindows()
        return iou_list

    else:
        print 'fix reference frames...'
        if rand_ref:
            print 'reference is fixed but picked randomly...'
            fixed_ref_ind = np.random.randint(0, img_num, (3,))
            first_batch = images[np.array(fixed_ref_ind), ..., 0:1]
        else:
            print 'reference is 0123...'
            first_batch = images[0:3, ..., 0:1]
        for i in range(img_num - 4):
            current_batch = images[i:i+4, ..., 0:1]
            current_batch[0:3, ...] = first_batch
            batch = {'InputImages': (current_batch[None, ...] / 255.0 * 2 - 1).astype('float32')}
            out = np.array(model.fit(batch))
            # print out.shape
            ref = out[0, 2, ...].reshape([-1, 64])
            tar = out[0, 3, ...].reshape([-1, 64])
            A = np.matmul(ref, np.transpose(tar)).astype('float64')
            # print A.shape

            A = np.exp(A)
            A = A / np.sum(A, axis=0, keepdims=True)

            # mask_ref = (np.sum(segs[i + 2, ...], -1) != 0).reshape([-1]).astype('float32')
            # cv2.imshow('k1', mask_ref.reshape([28, 28]).astype('uint8') * 100)
            # print mask_ref.shape
            seg_ref = Q[i + 2, ...].reshape([-1, 16])
            seg_tar_pred = np.matmul(np.transpose(A), seg_ref)
            seg_tar_pred = seg_tar_pred.reshape([28, 28, 16])
            cur = seg_tar_pred
            # mask_tar_pred = np.matmul(np.transpose(A), mask_ref).reshape([28, 28])
            tar_RGB = datasetUtils.quant_to_img(seg_tar_pred, images_small[i + 2, ...][..., 0])
            tar_Lab_up = cv2.cvtColor(cv2.resize(tar_RGB, (224, 224)), cv2.COLOR_BGR2Lab)
            tar_Lab_up[..., 0] = images[i + 3, ..., 0]
            tar_RGB_up = cv2.cvtColor(tar_Lab_up.astype('uint8'), cv2.COLOR_Lab2BGR)

            final = np.concatenate([tar_RGB_up.astype('uint8'),
                                    cv2.cvtColor(images[i + 3, ...], cv2.COLOR_Lab2BGR),
                                    segs_up[i + 3]], 1)

            if eval:
                # mask = np.equal(np.argmax(seg_tar_pred, -1), np.argmax(Q[i + 3, ...], -1)).astype('float')
                iou_list[i] = get_IOU(seg_tar_pred, Q[i + 3, ...])
            if display_image:
                cv2.imshow('k', final)
                # cv2.imshow('k', np.concatenate([(mask_tar_pred > 0.5).astype('uint8'), mask_ref.reshape([28, 28])], 1))
                cv2.waitKey(30)
            if img_save:
                img_save_path = '/hdd/figures/DAVIS_GOOGLE/' + \
                                img_save_prefix + '/' + obj_cls + '/'
                if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path)
                cv2.imwrite(img_save_path + str(i).zfill(6) + '.png', final)
                # cv2.imwrite('/hdd/figures/DAVIS_GOOGLE/'+ str(i).zfill(6) + '.png', final)
            if vid_save:
                out_vid.write(final)

        if vid_save:
            out_vid.release()
        if display_image:
            cv2.destroyAllWindows()
        return iou_list


########################################################################################################################
weight_restore_path = '/ssdubuntu/Google_Tracker_Weights/4frames/181110_RandStepSeq_noflip_VIDKITTI/2.lr1e-5'
weight_restore_path2 = '/ssdubuntu/Google_Tracker_Weights/4frames/181106_SeqInput_VID/1.lr1e-4'

txt_file = open('/hdd/data/DAVIS/DAVIS/ImageSets/2017/train.txt', 'r')
vid_list = txt_file.readlines()

frame_cut = 200
print len(vid_list)
iou_rand_step_seq = np.zeros([len(vid_list), 200])
iou_seq = np.zeros([len(vid_list), 200])

iou_list = Run_DAVIS(0, weight_restore_path=weight_restore_path, display_image=True,
                     vid_save=False, eval=False, propagate=False, rand_ref=False)
# print iou_list
# iou_list2 = Run_DAVIS(0, weight_restore_path=weight_restore_path2, display_image=True,
#                       vid_save=False, eval=False, propagate=False, rand_ref=True)
# print iou_list2
#####################################################
# for i in range(len(vid_list)):
#     iou_list = Run_DAVIS(i, weight_restore_path=weight_restore_path, display_image=False,
#                          vid_save=False, eval=False, propagate=False, rand_ref=True, img_save=True,
#                          img_save_prefix='RandStepSeq')
#     # print iou_list
#     iou_list2 = Run_DAVIS(i, weight_restore_path=weight_restore_path2, display_image=False,
#                           vid_save=False, eval=False, propagate=False, rand_ref=True, img_save=True,
#                           img_save_prefix='Seq')
#     # print iou_list2
#
#     min_ind1 = min(200, len(iou_list))
#     min_ind2 = min(200, len(iou_list2))
#     iou_rand_step_seq[i, :min_ind1] = iou_list[:min_ind1]
#     iou_seq[i, :min_ind2] = iou_list2[:min_ind2]

# save_path_1 = '/hdd/eval/rand_step_seq_fixref0123.npy'
# save_path_2 = '/hdd/eval/seq_fixref0123.npy'

# np.save(save_path_1, iou_rand_step_seq)
# np.save(save_path_2, iou_seq)
#####################################################
