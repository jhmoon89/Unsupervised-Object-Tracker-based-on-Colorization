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


def Filter_KITTI_Seg_Img(img, index):
    # input : (H, W, 10), output: (H, W, 10)
    seg_color_list = np.array([[0, 0, 0], [0, 128, 192], [128, 0, 0], [64, 0, 128],
                               [64, 64, 0], [128, 64, 128], [0, 0, 192],
                               [128, 128, 128], [128, 128, 0], [192, 192, 0]]
                              )
    seg_color_list = np.flip(seg_color_list, -1)
    seg_cls_list = ['Background', 'Bicyclist', 'Building', 'Car', 'Pedestrian',
                    'Road', 'Sidewalk', 'Sky', 'Tree', 'VegetationMisc']

    H, W, _ = img.shape
    if type(index) == str:
        index = seg_cls_list.index(index)

    one_hot = (np.arange(len(seg_color_list)) == index).astype('int')
    one_hot_tile = np.tile(one_hot[None, None, ...], [H, W, 1])

    return one_hot_tile * img


def Seg_to_Ind_KITTI(img):
    seg_color_list = np.array([[0, 0, 0], [0, 128, 192], [128, 0, 0], [64, 0, 128],
                               [64, 64, 0], [128, 64, 128], [0, 0, 192],
                               [128, 128, 128], [128, 128, 0], [192, 192, 0]]
                              )
    seg_color_list = np.flip(seg_color_list, -1)
    seg_cls_list = ['Bicyclist', 'Building', 'Car', 'Pedestrian', 'Road', 'Sidewalk', 'Sky', 'Tree', 'VegetationMisc']

    H, W, _ = img.shape
    img_tile = np.tile(img[..., None, :], [1, 1, len(seg_color_list), 1])
    seg_tile = np.tile(seg_color_list[None, None, ...], [H, W, 1, 1])
    mask = np.equal(img_tile, seg_tile)
    mask = np.all(mask, -1)
    return mask.astype('int')


def Ind_to_Seg_KITTI(img):
    assert img.shape[-1] == 10
    seg_color_list = np.array([[0, 0, 0], [0, 128, 192], [128, 0, 0], [64, 0, 128],
                               [64, 64, 0], [128, 64, 128], [0, 0, 192],
                               [128, 128, 128], [128, 128, 0], [192, 192, 0]]
                              )
    seg_color_list = np.flip(seg_color_list, -1)
    seg_cls_list = ['Bicyclist', 'Building', 'Car', 'Pedestrian', 'Road', 'Sidewalk', 'Sky', 'Tree', 'VegetationMisc']

    ind = np.argmax(img, -1)
    return seg_color_list[np.array(ind)].astype('uint8')


def Merge_Ind_to_Seg_KITTI(imgs):
    # input : H, W, 3, 16
    H, W, img_num, channel = imgs.shape
    out = np.zeros([H, W, channel], dtype=np.float32)
    max_prob = np.max(imgs, -1)
    max_ind = np.argmax(max_prob, -1)
    for i in range(H):
        for j in range(W):
            out[i, j] = imgs[i, j, max_ind[i, j]]

    return out


def Get_KITTI_Img(img_path, img_size=(224, 224)):
    img = cv2.imread(img_path)
    img = img_crop(img)
    img = cv2.resize(img, img_size)
    return img


def Run_KITTI(weight_restore_path='/ssdubuntu/Google_Tracker_Weights/4frames/'
                                  '181110_RandStepSeq_noflip_VIDKITTI/2.lr1e-5',
              img_save_path='/hdd/figures/KITTI_181112/2011_09_30_drive_0027_RandStepSeq',
              feature_num=16):
    ######################################################################################
    # ref_seg_path = '/hdd/data/KITTI/huhe_semantic_labels/groundtruthImg'
    # seg_img_list = os.listdir(ref_seg_path)
    # seg_img_list.sort(key=datasetUtils.natural_keys)
    # ref_ind = np.random.randint(0, len(seg_img_list), (CNConfig['consecutiveFrame'] - 1))
    #
    #
    # obj_list = ['Tree', 'Car', 'Road']
    #
    # Segmented_Ref_Imgs = []
    # batch_np = np.zeros([1, CNConfig['consecutiveFrame'], 224, 224, 1], dtype=np.float32)
    #
    # for j in range(CNConfig['consecutiveFrame'] - 1):
    #     seg_img_path = os.path.join(ref_seg_path, seg_img_list[ref_ind[j]])
    #     # print seg_img_path
    #     ref_img_path = seg_img_path.replace('groundtruthImg', 'Images')
    #     rgb_img = cv2.imread(ref_img_path)
    #     rgb_img = img_crop(rgb_img)
    #     rgb_img = cv2.resize(rgb_img, (224, 224))
    #     rgb_img_small = cv2.resize(rgb_img, (28, 28))
    #     Lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2Lab)
    #     Lab_img_small = cv2.cvtColor(rgb_img_small, cv2.COLOR_BGR2Lab)
    #     seg_img = cv2.imread(seg_img_path)
    #     seg_img = img_crop(seg_img)
    #     seg_img = cv2.resize(seg_img, (28, 28))
    #     seg_img_quant = datasetUtils.img_to_quant(seg_img)
    #     # seg_img_huhu = Seg_to_Ind_KITTI(seg_img)
    #     # seg_img_huhu = Filter_KITTI_Seg_Img(seg_img_huhu, obj_list[j]).astype('float32')
    #
    #     batch_np[:, j, ...] = Lab_img[..., 0:1]
    #     Segmented_Ref_Imgs.append(seg_img_quant)
    #     # Segmented_Ref_Imgs.append(seg_img_huhu)
    # Segmented_Ref_Imgs = np.array(Segmented_Ref_Imgs)

    batch_np = np.zeros([1, CNConfig['consecutiveFrame'], 224, 224, 1], dtype=np.float32)
    img_path_temp1 = '/hdd/data/KITTI/huhe_semantic_labels/groundtruthImg/2011_09_30_drive_0027_0000000053.png'
    img_path_temp2 = '/hdd/data/KITTI/huhe_semantic_labels/groundtruthImg/2011_09_30_drive_0027_0000000087.png'
    img_path_temp3 = '/hdd/data/KITTI/huhe_semantic_labels/groundtruthImg/2011_09_30_drive_0027_0000000107.png'

    Segmented_Ref_Imgs = []
    ####################################
    img1 = Get_KITTI_Img(img_path_temp1, (28, 28))
    img1 = Seg_to_Ind_KITTI(img1)
    img1 = Filter_KITTI_Seg_Img(img1, 'Tree')
    img1 = Ind_to_Seg_KITTI(img1)
    img1 = datasetUtils.img_to_quant(img1)
    Segmented_Ref_Imgs.append(img1)
    ####################################
    img2 = Get_KITTI_Img(img_path_temp2, (28, 28))
    img2 = Seg_to_Ind_KITTI(img2)
    img2 = Filter_KITTI_Seg_Img(img2, 'Building')
    img2 = Ind_to_Seg_KITTI(img2)
    img2 = datasetUtils.img_to_quant(img2)
    Segmented_Ref_Imgs.append(img2)
    ####################################
    img3 = Get_KITTI_Img(img_path_temp3, (28, 28))
    img3 = Seg_to_Ind_KITTI(img3)
    img3 = Filter_KITTI_Seg_Img(img3, 'Road')
    img3 = Ind_to_Seg_KITTI(img3)
    img3 = datasetUtils.img_to_quant(img3)
    Segmented_Ref_Imgs.append(img3)
    ####################
    Segmented_Ref_Imgs = np.array(Segmented_Ref_Imgs)
    ###################
    # cv2.imshow('k', img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ####################################
    org_path_temp1 = '/hdd/data/KITTI/train/2011_09_30_drive_0027_sync/image_02/data/0000000053.png'
    org_path_temp2 = '/hdd/data/KITTI/train/2011_09_30_drive_0027_sync/image_02/data/0000000087.png'
    org_path_temp3 = '/hdd/data/KITTI/train/2011_09_30_drive_0027_sync/image_02/data/0000000107.png'
    org1 = Get_KITTI_Img(org_path_temp1)
    org2 = Get_KITTI_Img(org_path_temp2)
    org3 = Get_KITTI_Img(org_path_temp3)

    ########################################################
    batch_np[0, 0, ...] = cv2.cvtColor(org1, cv2.COLOR_BGR2Lab)[..., 0:1]
    batch_np[0, 1, ...] = cv2.cvtColor(org2, cv2.COLOR_BGR2Lab)[..., 0:1]
    batch_np[0, 2, ...] = cv2.cvtColor(org3, cv2.COLOR_BGR2Lab)[..., 0:1]

    # ##########################################################################################
    # # tar_path = '/hdd/data/KITTI/test/'
    # # tar_sub_path = os.listdir(tar_path)
    # # # tar_folder_ind = np.random.randint(0, len(tar_sub_path))
    # # tar_folder_ind = -2
    # # img_path = os.path.join(tar_path, tar_sub_path[tar_folder_ind]) + '/image_02/data'
    # # tar_rgb_list = os.listdir(img_path)
    # # tar_rgb_list.sort(key=datasetUtils.natural_keys)
    # ###################################################################
    # tar_path = '/hdd/data/KITTI/testing/image_02'
    # tar_sub_path = os.listdir(tar_path)
    # tar_folder_ind = np.random.randint(0, len(tar_sub_path))
    # # tar_folder_ind = ref_folder_ind
    # img_path = os.path.join(tar_path, tar_sub_path[tar_folder_ind])
    img_path = '/hdd/data/KITTI/train/2011_09_30_drive_0027_sync/image_02/data'
    tar_rgb_list = os.listdir(img_path)
    tar_rgb_list.sort(key=datasetUtils.natural_keys)
    ###################################################################

    img_num = len(tar_rgb_list)

    images = []
    images_small = []

    for i in range(img_num):
        img = cv2.imread(os.path.join(img_path, tar_rgb_list[i]))
        img = img_crop(img)
        img = cv2.resize(img, (224, 224))
        img_small = cv2.resize(img, (28, 28))
        img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_small_Lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2Lab)
        images.append(img_Lab)
        images_small.append(img_small_Lab)

    images = np.array(images)
    images_small = np.array(images_small)

    # call model
    model = tracker.Google_Tracker(consecutiveFrame=CNConfig['consecutiveFrame'])
    model.restoreNetworks(restorePath=weight_restore_path)

    # save
    save_path = img_save_path
    for i in range(img_num - 4):
        batch_np[0, -1, ...] = images[i + 4, ..., 0:1].astype('float32')
        batch = {'InputImages': (batch_np / 255.0 * 2 - 1).astype('float32')}
        out = np.array(model.fit(batch))
        # print out.shape
        ref0 = out[0, 0, ...].reshape([-1, 64])
        ref1 = out[0, 1, ...].reshape([-1, 64])
        ref2 = out[0, 2, ...].reshape([-1, 64])
        tar = out[0, 3, ...].reshape([-1, 64])
        A0 = np.matmul(ref0, np.transpose(tar)).astype('float64')
        A1 = np.matmul(ref1, np.transpose(tar)).astype('float64')
        A2 = np.matmul(ref2, np.transpose(tar)).astype('float64')
        # print A.shape

        A0 = np.exp(A0)
        A0 = A0 / np.sum(A0, axis=0, keepdims=True)
        A1 = np.exp(A1)
        A1 = A1 / np.sum(A1, axis=0, keepdims=True)
        A2 = np.exp(A2)
        A2 = A2 / np.sum(A2, axis=0, keepdims=True)

        # mask_ref = (np.sum(segs[i + 2, ...], -1) != 0).reshape([-1]).astype('float32')
        # cv2.imshow('k1', mask_ref.reshape([28, 28]).astype('uint8') * 100)
        # print mask_ref.shape
        seg_ref0 = Segmented_Ref_Imgs[0, ...].reshape([-1, feature_num])
        seg_ref1 = Segmented_Ref_Imgs[1, ...].reshape([-1, feature_num])
        seg_ref2 = Segmented_Ref_Imgs[2, ...].reshape([-1, feature_num])

        seg_tar_pred0 = np.matmul(np.transpose(A0), seg_ref0).reshape([28, 28, feature_num])
        seg_tar_pred1 = np.matmul(np.transpose(A1), seg_ref1).reshape([28, 28, feature_num])
        seg_tar_pred2 = np.matmul(np.transpose(A2), seg_ref2).reshape([28, 28, feature_num])

        seg_tar_merge = np.stack([seg_tar_pred0, seg_tar_pred1, seg_tar_pred2], -2)
        # merged_quant = Merge_Ind_to_Seg_KITTI(seg_tar_merge)
        merged_quant = (seg_tar_pred0 + seg_tar_pred1 + seg_tar_pred2)

        # seg_tar_pred0 = seg_tar_pred0 / np.sum(seg_tar_pred0, -1, keepdims=True)
        # seg_tar_pred1 = seg_tar_pred1 / np.sum(seg_tar_pred1, -1, keepdims=True)
        # seg_tar_pred2 = seg_tar_pred2 / np.sum(seg_tar_pred2, -1, keepdims=True)

        # tar_RGB0 = Ind_to_Seg_KITTI(seg_tar_pred0)
        # tar_RGB1 = Ind_to_Seg_KITTI(seg_tar_pred1)
        # tar_RGB2 = Ind_to_Seg_KITTI(seg_tar_pred2)

        tar_RGB0 = datasetUtils.quant_to_img(seg_tar_pred0, images_small[i + 3, ...][..., 0])
        tar_RGB1 = datasetUtils.quant_to_img(seg_tar_pred1, images_small[i + 3, ...][..., 0])
        tar_RGB2 = datasetUtils.quant_to_img(seg_tar_pred2, images_small[i + 3, ...][..., 0])
        tar_RGB_merge = datasetUtils.quant_to_img(merged_quant, images_small[i + 3, ...][..., 0])
        # tar_RGB_merge = (tar_RGB0 + tar_RGB1 + tar_RGB2) / 3
        # cv2.imshow('k', np.concatenate([tar_RGB0, tar_RGB1, tar_RGB2], 1))
        # cv2.waitKey(30)
        tar_RGB_avg = tar_RGB_merge
        tar_Lab_up0 = cv2.cvtColor(cv2.resize(tar_RGB0, (224, 224)), cv2.COLOR_BGR2Lab)
        tar_Lab_up1 = cv2.cvtColor(cv2.resize(tar_RGB1, (224, 224)), cv2.COLOR_BGR2Lab)
        tar_Lab_up2 = cv2.cvtColor(cv2.resize(tar_RGB2, (224, 224)), cv2.COLOR_BGR2Lab)
        tar_Lab_up_avg = cv2.cvtColor(cv2.resize(tar_RGB_avg, (224, 224)), cv2.COLOR_BGR2Lab)
        # tar_Lab_up_avg = (tar_Lab_up0 + tar_Lab_up1 + tar_Lab_up2) / 3
        tar_Lab_up0[..., 0] = images[i + 3, ..., 0]
        tar_Lab_up1[..., 0] = images[i + 3, ..., 0]
        tar_Lab_up2[..., 0] = images[i + 3, ..., 0]
        tar_Lab_up_avg[..., 0] = images[i + 3, ..., 0]

        # tar_final = (tar_Lab_up0 + tar_Lab_up1 + tar_Lab_up2) / (CNConfig['consecutiveFrame'] - 1)
        tar_RGB_up0 = cv2.cvtColor(tar_Lab_up0.astype('uint8'), cv2.COLOR_Lab2BGR)
        tar_RGB_up1 = cv2.cvtColor(tar_Lab_up1.astype('uint8'), cv2.COLOR_Lab2BGR)
        tar_RGB_up2 = cv2.cvtColor(tar_Lab_up2.astype('uint8'), cv2.COLOR_Lab2BGR)
        tar_RGB_up_avg = cv2.cvtColor(tar_Lab_up_avg.astype('uint8'), cv2.COLOR_Lab2BGR)

        final = np.concatenate([tar_RGB_up_avg.astype('uint8'),
                                tar_RGB_up0.astype('uint8'),
                                tar_RGB_up1.astype('uint8'),
                                tar_RGB_up2.astype('uint8'),
                                cv2.cvtColor(images[i + 3, ...], cv2.COLOR_Lab2BGR)], 1)
        cv2.imshow('k', final)
        # cv2.imshow('k', np.concatenate([(mask_tar_pred > 0.5).astype('uint8'), mask_ref.reshape([28, 28])], 1))
        cv2.waitKey(40)
        cv2.imwrite(save_path + '/' + str(i).zfill(10) + '.png', final)
    # #     if vid_save:
    # #         out_vid.write(final)
    # #
    # # if vid_save:
    # #     out_vid.release()
    # cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    return


########################################################################################################################
weight_restore_path = '/ssdubuntu/Google_Tracker_Weights/4frames/181110_RandStepSeq_noflip_VIDKITTI/2.lr1e-5'
# weight_restore_path = '/ssdubuntu/Google_Tracker_Weights/4frames/181106_SeqInput_VID/1.lr1e-4'
#####################################################
img_save_path = '/hdd/figures/KITTI_181112/2011_09_30_drive_0027_RandStepSeq'
# img_save_path = '/hdd/figures/KITTI_181112/2011_09_30_drive_0027_Seq'


########################################################################################################################

# Run_KITTI(weight_restore_path=weight_restore_path, img_save_path=img_save_path)

# img_path = '/hdd/temp'
#
# img_list = os.listdir(img_path)
#
# for img in img_list:
#     img_file = img_crop(cv2.imread(os.path.join(img_path, str(img))))
#     cv2.imwrite(os.path.join(img_path, str(img)), img_file)
########################################################################################################################
# img_path = '/hdd/data/KITTI/huhe_semantic_labels/groundtruthImg/2011_09_26_drive_0039_0000000237.png'
# img_org = img_crop(cv2.imread(img_path.replace('groundtruthImg', 'Images')))
# img = img_crop(cv2.imread(img_path))
# img = cv2.GaussianBlur(img, (5, 5), 0.1)
# mask = Seg_to_Ind_KITTI(img)
#
# filter1 = Filter_KITTI_Seg_Img(mask, 'Road')
# filter2 = Filter_KITTI_Seg_Img(mask, 'Car')
# filter3 = Filter_KITTI_Seg_Img(mask, 'Building')
#
# # print filter.shape
# recover = Ind_to_Seg_KITTI(filter1)
# recover2 = Ind_to_Seg_KITTI(filter2)
# recover3 = Ind_to_Seg_KITTI(filter3)
# # recover_tot = Ind_to_Seg_KITTI(mask)
# recover_tot = recover + recover2 + recover3
# # cv2.imshow('k', np.concatenate([recover, recover2], 1))
# # cv2.imshow('org', cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY))
# # # cv2.imshow('k', recover)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite('/hdd/temp/gray.png', cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY))
# cv2.imwrite('/hdd/temp/road.png', recover)
# cv2.imwrite('/hdd/temp/car.png', recover2)
# cv2.imwrite('/hdd/temp/building.png', recover3)
# cv2.imwrite('/hdd/temp/seg.png', recover_tot)
########################################################################################################################
my_path = '/hdd/figures/KITTI_181112/2011_09_30_drive_0027_RandStepSeq/'
google_path = '/hdd/figures/KITTI_181112/2011_09_30_drive_0027_Seq/'

H =224
start_ind = 120
step = 3
final_img = np.zeros([H * step, H * 5, 3], dtype='uint8')
print final_img.shape

for i in range(step):
    index = str(i * 5 + start_ind).zfill(10)
    my_img = cv2.imread(my_path + index + '.png')
    google_img = cv2.imread(google_path + index + '.png')
    # print my_img.shape
    # print google_img.shape

    final_img[(H * i):(H * (i + 1)), :(H * 2), ...] = my_img[:, (H * 2):(H * 4)]
    final_img[(H * i):(H * (i + 1)), (H * 2):] = google_img[:, (H * 2):]

cv2.imwrite('/hdd/figures/kitti_raw.png', final_img)
# cv2.imshow('k', final_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
########################################################################################################################