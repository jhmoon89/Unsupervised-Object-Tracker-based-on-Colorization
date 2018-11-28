import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import imgaug as ia
from imgaug import augmenters as iaa
from scipy.spatial import distance

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def resize_Lab_image(img, size):
    RGB_img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    RGB_resize = cv2.resize(RGB_img, size)
    return cv2.cvtColor(RGB_resize, cv2.COLOR_BGR2Lab)

# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noisy(image, noise_typ):
    if noise_typ == "gaussian":
        # np.array([103.939, 116.779, 123.68])
        mean = 0
        var = 0.01 * 255.0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "salt&pepper":
        s_vs_p = 0.5
        amount = 0.05
        out = image.copy()

        # print image.size
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, np.max((i - 1, 1)), int(num_salt)) for i in image.shape]
        out[coords] = 255.0

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, np.max((i - 1, 1)), int(num_pepper)) for i in image.shape]
        out[coords] = 0.0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(np.abs(image) * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        gauss = np.random.randn(*image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = image + image * 0.1 * gauss
        return noisy
    else:
        return image.copy()


def imageAugmentation(inputImages):
    noiseTypeList = ['gaussian', 'salt&pepper', 'poisson', 'speckle']
    random.shuffle(noiseTypeList)
    select = np.random.randint(0, 2, len(noiseTypeList))
    for i in range(len(noiseTypeList)):
        if select[i] == 1:
            inputImages = noisy(image=inputImages, noise_typ=noiseTypeList[i])
    return inputImages


def cvt_image_to_z(img, gamut): # input image : Lab channel (batch_num, frame_num, H, W, 3)
    gamut_range = np.argwhere(gamut != 0)
    color_bin_num = len(gamut_range)
    # print color_bin_num

    # cv2.imshow('img', cv2.resize(cv2.cvtColor(img, cv2.COLOR_Lab2BGR), img_size))

    # ab channel to bin
    img = (img[..., 1:] / 8).astype('uint8')
    img_size = np.shape(img)[-3:-1]

    tic = time.time()
    img_tile = np.tile(np.expand_dims(img, -2), [1, 1, 1, 1, color_bin_num, 1]).astype('float32')
    gamut_range_tile = (np.tile(
        np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(gamut_range, 0), 0), 0), 0),
        [np.shape(img)[0], np.shape(img)[1], img_size[0], img_size[1], 1, 1])).astype('float32')
    toc = time.time()
    # print ("tile gamut: "), toc-tic

    tic = time.time()
    sigma = 0.5
    dist = np.exp(-1 * np.sum(np.square(img_tile - gamut_range_tile), -1) / (2 * sigma * sigma)) / \
           (np.sqrt(2 * np.pi * sigma * sigma))
    dist_sorted = np.sort(dist, -1)
    toc = time.time()
    # print ("gaussian: "), toc-tic

    # smoothing by Gaussian Filter
    tic = time.time()
    mask = (dist >= np.tile(np.expand_dims(dist_sorted[..., -5], -1), [1, 1, color_bin_num])).astype('float32')
    toc = time.time()
    # print ("mask: "), toc-tic
    tic = time.time()
    dist_truncated = dist * mask
    toc = time.time()
    # print ("matrix mul: "), toc-tic
    tic = time.time()
    value_sum = np.tile(np.expand_dims(np.sum(dist_truncated, -1), -1), [1, 1, color_bin_num])
    dist_truncated = dist_truncated / (value_sum + 1e-9)
    toc = time.time()
    # print ("final: "), toc-tic

    return dist_truncated

    # print np.sort(dist_truncated, -1)[0][0][-5:]
    # print np.sort(dist_truncated2, -1)[0][0][-5:]

    # ind = np.argsort(dist, -1)[4]
    # th = np.take(dist, ind)


    # cv2.imshow('class', ind)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ind = np.argsort(dist, -1)[..., 5:]
    # ind2 = np.unravel_index(ind, np.shape(dist))
    # dist[ind2] = 0
    # # print dist
    # dist[dist < 1e-9] = 0
    # print np.argwhere(dist[100, 100] != 0)

def cvt_z_to_image(Z, gamut=np.load("/ssdubuntu/color/gamut.npy"), T=0.38): # H function
    gamut_range = np.argwhere(gamut != 0)

    Z_shape = list(Z.shape)
    # print Z_shape
    # Interporlated_Z = np.exp(np.log(Z.astype('float32') + 1e-9) / (T + 1e-9)).astype('uint8').astype('float32')
    Interporlated_Z = np.exp(np.log(Z.astype('float32') + 1e-9) / (T + 1e-9))
    Interporlated_Z = Interporlated_Z / (np.sum(Interporlated_Z, axis=-1, keepdims=True) + 1e-9)
    # Interporlated_Z = Z - np.exp(T)
    # print np.shape(Interporlated_Z)
    a_list = np.expand_dims(np.expand_dims(gamut_range[:, 0] * 8 + 4, 0), 0).astype('float32')
    b_list = np.expand_dims(np.expand_dims(gamut_range[:, 1] * 8 + 4, 0), 0).astype('float32')

    # print np.shape(a_list)
    Height = np.shape(Interporlated_Z)[0]
    Width = np.shape(Interporlated_Z)[1]

    output_a = np.sum(Interporlated_Z * np.tile(a_list, Z_shape[:-1] + [1]), -1)
    output_b = np.sum(Interporlated_Z * np.tile(b_list, Z_shape[:-1] + [1]), -1)

    return (np.stack([output_a, output_b], -1)).astype('uint8')

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 255

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

#'''https://github.com/aleju/imgaug'''
def imgAug(inputImage, crop=True, flip=True, gaussianBlur=True, channelInvert=True, brightness=True, hueSat=True):
    augList = []
    if crop:
        augList += [iaa.Crop(px=(0, 16))]  # crop images from each side by 0 to 16px (randomly chosen)
    if flip:
        augList += [iaa.Fliplr(0.5)]  # horizontally flip 50% of the images
    if gaussianBlur:
        augList += [iaa.GaussianBlur(sigma=(0, 3.0))]  # blur images with a sigma of 0 to 3.0
    if channelInvert:
        augList += [iaa.Invert(0.05, per_channel=True)]  # invert color channels
    if brightness:
        augList += [iaa.Add((-10, 10), per_channel=0.5)]  # change brightness of images (by -10 to 10 of original value)
    if hueSat:
        augList += [iaa.AddToHueAndSaturation((-20, 20))]  # change hue and saturation
    seq = iaa.Sequential(augList)
    # seq = iaa.Sequential([
    #     iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    #     # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    #     iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
    #     iaa.Invert(0.05, per_channel=True),  # invert color channels
    #     iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
    #     iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    # ])

    image_aug = seq.augment_image(inputImage)
    return image_aug

def img_to_quant(img, centers=np.load('/ssdubuntu/color/center_256_MiniBatchKmeans_ab.npy')):
    # input : BGR
    img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    H, W, _ = img_Lab.shape
    img_ab = img_Lab[..., 1:]
    img_ab = img_ab.reshape(-1, 2)

    dist = distance.cdist(img_ab, centers)
    ind = np.argmin(dist, -1).reshape(H, W)
    return (np.arange(16) == ind[..., None]).astype('int')
    # min_cls = centers[np.array(ind), :].astype('uint8')
    # # print min_cls.shape
    # # print img_Lab.shape
    # # print min_cls.shape
    # quantitized_Lab = np.concatenate([img_Lab[..., 0:1], min_cls], -1)
    # return cv2.cvtColor(quantitized_Lab, cv2.COLOR_Lab2BGR)

def quant_to_img(quant_img, L_img, T=0.5, centers = np.load('/ssdubuntu/color/center_256_MiniBatchKmeans_ab.npy')):
    # output : BGR
    assert np.array_equal(quant_img.shape[:2], L_img.shape[:2])
    H, W, _ = quant_img.shape
    center_tile = np.tile(centers[None, None, ...], [H, W, 1, 1])
    a_tile = center_tile[..., 0].astype('float32')
    b_tile = center_tile[..., 1].astype('float32')
    quant_temp = quant_img.astype('float32')
    quant_interpolated = np.exp(np.log(quant_temp + 1e-9)) / (T + 1e-9)
    quant_interpolated = quant_interpolated / (np.sum(quant_interpolated, axis=-1, keepdims=True) + 1e-9)

    output_a = np.sum(quant_interpolated * a_tile, -1).astype('uint8')
    output_b = np.sum(quant_interpolated * b_tile, -1).astype('uint8')

    # ind = np.argmax(quant_img, -1)
    # center_img = centers[np.array(ind), :].astype('uint8')
    if len(L_img.shape)==2:
        L_img = L_img[..., None]
    img_Lab = np.concatenate([L_img, output_a[..., None], output_b[..., None]], -1)
    return cv2.cvtColor(img_Lab, cv2.COLOR_Lab2BGR)

########### Rebalance Factaor ############
# gamut = np.load("/ssdubuntu/color/gamut.npy")
# gamut_normalize = gamut / np.sum(gamut)
# ind_mask = np.where(gamut_normalize != 0)
# Q = np.shape(ind_mask)[1]
# gamut_blur = cv2.GaussianBlur(gamut_normalize, (5, 5), 5)
# gamut_smooth = np.zeros_like(gamut)
# gamut_smooth[ind_mask] = 1.0 / (0.5 * (gamut_blur[ind_mask] + 1 / float(Q)))
# gamut_smooth = gamut_smooth / np.sum(gamut_smooth)
# weight_list = gamut_smooth[ind_mask]
# np.save("/ssdubuntu/color/rebalance_weight.npy", weight_list)
###########################################


# gamut = np.load("/ssdubuntu/color/gamut.npy")
# # print len(np.argwhere(gamut != 0))
# # weight_list = np.load("/ssdubuntu/color/rebalance_weight.npy")
# # print np.shape(weight_list)
# #
# img_name = "/ssdubuntu/data/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00001003/000000.JPEG"
# img = cv2.imread(img_name)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
#
# cv2.imshow('k', cv2.cvtColor(resize_Lab_image(img, (56, 56)), cv2.COLOR_Lab2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img = cv2.cvtColor(cv2.resize(cv2.imread(img_name, cv2.IMREAD_COLOR), (224, 224)), cv2.COLOR_BGR2Lab)
# # data = cvt_image_to_z(img, gamut)
# bin_path = img_name.replace("data/ILSVRC/Data/VID/train", "Bin_Data").replace("JPEG", "npy")
# data = np.load(bin_path)
# data = (np.arange(374) == data[..., None]).astype('float')
#
# data_smooth = cv2.GaussianBlur(data, (5, 5), 0.1)
#
# print np.array_equal(data_smooth, data)
#
# # max_cls = (np.argmax(data, -1) / np.max(data) * 255).astype('uint8')
# z_test = cvt_z_to_image(data_smooth, gamut, 0.5)
# # print img.shape, z_test.shape
# final_img = np.stack([img[..., 0], z_test[..., 0], z_test[..., 1]], -1)
#
# # print np.shape(final_img)
# # print np.shape(z_test)
# # print img - final_img
#
# cv2.imshow('org', cv2.cvtColor(img, cv2.COLOR_Lab2BGR))
# cv2.imshow('image', cv2.cvtColor(final_img, cv2.COLOR_Lab2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
