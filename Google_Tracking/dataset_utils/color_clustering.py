import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import imgaug as ia
from imgaug import augmenters as iaa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import distance
import datasetUtils

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def Get_Cluster():
    imageSize = (256, 256)
    data_path = "/ssdubuntu/data/ILSVRC/Data/VID/train"
    sub_path_list = os.listdir(data_path)
    final_path_list = []
    Data_List = pd.DataFrame()
    for sub_path in sub_path_list:
        sub_sub_list = os.listdir(os.path.join(data_path, sub_path))
        # iter_num = len(sub_sub_list)
        iter_num = 500
        for i in range(iter_num):
            start = time.time()
            sub_sub = sub_sub_list[i]
            final_path = os.path.join(data_path, sub_path, sub_sub)
            img = cv2.imread(final_path+"/000000.JPEG", cv2.IMREAD_COLOR)
            img = cv2.resize(img, imageSize)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            # plt.plot(np.reshape(img[:, :, 0], [-1]), np.reshape(img[:, :, 1], [-1]), '.')
            points_ab = np.stack([
                                  np.reshape(img[:, :, 1], -1),
                                  np.reshape(img[:, :, 2], -1)], -1)
            # one_hot = np.zeros([np.shape(img[..., 0]), 16])
            data = pd.DataFrame(points_ab)
            Data_List = Data_List.append(data)
            # kmeans_result = KMeans(n_clusters=8, random_state=0).fit(points_ab)
            # Center_Data = pd.DataFrame(kmeans_result.cluster_centers_)
            # Center_Data_List = Center_Data_List.append(Center_Data)
            final_path_list.append(final_path)
            end = time.time()
            print end-start, i

    kmeans = MiniBatchKMeans(16)
    kmeans.fit(Data_List)
    centroids = kmeans.cluster_centers_
    new_colors = kmeans.cluster_centers_[kmeans.predict(Data_List)]
    # print new_colors
    print centroids

    data_save_path = "/ssdubuntu/color/data_256_MiniBatchKmeans_ab.csv"
    Data_List.to_csv(data_save_path)
    center_path = '/ssdubuntu/color/center_256_MiniBatchKmeans_ab.csv'
    center_np_path = '/ssdubuntu/color/center_256_MiniBatchKmeans_ab.npy'
    np.save(center_np_path, centroids)
    return

def Get_Dist(centers=np.load('/ssdubuntu/color/center_256_MiniBatchKmeans_ab.npy')):
    count = np.zeros(centers.shape[0], dtype=np.float32)
    imageSize = (256, 256)
    data_path = "/ssdubuntu/data/ILSVRC/Data/VID/train"
    sub_path_list = os.listdir(data_path)
    for sub_path in sub_path_list:
        sub_sub_list = os.listdir(os.path.join(data_path, sub_path))
        # iter_num = len(sub_sub_list)
        iter_num = 500
        for i in range(iter_num):
            sub_sub = sub_sub_list[i]
            final_path = os.path.join(data_path, sub_path, sub_sub)
            img = cv2.imread(final_path+"/000000.JPEG", cv2.IMREAD_COLOR)
            img = cv2.resize(img, imageSize)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            quant_img = datasetUtils.img_to_quant(img, centers=centers)
            count = count + np.sum(quant_img.reshape(-1, 16), 0) / (256.0 * 256.0)
            # print count
            print i
    count = count / np.sum(count, keepdims=True)
    np.save('/ssdubuntu/color/distribution.npy', count)

# def Cluster_Image(img, centers):


########################################################################################################################
data_path = "/ssdubuntu/color/data_256_MiniBatchKmeans_ab.csv"
center_path = '/ssdubuntu/color/center_256_MiniBatchKmeans_ab.csv'
center_Lab_path = '/ssdubuntu/color/center_256_MiniBatchKmeans_Lab.npy'
center_Lab_path2 = '/ssdubuntu/color/center_256_MiniBatchKmeans_ab.npy'

dist = np.load('/ssdubuntu/color/distribution.npy')
print dist
fig = plt.figure()
plt.bar(np.arange(16), dist)
plt.show()

# Get_Dist()
# Get_Cluster()
# Load_Cluster(data_path)
# Load_Cluster(center_path)

# center = pd.read_csv(center_path)
# print center


# img_name = '/home/jihoon/Pictures/detection.png'
# img = cv2.imread(img_name)
# img = cv2.resize(img, (256, 256))
# img_L = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[..., 0]
# quant = datasetUtils.img_to_quant(img)
# recovered_img = datasetUtils.quant_to_img(quant, img_L, T=0.5)
#
# cv2.imshow('k', np.concatenate([img, recovered_img], 1))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# centers = np.load(center_Lab_path2)
# # print centers
#
# img_name = '/home/jihoon/Pictures/detection.png'
# img = cv2.imread(img_name)
# img_size = (256, 256)
# img = cv2.resize(img, img_size)
# img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# img_ab = img_Lab[..., 1:]
# img_ab = img_ab.reshape(-1, 2)
#
# dist = distance.cdist(img_ab, centers)
# ind = np.argmin(dist, -1).reshape(img_size)
# min_cls = centers[np.array(ind), :].astype('uint8')
# # print min_cls.shape
#
# quantitized_Lab = np.concatenate([img_Lab[..., 0:1], min_cls], -1)
#
# # print dist.shape
# # print dist
#
# # print np.unique(img_ab, 1)
# # print np.unique(min_cls, 1)
#
# cv2.imshow('k', img)
# cv2.imshow('k1', cv2.cvtColor(quantitized_Lab, cv2.COLOR_Lab2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2])
# plt.plot(centers[:, 0], centers[:, 1], '.')
# plt.show()