import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
import sklearn.cluster as sc
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc,centroid,m
def ap(data,index):
    model = AffinityPropagation(damping=0.75, max_iter=100, convergence_iter=30,preference=-2).fit(data)
#     model = AffinityPropagation(damping=0.75, max_iter=100, convergence_iter=30,preference=-0.05).fit(data)
    cluster_centers_indices = model.cluster_centers_indices_
    y_pred = model.labels_
    new_data = np.hstack((data, y_pred[:, np.newaxis]))
    np.savetxt(path1+index+"_AP.txt", new_data, fmt='%f', delimiter=" ")
    n_clusters_ = len(set(y_pred))
    print(n_clusters_)
    return n_clusters_
def dbscan(data,index):
    db = DBSCAN(eps=0.1, min_samples=30).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    new_data = np.hstack((data, labels[:, np.newaxis]))
    np.savetxt(path1+index+"_DBSCAN.txt", new_data, fmt='%f', delimiter=" ")
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    return n_clusters_
def ms(data,index):
    bw = sc.estimate_bandwidth(data, n_samples=len(data), quantile=0.1)
#     bw = sc.estimate_bandwidth(data, n_samples=len(data), quantile=0.07)
    print(bw)
    bw = 0.045
    model = sc.MeanShift(bandwidth=bw, bin_seeding=True,cluster_all=False)
    model.fit(Xn)  # 完成聚类
    pred_y = model.predict(Xn)  # 预测点在哪个聚类中
    centers = model.cluster_centers_
#     print(centers)
#     np.savetxt(path1+index+str(bw)+"_MS_centers.txt", centers, fmt='%f', delimiter=" ")
    new_data = np.hstack((data, pred_y[:, np.newaxis]))
    np.savetxt(path1+index+"_"+str(bw)[3:5]+"_MS.txt", new_data, fmt='%f', delimiter=" ")
    n_clusters_ = len(set(pred_y))
    print(n_clusters_)
    return n_clusters_

'''
  sklearn.cluster.AffinityPropagation函数
  主要参数: damping 阻尼系数，取值[0.5,1)  convergence_iter:比较多少次聚类中心不变后停止迭代，默认15
          max_iter:最大迭代次数  preference:参考度（即p值）
  主要属性： cluster_centers_indices_  存放聚类中心数组
          labels_ 存放每个点的分类的数组
          n_iter_ 迭代次数
'''
# path = "C:/Users/Haitao/Desktop/data-instance/data_set/predict/"
path1 = "C:/Users/Haitao/Desktop/data-instance/data_set/ins_seg/seg/"
path = "C:/Users/Haitao/Desktop/data-instance/data_set/ins_seg/branches/"
path2 = "C:/Users/Haitao/Desktop/data-instance/data_set/ins_seg/stem/"
l1 = []
l2 = []
l3 = []
for file in os.listdir(path):
    index = file.split(".")[0]
    print(file)
    data = np.loadtxt(path+file,dtype=np.float32,delimiter=" ")
    x = data.shape[1]
#     print(x)
#     print(data[:,x-1:x])
    data = np.hstack((data[:,0:3],data[:,x-1:x]))
#     print(data.shape)
    #取叶子点云
    data = data[np.argsort(-data[:,3])]
    for i in range(data.shape[0]):
        if data[i,3] != 1:
            data_leaf = data[:i,:]
            data_stem = data[i:2048,:]
            np.savetxt(path2+index+"_MS_stem.txt", data_stem, fmt='%f', delimiter=" ")
            break
    #Xn为叶子点云
    Xn = data_leaf
#     Xn = data
#     Xn,centroid,m=pc_normalize(data)

    n3 = ms(Xn,index)