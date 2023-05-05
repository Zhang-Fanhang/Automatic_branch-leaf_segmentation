
import math
import os
import random

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KDTree


from scipy import spatial



path = "C:/Users/Haitao/Desktop/data-instance/test11/"
path_plane = "C:/Users/Haitao/Desktop/data-instance/test11_plane/"
path_plane_seg = "C:/Users/Haitao/Desktop/data-instance/test11_plane_seg/"


path = "C:/Users/Haitao/Desktop/data-instance/data_set/test_leaf/"
path_plane = "C:/Users/Haitao/Desktop/data-instance/data_set/test_leaf_plane/"
path_plane_seg = "C:/Users/Haitao/Desktop/data-instance/data_set/test_leaf_plane_seg/"
def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2]))

def Hailun(tri1,tri2,tri3):
    a = Distance(tri1,tri2)
    b = Distance(tri2,tri3)
    c = Distance(tri1,tri3)
    p = (a+b+c)/2
    return math.sqrt(p*(p-a)*(p-b)*(p-c))
def Distance(tri1,tri2):
    return math.sqrt((tri1[0]-tri2[0])**2+(tri1[1]-tri2[1])**2+(tri1[2]-tri2[2])**2)


def save_yemai(list1,file,path):
    output1 = list1[0]
    for i in range(1,len(list1)):
        output1 = np.vstack((output1,list1[i]))


    # print(output1)
    np.savetxt("{}".format(os.path.join(path, file.split(".")[0]+"_yemai.txt")), output1, fmt='%f', delimiter=" ")
def save_yekuan(list1,file,path,j):
    output1 = list1[0]
    for i in range(1,len(list1)):
        output1 = np.vstack((output1,list1[i]))


    # print(output1)
    np.savetxt("{}".format(os.path.join(path, file.split(".")[0]+"_yekuan_"+str(j)+".txt")), output1, fmt='%f', delimiter=" ")

def length(list):
    leng = 0
    for i in range(len(list)-1):
        leng = leng+distance(list[i],list[i+1])
    return leng

def curve(pts,p1,p2,kk):
    cloud2 = pts
    cloud1 = p1
    # cloud1 = np.hstack((cloud1[:,0:3],cloud1[:,5:6]))
    print(cloud2.shape)
    X0 = cloud2

    X = cloud1
    d = 0.01
    list = []
    list.append(X)
    cout = 0
    while (not (X==p2).all()):
        base = X
        X = X.reshape(1,3)
        tree = KDTree(X0)
        dist_to_knn, idx_of_knn = tree.query(X=X, k=kk)

        # select = np.array([x for x in range(len(idx_of_knn)) if dist_to_knn[x] <= d])
        select = idx_of_knn[0].tolist()

        #     print(len(select))
        min_d = 10
        num = -1
        flag=0
        while(1):
            min_d=10
            flag=0
            for i in select:
                k = X0[i].reshape(3)
                d1 = distance(k,base)
                if(d1==0):
                    continue;
                d2 = distance(k,p2)
                d = d1+d2
                if(d<min_d):
                    min_d = d
                    num = i

            # if (X0[num].all() in list):

            t = X0[num]
            for j in list:
                if((j==X0[num]).all()):
                    select.remove(num)
                    flag = 1
                    break
            if(flag==1):
                continue
            else:
                list.append(X0[num])
                break;

        X = X0[num]
        X0=np.delete(X0,num,axis=0)
        cout = cout+1
        # print(cout)
    # print(list)
    return list

def planefit(points):
    '''
    todo:   1.generate Auditory canthus line plane
            2.
            --- Ax + By + C = z
                get A B C
    reference:  https://github.com/SeaTimeMoon/PlaneFit/blob/master/Example1.py
    '''
    xs = []
    ys = []
    zs = []
    for i in range(len(points)):
        xs.append(points[i][0])
        ys.append(points[i][1])
        zs.append(points[i][2])
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    return fit[0][0,0], fit[1][0,0],fit[2][0,0]

def equation_plane(p1, p2, pts):
    from numpy.linalg import lstsq  # 解超定方程
    from numpy.linalg import solve  # 解线性方程

    a, b, c = planefit(pts)
    xishu = np.mat([[p1[0], p1[1], p1[2]], [p2[0], p2[1], p2[2]], [a, b, -1]])  # 系数矩阵
    # a = np.mat([[1, 1, 1], [1, 1, 0], [1, 0, 0]])  # 系数矩阵
    changshu = np.mat([1, 1, 0]).T  # 常数项列矩阵
    ans = solve(xishu, changshu)  # 方程组的解
    ans = np.array(ans).reshape(1, -1)

    # a, b, c = planefit(pts)
    # x = random.randint(-140, -120) * 0.01
    # y = random.randint(120, 140) * 0.01
    # z = a * x + b * y + c
    normal_in_plane = ans[0]
    return normal_in_plane

def ProjectPointsToPlane(point_in_plane, normal_in_plane, points):
    v = points - point_in_plane
    normalized_normal_in_plane = normal_in_plane / np.linalg.norm(normal_in_plane)
    dist = v.dot(normalized_normal_in_plane)
    projected_points = (points - dist * normalized_normal_in_plane)

    return projected_points

def save_projectionplane(normal,pts,str):
    x = np.random.randint(np.min(pts[:, 0]) * 500 - 30, np.max(pts[:, 0]) * 500 + 30, size=(50000, 1)) * 0.002
    y = np.random.randint(np.min(pts[:, 1]) * 500 - 30, np.max(pts[:, 1]) * 500 + 30, size=(50000, 1)) * 0.002
    z = (1-(normal[0]*x+normal[1]*y))/normal[2]
    yes = np.array([255]*50000).reshape(-1,1)
    no = np.array([0]*50000).reshape(-1,1)
    if(str=="leaf"):
        np.savetxt("{}".format(os.path.join(path_plane, file.split(".")[0] + "_"+str+"_ProjectionPlane.txt")), np.hstack((x, y, z,yes,no,no)), fmt='%f', delimiter=" ")
    if (str == "yemai"):
        np.savetxt("{}".format(os.path.join(path_plane, file.split(".")[0] + "_" + str + "_ProjectionPlane.txt")),np.hstack((x, y, z,no,yes,no)), fmt='%f', delimiter=" ")
    if (str == "yekuan"):
        np.savetxt("{}".format(os.path.join(path_plane, file.split(".")[0] + "_" + str + "_ProjectionPlane.txt")),np.hstack((x, y, z,no,no,yes)), fmt='%f', delimiter=" ")
def PlaneSeg(normal,pts):
    list1 = []
    list2 = []
    for point in pts:
        if(normal[0]*point[0]+normal[1]*point[1]+normal[2]*point[2]>=1):
            list1.append(point)
        else:
            list2.append(point)
    pts1 = np.array(list1)
    pts2 = np.array(list2)
    np.savetxt("{}".format(os.path.join(path_plane_seg, file.split(".")[0] + "_Plane_seg1.txt")),
               pts1, fmt='%f', delimiter=" ")
    np.savetxt("{}".format(os.path.join(path_plane_seg, file.split(".")[0] + "_Plane_seg2.txt")),
               pts2, fmt='%f', delimiter=" ")
    list = []
    a,b,c = planefit(pts1)
    normal_in_plane = np.array([a,b,-1])
    x = pts1[0,0]
    y = pts1[0,1]
    z = a * x + b * y + c
    point_on_plane = np.array([x,y,z])
    for i in range(len(pts1)):
        # a = ProjectPointsToPlane(np.array([x,y,z]), normal_in_plane, pts[i])
        aa = ProjectPointsToPlane(point_on_plane, normal_in_plane, pts1[i])
        list.append(aa)
    np.savetxt("{}".format(os.path.join(path_plane_seg, file.split(".")[0] + "Plane_projection1.txt")), np.array(list), fmt='%f',delimiter=" ")

    list = []
    a,b,c = planefit(pts2)
    normal_in_plane = np.array([a,b,-1])
    x = pts2[0,0]
    y = pts2[0,1]
    z = a * x + b * y + c
    point_on_plane = np.array([x,y,z])
    for i in range(len(pts2)):
        # a = ProjectPointsToPlane(np.array([x,y,z]), normal_in_plane, pts[i])
        aa = ProjectPointsToPlane(point_on_plane, normal_in_plane, pts2[i])
        list.append(aa)
    np.savetxt("{}".format(os.path.join(path_plane_seg, file.split(".")[0] + "Plane_projection2.txt")), np.array(list), fmt='%f',delimiter=" ")



leng1 = []
leng2 = []
for file in os.listdir(path):
    print(file)
    print(file.split(".")[0])
    pts = np.loadtxt(path+file, dtype=np.float32, delimiter=' ')
    pts = pts[:,:3]
    # two points which are fruthest apart will occur as vertices of the convex hull
    candidates = pts[spatial.ConvexHull(pts).vertices]
    # get distances between each pair of candidate points
    dist_mat = spatial.distance_matrix(candidates, candidates)
    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
#     print(candidates[i], candidates[j])
    p1 = candidates[i]
    p2 = candidates[j]
    print(p1,p2)
    list1 = []
    list1 = curve(pts,p1,p2,5)

    # save_yemai(list1, file, path)
    leng1.append(length(list1)*100)

    a, b, c = planefit(pts)
    save_projectionplane(np.array([-a/c,-b/c,1/c]),pts,"leaf")
    list = []

    points = np.array(list1)
    # point = np.vstack((np.array([points[0,0],points[0,1],0]),points[0],points[len(points)-1]))
    # a, b, c = equation_plane(points[0, 0], points[0, 1] ,0, points[0, 0], points[0, 1], points[0, 2], points[len(points) - 1, 0], points[len(points) - 1, 1],points[len(points) - 1, 2])


    # a, b, c = planefit(pts)
    # x = random.randint(-140, -120) * 0.01
    # y = random.randint(120, 140) * 0.01
    # z = a * x + b * y + c
    normal_in_plane = equation_plane(p1, p2, pts)
    save_projectionplane(normal_in_plane,pts,"yemai")
    PlaneSeg(normal_in_plane, pts)



    for i in range(len(list1)):
        # a = ProjectPointsToPlane(np.array([x,y,z]), normal_in_plane, pts[i])
        aa = ProjectPointsToPlane(points[0], normal_in_plane, points[i])
        list.append(aa)

    np.savetxt("{}".format(os.path.join(path, file.split(".")[0] + "_yemai_origin.txt")), np.array(list1), fmt='%f', delimiter=" ")
    np.savetxt("{}".format(os.path.join(path, file.split(".")[0] + "_yemai.txt")), np.array(list), fmt='%f', delimiter=" ")
    # continue






    # normal_in_plane = np.array([1, 1, a + b])
    #
    # list = []
    # # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    # # output = pts[0]
    # # print(len(pts))
    # pts = np.loadtxt(path + '008_yekuan_0.txt', dtype=np.float32, delimiter=' ')
    # for i in range(len(pts)):
    #     # a = ProjectPointsToPlane(np.array([x,y,z]), normal_in_plane, pts[i])
    #     a = ProjectPointsToPlane(pts[1], normal_in_plane, pts[i])
    #     list.append(a)
    #
    #     print(a)
    #
    # output = np.array(list)



    # list1 = curve(pts, p1, p2, 5)
    # 半叶宽



    area_max = 0
    min_d = 100
    num2 = -1

    for i in range(len(pts)):
        area = Hailun(pts[i],p1,p2)
        if(area>area_max):
            area_max = area
            num2 = i
    # list = list1

    for j in range(1):
        for i in range(len(list)):
            d = distance(pts[num2],list[i])
            if(d<min_d):
                min_d = d
                min_point = i
        pts = np.vstack((pts, np.array(list[min_point])))
        list2 = curve(pts, pts[num2], list[min_point],5)
        p = pts[num2]
        q = list[min_point]
        list = []
        list1 =list2

        # a, b, c = planefit(pts)
        # x = random.randint(-140, -120) * 0.01
        # y = random.randint(120, 140) * 0.01
        # z = a * x + b * y + c
        normal_in_plane = equation_plane(p, q, pts)
        save_projectionplane(normal_in_plane, pts,"yekuan")
        pts = np.array(list1)
        for i in range(len(list1)):
            # a = ProjectPointsToPlane(np.array([x,y,z]), normal_in_plane, pts[i])
            bb = ProjectPointsToPlane(pts[len(list1)-1], normal_in_plane, pts[i])
            list.append(bb)
        np.savetxt("{}".format(os.path.join(path, file.split(".")[0] + "_yekuan_origin.txt")), np.array(list1), fmt='%f',delimiter=" ")
        np.savetxt("{}".format(os.path.join(path, file.split(".")[0] + "_yekuan.txt")), np.array(list), fmt='%f',delimiter=" ")

        # save_yekuan(list2, file, path,j)
        leng2.append(length(list2)*200)
        # del list[min_point]




    # list2 = curve(pts, pts[num2], min_point)






    # output1 = list1[0]
    # for i in range(1,len(list1)):
    #     output1 = np.vstack((output1,list1[i]))
    #
    #
    # # print(output1)
    # np.savetxt("{}".format(os.path.join(path, file.split(".")[0]+"_yemai.txt")), output1, fmt='%f', delimiter=" ")
    #
    # output2 = list2[0]
    # for i in range(1,len(list2)):
    #     output2 = np.vstack((output2,list2[i]))
    #
    #
    # # print(output2)
    # np.savetxt("{}".format(os.path.join(path, file.split(".")[0]+"_yekuan.txt")), output2, fmt='%f', delimiter=" ")
print(leng1)
print(leng2)
np.savetxt("{}".format(os.path.join(path, "leng.txt")), np.hstack((np.array(leng1).reshape(-1,1),np.array(leng2).reshape(-1,1))), fmt='%f', delimiter="\t")


