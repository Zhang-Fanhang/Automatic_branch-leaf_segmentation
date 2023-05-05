import os
import random
import json
path2 = 'shape_data'

list=[]
list1=[]
list2 = []
path = '/home/nau/pointnet22/Pointnet2/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
# path = '/home/nau/pointnet++/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
path3 =os.path.join(path,"shuffled_train_file_list.json")
path4 =os.path.join(path,"shuffled_val_file_list.json")
print(path3)
for file in os.listdir(path):
    # print(file1)
    # for file in os.listdir(os.path.join(path,file1)):
    #     print(file)
    for new_file in os.listdir(os.path.join(path,file)):
        if os.path.splitext(new_file)[1] == '.txt':
            #print(new_file)
            #print(os.path.join(path,os.path.splitext(new_file)[0]))
            list.append(os.path.join(os.path.join(path2,file),os.path.splitext(new_file)[0]))

random.shuffle(list)
# if len(list)>5000:
#     list = list[:5000]
num = len(list)
print(num)

list1 = list[:40]
list2 = list[40:]
with open(path3,'w') as file_obj1:
    json.dump(list2,file_obj1)
with open(path4,'w') as file_obj2:
    json.dump(list1,file_obj2)

#print(list)
