from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        # fn: [class_idx, point_set_file, segmentation_file]
        cls = self.classes[fn[0]] # which class does this point set belong to, e.g. "Airplane", "Pisto", ...
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        # print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample 'num_points' points from original point set
        point_set = point_set[choice, :]
        # normalization to center unit sphere
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale
        # data augmentation
        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice] # segmentation tags corresponding to each point
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            # for classification network used, but normally,
            # ModelNet40 dataset can handle the work, so the
            # ShapeNet would be better to train segmentation
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=1024,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = os.path.realpath(root)
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):

        '''
        function overloading of operator [] for dataset ModelNet40

        params:
            - index:    class index of label(class indexes are self defined)
        
        return:
            - points:   xyz list of the points
            - cls:      catagory of the points
        '''
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # move to center(0, 0, 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale to unit sphere

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

class S3DISNetDataset(data.Dataset):
    def __init__(
            self,
            data_root='trainval_fullarea', 
            split="train",
            num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None
        ):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        areas = sorted(os.listdir(data_root))
        test_area = "Area_%d"%test_area
        rooms_split = []
        for area in areas:
            if split == "train" and area == test_area:
                continue
            if split == "test"  and area != test_area:
                continue
            rooms_split.extend([(area, room) for room in sorted(os.listdir(os.path.join(data_root, area))) if room.endswith(".ply")])

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(14) # initialize weights of each class

        for area, room in rooms_split:
            room_path = os.path.join(data_root, area, room)
            plydata = PlyData.read(room_path)
            room_data = np.vstack(
                [
                    plydata["vertex"]['x'], 
                    plydata["vertex"]['y'], 
                    plydata["vertex"]['z'], 
                    plydata["vertex"]["red"], 
                    plydata["vertex"]["green"], 
                    plydata["vertex"]["blue"], 
                    plydata["vertex"]["label"]
                ]
            ).T
            points, labels = room_data[:, 0:6], room_data[:, 6]  # [x,y,z,r,g,b], [l]
            tmp, _ = np.histogram(labels, range(0, 15)) # labels in this room, possible classes are 0-13, so we use range(14)
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3] # min/max value from 3 dims [x,y,z]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size) # 每个房间的总点数 比如训练时为204个房间 list:204

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0) #（倒数*最大值）^(1/3)
        print(self.labelweights) # 样本中所含各类标签权重
        sample_prob = num_point_all / np.sum(num_point_all) # 每个房间的点数/所有房间点总数
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point) # 迭代次数：所有房间总点数*采样率（1代表全采）/4096 总共要迭代这么多次才能把所有房间采完
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter))) # 不断往后面添加元素，添加元素为索引0~203，每次循环添加个数为：迭代次数*该索引房间的采样率。元素（索引0~204）的个数就是对应索引的房间采样的次数
        self.room_idxs = np.array(room_idxs) # 每次采样都是按照room_idxs中的元素选房间来采,按比例分配的思想，不是随机采或者大家一样
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split)) # 样本数

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx] # 该索引房间内的所有点 n * 6
        labels = self.room_labels[room_idx] # 该索引房间内所有点的标签 n
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3] # 随机选择一个点坐标作为block中心
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0] # block的区域范围：三个坐标的最小值
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0] # block的区域范围：三个坐标的最大值
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]#在block范围内的点的索引
            if point_idxs.size > 1024: # 直到采集的点超过1024个，否则就再随机采
                break
        
        if point_idxs.size >= self.num_point: # 如果采的点比4096多就再选4096，少就复制一些点凑满4096
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False) # 就在这些点里面再随机选4096个
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True) # replace:True表示可以取相同数字，False表示不可以取相同数字

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0] #论文中提到的输入为9维，最后三维为该点在房间中的相对位置
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0] # 对x,y去中心化,但z没有
        selected_points[:, 1] = selected_points[:, 1] - center[1] # 对x,y去中心化,但z没有
        selected_points[:, 3:6] /= 255.0 # RGB归一化
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels # 返回4096个点和对应的标签
 
    def __len__(self):
        return len(self.room_idxs)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

