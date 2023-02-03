from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# "STN" is the abbreviation for "Spatial Transform Network"
class STN3d(nn.Module):
    # 空间转换网络只需要3x3，而不需要4x4的齐次矩阵是因为数据
    # 样本已经做了归一化和中心化处理，所有样本坐标中心都是原
    # 点，只剩下旋转变换需要学习
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        # 一维卷积网络不是很常见，可以类比二维卷积网络，当卷积的通道深度大于1时
        # 卷积核从二维“面”变为三维的“体”，那么一维卷积的通道数大于1时，可以理解
        # 为一条线，往后拓展扫过变成面，一维的线在高维卷积核的作用下，变成了面。
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9) # 旋转矩阵的九个分量，以一维向量的形式表达方便构建全连接层，进行矩阵乘法的时候再torch.view()变换shape进行矩阵乘法
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    # 原论文：mini-network "T-Net" resembles the total net work
    def forward(self, x):
        # 通过输入一个样本点云'x'，得到该点云的预测空间变换矩阵'trans': shape([1, 9])
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0] # torch.max will return 2 values, max values and indices of these max values
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # 向仿射变换矩阵添加单位阵的目的是，使得变换符合正交矩阵，因为刚体旋转
        # 矩阵一定是正交阵，单一阵就相当于什么变换都不做的旋转矩阵，让空间变换
        # 网络在标准单一阵上学习偏差量，保证变换矩阵是由正交阵学习而来。这种思
        # 想有点类似于残差网络？
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        # 使用一维卷积而非全连接层的原因是，所有的点输入组成一个样本
        # 而我们不希望顺序的对每个点使用这一套全连接层，卷积层能很好
        # 的解决这个问题，尺寸为1的卷积核能并行的将每个点云点的坐标
        # 输入全连接映射到高维特征输出
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # repeat(dim0[, dim1, ...]), repeat the data in the corresponding dimension 'dim' times
        # repeat(1) for keeping original no change
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, in_channel=3, ft_channel=64, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.in_channel = in_channel
        self.ft_channel = ft_channel
        # first pass through a 3d spatial transform network
        self.stn = STN3d(channel=in_channel)
        # then, pass through a kd feature transform network
        self.conv1 = torch.nn.Conv1d(in_channel, ft_channel, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=ft_channel)

    def forward(self, x):
        n_pts = x.size()[2]                 # [batch_size, channel_size=3, sample_points_num]
        trans = self.stn(x)
        x = x.transpose(2, 1)               # [batch_size, sample_points_num, channel_size=3]
        if self.in_channel > 3:
            other_feat = x[:, :, 3:]        # features other than (x,y,z) starts from index=3
            x = x[:, :, :3]
        x = torch.bmm(x, trans)             # batch multiplication between matrix, not accept other data type(e.g. vector)
        if self.in_channel > 3:
            # remember to concatenate back after xyz transformation
            x = torch.cat([x, other_feat], dim=2)
        x = x.transpose(2, 1)               # [batch_size, channel_size=3, sample_points_num]

        x = F.relu(self.bn1(self.conv1(x))) # [batch_size, channel_size=64, sample_points_num]
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))          # the last layer of the fully connected block don't use activation
        x = torch.max(x, 2, keepdim=True)[0] # just need the result, don't need values' indices. [bs, 1024, 1]
        x = x.view(-1, 1024)                 # summarize the feature point of each channel. [bs, 1024]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            # False, the whole pipeline has not done yet
            # return the local feature concatenated with
            # the global feature

            # repeat 'n_pts' times, because input points
            # in every  channel(here is 1024)  need  the
            # feature value of that channel.
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            # concatenate on 'channel' dimension. Channel in 'x' and 'pointfeat' of each batch
            # are concatenated together. Dimension at which the concate is taken place on  can 
            # be unequal, other dimensions must match.
            return torch.cat([x, pointfeat], 1), trans, trans_feat # 1024 + 64

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(in_channel=3, ft_channel=64, global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, use_feattrans=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.use_feattrans=use_feattrans
        self.feat = PointNetfeat(in_channel=3, ft_channel=64, global_feat=False, feature_transform=use_feattrans)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

class PointNetSegmnCls(nn.Module):
    def __init__(self, in_channel=9, num_cls=14, use_feattrans=True) -> None:
        super(PointNetSegmnCls, self).__init__()
        # output of STN-feat block is concatenated to the back of original input
        self.k = num_cls
        self.feat = PointNetfeat(in_channel=in_channel, ft_channel=64, global_feat=False, feature_transform=use_feattrans)
        self.conv1 = torch.nn.Conv1d(in_channels=1088, out_channels=512,    kernel_size=1)
        self.conv2 = torch.nn.Conv1d(in_channels=512,  out_channels=256,    kernel_size=1)
        self.conv3 = torch.nn.Conv1d(in_channels=256,  out_channels=128,    kernel_size=1)
        self.conv4 = torch.nn.Conv1d(in_channels=128,  out_channels=self.k, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self, x: torch.Tensor):
        batch_size, n_pts = x.size()[0], x.size()[2]
        x, trans, trans_feat = self.feat(x) # spatial trans: 3x3, feature trans: 64x64
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1) # used with NLLLoss
        x = x.view(batch_size, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
