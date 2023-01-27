from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import tensorboardX

import sys
sys.path.append(".") # to import outter dir's modules
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',   type=int, default=8, help='input batch size')
parser.add_argument('--num_points',   type=int, default=1024, help='points per sample')
parser.add_argument('--workers',      type=int, default=4, help='number of data loading workers')
parser.add_argument('--nepoch',       type=int, default=100, help='number of epochs to train for')
parser.add_argument('--out_folder',   type=str, default="cls", help='output folder [cls|seg]')
parser.add_argument('--model',        type=str, default="", help='existing previous model path')
parser.add_argument('--dataset_path', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default="modelnet40", help="dataset type [shapenet|modelnet40]")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
print("cuda available: {}, device count: {}".format(torch.cuda.is_available(), torch.cuda.device_count()))

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset_path,
        classification=True,
        split="train",
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset_path,
        classification=True,
        split="test",
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset_path,
        npoints=opt.num_points,
        split="train")

    test_dataset = ModelNetDataset(
        root=opt.dataset_path,
        split="test",
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.out_folder)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda() # after giving out parameters, then shift to gpu

tensorboard_writer = tensorboardX.SummaryWriter(os.path.join(opt.out_folder, "tf_log_bs%d_np%d_dt[%s]"%(opt.batch_size, opt.num_points, opt.dataset_type)))
num_batch = len(dataset) / opt.batch_size
for epoch in range(opt.nepoch):
    for i, data in tqdm(enumerate(dataloader, 0), total=int(num_batch), ncols=100, desc="batch progress"):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        tqdm.write(" %s loss: %.3f accr: %.3f" % (blue("train"), loss.item(), correct.item() / float(opt.batch_size)))
        tensorboard_writer.add_scalar(tag="train loss", scalar_value=loss.item(), global_step=(epoch * num_batch + i + 1))
        tensorboard_writer.add_scalar(tag="train accr", scalar_value=correct.item(), global_step=(epoch * num_batch + i + 1))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            tqdm.write(" %s loss: %.3f accr: %.3f" % (blue("test "), loss.item(), correct.item() / float(opt.batch_size)))
    scheduler.step()
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.out_folder, epoch))

total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))