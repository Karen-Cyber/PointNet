from __future__ import print_function
import argparse
import os
import random
from tqdm import tqdm
import datetime

import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import tensorboardX as tfx

import sys
sys.path.append(".") # to import outter dir's modules
from pointnet.dataset import S3DISNetDataset
from pointnet.model import PointNetSegmnCls, feature_transform_regularizer

blue = lambda s: "\033[94m" + s + "\033[0m"
redd = lambda s: "\033[91m" + s + "\033[0m"
gren = lambda s: "\033[92m" + s + "\033[0m"

# for debug convenience, set default values for all arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',   type=int, default=8, help='input batch size')
parser.add_argument('--num_points',   type=int, default=4096, help='points per sample')
parser.add_argument('--num_workers',  type=int, default=8, help='number of data loading workers')
parser.add_argument('--num_epochs',   type=int, default=100, help='number of epochs to train for')
parser.add_argument('--out_folder',   type=str, default="./results/sem", help='output folder [cls|sem]')
parser.add_argument('--model',        type=str, default="", help='existing previous model path')
parser.add_argument('--dataset_path', type=str, default="", help="dataset path", required=True)
parser.add_argument('--dataset_type', type=str, default="", help="dataset type [shapenet|modelnet40|s3dis]")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

args = parser.parse_args()
for arg, val in vars(args).items():
    print("{}: {}".format(blue(arg), val))

try:
    os.makedirs(args.out_folder)
except Exception as e:
    print(redd(str(e)))

args.manual_seed = random.randint(1, 10000)  # fix seed
print("random seed: ", args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)


print("cuda available: {}, device count: {}".format(torch.cuda.is_available(), torch.cuda.device_count()))
using_device = "cuda" if torch.cuda.is_available() else "cpu"

cls2id = dict()
with open("./misc/scen_seg_classes.txt", 'r') as f:
    for line in f.readlines():
        line = line.rstrip().split(' ')
        cls2id[line[0]] = int(line[1])
id2cls = {v: k for k, v in cls2id.items()}



# dataset preparation
dataset = S3DISNetDataset(
    data_root=args.dataset_path,
    split="train",
    num_point=args.num_points,
    test_area=4
)
test_dataset = S3DISNetDataset(
    data_root=args.dataset_path,
    split="test",
    num_point=args.num_points,
    test_area=4
)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)

print(gren("train set len:"), len(dataset))
print(gren("test  set len:"), len(test_dataset))
num_classes = len(cls2id)
print('classes', num_classes)


classifier = PointNetSegmnCls(in_channel=9, num_cls=num_classes, use_feattrans=args.feature_transform)
if args.model != '':
    # we only save the state dict of the model
    classifier.load_state_dict(torch.load(args.model))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier = classifier.to(device=using_device) # after giving out parameters, then shift to gpu




# training preparation

tensorboard_writer =tfx.SummaryWriter(
    os.path.join(
        args.out_folder, 
        "tf_log_%s_bs%d_np%d_nk%d_tf%i_dt%s"%(
            datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S"), 
            args.batch_size, 
            args.num_points, 
            args.num_workers, 
            args.feature_transform, 
            args.dataset_type
        )
    )
)

def rotate_augmentt(batch_data: torch.Tensor):
    ''' 
    Randomly rotate the point clouds along the Y-aixs\
    to augment the dataset.
    
    params:
    * batch_data - BxNxF array, original batch of point clouds
    
    return:
    * batch_data - BxNxF array, rotated batch of point clouds
    '''
    # for semantic segmentation, F stands for 9 features
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotmat = np.array(
        [
            [cosval,  0.0, sinval],
            [1.0,     0.0,    1.0],
            [-sinval, 0.0, cosval]
        ]
    )
    batch_data = batch_data.numpy()
    batch_data[:, :, :3] = np.dot(batch_data[:, :, :3], rotmat)
    return torch.from_numpy(batch_data)

def class_info_dump(sem_pred_choice: torch.Tensor, target: torch.Tensor, global_step: int, is_training=True):
    main_tag = "train" if is_training else "test"
    total_seen = np.array([0.0] * num_classes) # points of each class in the label
    total_accr = np.array([0.0] * num_classes) # accuracy of each class
    total_diou = np.array([0.0] * num_classes) # denominator of IoU

    for cls in range(num_classes):
        total_seen[cls] = (target == cls).float().sum().item() + 1e-6
        total_accr[cls] = ((sem_pred_choice == cls) & (target == cls)).float().sum().item()
        total_diou[cls] = ((sem_pred_choice == cls) | (target == cls)).float().sum().item() + 1e-6

    accr_dict = dict()
    for cls in range(num_classes):
        accr_dict[id2cls[cls]] = total_accr[cls] / total_seen[cls] # avoid zero-division
    tensorboard_writer.add_scalars(main_tag=main_tag, tag_scalar_dict=accr_dict, global_step=global_step)

    ciou_dict = dict()
    for cls in range(num_classes):
        ciou_dict[id2cls[cls]] = total_accr[cls] / total_diou[cls] # avoid zero-division
    tensorboard_writer.add_scalars(main_tag=main_tag, tag_scalar_dict=ciou_dict, global_step=global_step)

    for (cls, accr), (_, ciou) in zip(accr_dict.items(), ciou_dict.items()):
        tqdm.write("    %s\taccr: %.3f, ciou: %.3f" % (blue(cls), accr, ciou))

    mIoU = np.mean(total_accr / total_diou)
    macc = np.mean(total_accr / total_seen)
    return mIoU, macc

def lossfn(sem_pred, target, trans_feat, labelweights):
    loss = F.nll_loss(sem_pred, target, labelweights)
    if args.feature_transform and trans_feat is not None:
        loss += feature_transform_regularizer(trans_feat) * 0.001
    return loss

def train_one_epoch(epoch_num):
    batch_num_per_epoch = len(dataset) / args.batch_size
    labelweights = torch.from_numpy(dataset.labelweights).to(device=using_device)
    total_accr = 0.0
    total_loss = 0.0
    total_seen = 0
    bestt_mIoU = 0.0 # bestt mIou
    batch_mIoU = 0.0 # batch mIoU

    for i, (points, target) in tqdm(enumerate(dataloader, 0), total=len(dataloader), ncols=100, desc="batch progress"):
        classifier.train()

        points = rotate_augmentt(points)
        points, target = points.float().to(device=using_device), target.long().to(device=using_device)
        points = points.transpose(2, 1)

        optimizer.zero_grad()
        sem_pred, trans, trans_feat = classifier(points)

        sem_pred = sem_pred.contiguous().view(-1, num_classes) # [batch_size, points_per_sample, classes] -> [points_per_batch, classes]
        target = target.view(-1, 1)[:, 0]
        batch_loss = lossfn(sem_pred, target, trans_feat, labelweights)
        batch_loss.backward()
        optimizer.step()

        n_pts = sem_pred.shape[0]
        sem_pred_choice = sem_pred.max(dim=1)[1]
        total_seen += n_pts
        batch_accr  = (sem_pred_choice == target).float().sum().item()
        total_accr += batch_accr
        total_loss += batch_loss
        batch_accr  = batch_accr / n_pts

        if i % 10 == 0:
            tqdm.write("%s loss: %.3f, accr: %.3f"%(blue("train"), batch_loss, batch_accr))
            tensorboard_writer.add_scalar(tag="train/train_loss", scalar_value=batch_loss, global_step=epoch_num*batch_num_per_epoch + i)
            tensorboard_writer.add_scalar(tag="train/train_accr", scalar_value=batch_accr, global_step=epoch_num*batch_num_per_epoch + i)

        if i % 100 == 0:
            batch_mIoU, _ = class_info_dump(sem_pred_choice, target, epoch_num*batch_num_per_epoch + i)
            if batch_mIoU > bestt_mIoU:
                torch.save(classifier.state_dict(), "%s/sem_model_best.pth" % args.out_folder)

        if i % 100 == 0:
            # have a test
            labelweights = torch.from_numpy(test_dataset.labelweights).to(device=using_device)
            with torch.no_grad():
                classifier.eval()
                j, data = next(enumerate(test_dataloader, 0))
                points, target = data
                points = rotate_augmentt(points)
                points, target = points.float().to(device=using_device), target.long().to(device=using_device)
                points = points.transpose(2, 1)

                sem_pred, trans, trans_feat = classifier(points)

                sem_pred = sem_pred.contiguous().view(-1, num_classes) # [batch_size, points_per_sample, classes] -> [points_per_batch, classes]
                target = target.view(-1, 1)[:, 0]
                batch_loss = lossfn(sem_pred, target, trans_feat, labelweights)

                n_pts = sem_pred.shape[0]
                sem_pred_choice = sem_pred.max(dim=1)[1]
                batch_accr = (sem_pred_choice == target).float().sum().item() / n_pts

                tqdm.write("%s loss: %.3f, accr: %.3f"%(redd("test "), batch_loss, batch_accr))
                tensorboard_writer.add_scalar(tag="train/test_loss", scalar_value=batch_loss, global_step=epoch_num*batch_num_per_epoch + i)
                tensorboard_writer.add_scalar(tag="train/test_accr", scalar_value=batch_accr, global_step=epoch_num*batch_num_per_epoch + i)
    scheduler.step()

    epoch_accr = total_accr / total_seen
    epoch_loss = total_loss / batch_num_per_epoch
    tqdm.write("%s loss: %.3f, accr: %.3f"%(gren("epoch"), epoch_loss, epoch_accr))
    tensorboard_writer.add_scalar(tag="train/epoch_loss", scalar_value=epoch_loss, global_step=epoch_num)
    tensorboard_writer.add_scalar(tag="train/epoch_accr", scalar_value=epoch_accr, global_step=epoch_num)


def testt_one_epoch(epoch_num):
    batch_num_per_epoch = len(test_dataset) / args.batch_size
    labelweights = torch.from_numpy(test_dataset.labelweights).to(device=using_device)
    total_accr = 0.0
    total_loss = 0.0
    total_seen = 0

    class_seen = np.array([0.0] * num_classes)
    class_accr = np.array([0.0] * num_classes)
    class_diou = np.array([0.0] * num_classes)

    with torch.no_grad():
        classifier.eval()
        for i, (points, target) in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader), ncols=100, desc="batch progress"):
            points = rotate_augmentt(points)
            points, target = points.float().to(device=using_device), target.long().to(device=using_device)
            points = points.transpose(2, 1)

            sem_pred, trans, trans_feat = classifier(points)

            sem_pred = sem_pred.contiguous().view(-1, num_classes) # [batch_size, points_per_sample, classes] -> [points_per_batch, classes]
            target = target.view(-1, 1)[:, 0]

            batch_loss = lossfn(sem_pred, target, trans_feat, labelweights)

            n_pts = sem_pred.shape[0]
            sem_pred_choice = sem_pred.max(dim=1)[1]

            batch_accr = (sem_pred_choice == target).float().sum().item()
            total_seen += sem_pred.shape[0]
            total_accr += batch_accr
            total_loss += batch_loss

            for cls in range(num_classes):
                class_seen[cls] += (target == cls).float().sum().item()
                class_accr[cls] += ((sem_pred_choice == cls) & (target == cls)).float().sum().item()
                class_diou[cls] += ((sem_pred_choice == cls) | (target == cls)).float().sum().item()
    
    epoch_loss = total_loss / batch_num_per_epoch
    epoch_accr = total_accr / total_seen
    tqdm.write("%s loss: %.3f, accr: %.3f"%(redd("test "), epoch_loss, epoch_accr))
    tensorboard_writer.add_scalar(tag="test/epoch_loss", scalar_value=epoch_loss, global_step=epoch_num)
    tensorboard_writer.add_scalar(tag="test/epoch_accr", scalar_value=epoch_accr, global_step=epoch_num)

    for (cls, accr), (_, ciou) in zip(accr_dict.items(), ciou_dict.items()):
        tqdm.write("    %s accr: %.3f, ciou: %.3f" % (blue(cls), accr, ciou))
    
    accr_dict = dict()
    for cls in range(num_classes):
        accr_dict[id2cls[cls]] = class_accr[cls] / (class_seen[cls] + 1e-6) # avoid zero-division
    tensorboard_writer.add_scalars(main_tag="test", tag_scalar_dict=accr_dict, global_step=epoch_num)

    ciou_dict = dict()
    for cls in range(num_classes):
        ciou_dict[id2cls[cls]] = class_accr[cls] / (class_diou[cls] + 1e-6) # avoid zero-division
    tensorboard_writer.add_scalars(main_tag="test", tag_scalar_dict=ciou_dict, global_step=epoch_num)







# start training
for epoch_num in range(args.num_epochs):
    train_one_epoch(epoch_num)

    if epoch_num > 0 and epoch_num % 10 == 0:
        torch.save(classifier.state_dict(), '%s/sem_model_%d.pth' % (args.out_folder, epoch_num))

testt_one_epoch(0)