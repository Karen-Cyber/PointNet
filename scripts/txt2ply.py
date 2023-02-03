import os
import sys
import numpy as np
import time
import argparse
from tqdm import tqdm

from multiprocessing import Process, Lock
from plyfile import PlyData, PlyElement

S3DIS_DIR_TXT = "/home/hm/fuguiduo/datasets/S3DIS_txt"
S3DIS_DIR_PLY = "/home/hm/fuguiduo/datasets/S3DIS_ply"

cls2id = {
    "beam": 0,
    "board": 1,
    "bookcase": 2,
    "ceiling": 3,
    "chair": 4,
    "clutter": 5,
    "door": 6,
    "floor": 7,
    "table": 8,
    "wall": 9,
    "column" :10,
    "sofa" :11,
    "window" :12,
    "stairs" :13
}
id2cls = {v : k for k, v in cls2id.items()}
ply_line_type = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "u1")])

def multiprocess_print(msg, lck, is_err=False):
    lck.acquire()
    if is_err:
        print(msg, file=sys.stderr)
    else:
        print(msg)
    lck.release()

def txt2ply(src_dir, dst_dir, proc_idx, area, print_lock):
    t_curr = time.time()

    rooms = sorted([room for room in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, room))])
    room_num = len(rooms)
    multiprocess_print("proc#{} will transform {}:{}".format(proc_idx, area, rooms), print_lock)
    for room_idx, room in enumerate(rooms):
        if os.path.exists(os.path.join(dst_dir, room + ".ply")):
            multiprocess_print("{}-{}.ply exists, skip...".format(area, room), print_lock)
            continue
        multiprocess_print("proc#%d processing %s" % (proc_idx, room), print_lock)
        room_dir = os.path.join(src_dir, room, "Annotations")
        room_points = np.zeros(0, dtype=ply_line_type)

        # objects' annotation in this room
        objs = sorted([obj for obj in os.listdir(room_dir) if obj.endswith(".txt")])
        obj_num = len(objs)
        for obj_idx, obj in enumerate(objs):
            if os.path.exists(os.path.join(dst_dir, room, obj[:-4] + ".ply")):
                multiprocess_print("{}-{}-{}.ply exists, skip...".format(area, room, obj[:-4]), print_lock)
                continue
            obj_points = np.zeros(0, dtype=ply_line_type)
            cls = cls2id[obj.split('_')[0]]
            with open(os.path.join(room_dir, obj), 'r') as f:
                lines = f.readlines()
                line_num = len(lines)
                for line_idx, line in enumerate(lines):
                    str_vals = line.rstrip().split(' ')
                    try:
                        # some data are all floats, some has xyz float and rgb int
                        # need to handle the converting error
                        ply_line = np.array(tuple([float(val) for val in str_vals[:3]] + [int(float(val)) for val in str_vals[3:]] + [cls]), dtype=ply_line_type)
                    except Exception as e:
                        multiprocess_print("proc#{}: {}, {}:[{}/{}], {}:[{}/{}], points:[{}/{}], err:{}".format(
                                proc_idx, area, 
                                room, room_idx + 1, room_num,
                                obj, obj_idx + 1, obj_num,
                                line_idx + 1, line_num,
                                e
                            ),
                            print_lock,
                            is_err=True
                        )
                    obj_points = np.append(obj_points, ply_line)
                    # print progress every 10 secs
                    if time.time() - t_curr > 10.0:
                        multiprocess_print("proc#{}: {}, {}:[{}/{}], {}:[{}/{}], points:[{}/{}]".format(
                                proc_idx, area, 
                                room, room_idx + 1, room_num,
                                obj, obj_idx + 1, obj_num,
                                line_idx + 1, line_num
                            ),
                            print_lock
                        )
                        t_curr = time.time()
            if not os.path.exists(os.path.join(dst_dir, room)):
                os.makedirs(os.path.join(dst_dir, room))
            PlyData([PlyElement.describe(obj_points, "vertex", comments=["vertex with rgb and label"])]).write(os.path.join(dst_dir, room, obj[:-4] + ".ply"))

            room_points = np.append(room_points, obj_points)
        PlyData([PlyElement.describe(room_points, "vertex", comments=["vertex with rgb and label"])]).write(os.path.join(dst_dir, room + ".ply"))

def intergrity_check(dst_dir):
    check_log = open("completeness_check.log", 'w')
    to_be_fixed = []

    areas = sorted(os.listdir(dst_dir))
    for area in areas:
        rooms = sorted([room for room in os.listdir(os.path.join(dst_dir, area)) if room.endswith(".ply")])
        for room in tqdm(rooms, ncols=100, total=len(rooms), desc="checking {}".format(area)):
            plydata = PlyData.read(os.path.join(dst_dir, area, room))
            room_label_dict = dict()
            for l in plydata["vertex"]["label"]:
                if l not in room_label_dict.keys():
                    room_label_dict[l] =  0
                else:
                    room_label_dict[l] += 1
            
            parts = sorted(os.listdir(os.path.join(dst_dir, area, room[:-4])))
            part_label_dict = dict()
            for part in parts:
                plydata = PlyData.read(os.path.join(dst_dir, area, room[:-4], part))
                cls_id = cls2id[part.split('_')[0]]
                if cls_id not in part_label_dict.keys():
                    part_label_dict[cls_id] = 0
                part_label_dict[cls_id] += len(plydata["vertex"])
            
            missing_parts = []
            for cls_id, pts_num in part_label_dict.items():
                if cls_id not in room_label_dict:
                    missing_parts.append((id2cls[cls_id], pts_num))
                    continue
                if pts_num != room_label_dict[cls_id]:
                    missing_parts.append((id2cls[cls_id], "{}:{}".format(pts_num, room_label_dict[cls_id])))
                    continue
            
            check_log.write("{}-{} missing:\n".format(area, room))
            for part in missing_parts:
                check_log.write("\tcls:{}, pts:{}\n".format(part[0], part[1]))
            if len(missing_parts) != 0:
                to_be_fixed.append((area, room))
            
    check_log.close()
    return to_be_fixed

def intergrate_room_objs(dst_dir, room_list, from_file=False):
    if from_file:
        with open(room_list, 'r') as f:
            room_list = []
            for line in f.readlines():
                room_list.append(tuple(line.rstrip().split(' ')))

    for area, room in tqdm(room_list, ncols=100, total=len(room_list), desc="fixed progress"):
        room_pts = np.zeros(0, dtype=ply_line_type)

        part_list = sorted(os.listdir(os.path.join(dst_dir, area, room[:-4])))
        for part in part_list:
            part_plydata = PlyData.read(os.path.join(dst_dir, area, room[:-4], part))
            part_pts = np.array(
                [
                    (x,y,z,r,g,b,l) for x,y,z,r,g,b,l in zip(
                    part_plydata["vertex"]['x'], 
                    part_plydata["vertex"]['y'], 
                    part_plydata["vertex"]['z'], 
                    part_plydata["vertex"]["red"], 
                    part_plydata["vertex"]["green"], 
                    part_plydata["vertex"]["blue"], 
                    part_plydata["vertex"]["label"])
                ], dtype=ply_line_type
            )
            tqdm.write("{}-{}-{}: {}+{}={}".format(area, room, part, len(room_pts), len(part_pts), len(room_pts) + len(part_pts)))
            room_pts = np.append(room_pts, part_pts)
        PlyData([PlyElement.describe(room_pts, "vertex", comments=["vertex with rgb and label"])]).write(os.path.join(dst_dir, area, room))
        tqdm.write("{}-{} has been fixed".format(area, room))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fixdt", help="path to crack room list txt")
    args = parser.parse_args()

    if args.trans:
        print_lock = Lock()
        processes = []
        areas = sorted([area for area in os.listdir(S3DIS_DIR_TXT) if "Area_" in area])
        for idx, area in enumerate(areas):
            src_dir = os.path.join(S3DIS_DIR_TXT, area)
            dst_dir = os.path.join(S3DIS_DIR_PLY, area)
            sub_process = Process(target=txt2ply, args=(src_dir, dst_dir, idx, area, print_lock))
            processes.append(sub_process)
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    if args.check:
        to_be_fixed = intergrity_check(S3DIS_DIR_PLY)
        dump_file = open("to_be_fixed.txt", "w")
        for area, room in to_be_fixed:
            dump_file.write("{} {}\n".format(area, room))
        dump_file.close()

    if args.fixdt is not None:
        intergrate_room_objs(S3DIS_DIR_PLY, args.fixdt, from_file=True)

