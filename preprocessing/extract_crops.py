import argparse
import json
import os
from os import cpu_count
from pathlib import Path
from collections import OrderedDict

import pandas as pd
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm

from utils import get_video_paths, get_method, get_method_from_name

def extract_video(video, data_path):

    bboxes_path = data_path + "/boxes/" + video.split("release")[1] + ".json"
    if not os.path.exists(bboxes_path) or not os.path.exists(video):
        print(video, "not found")
        return
    with open(bboxes_path, "r") as bbox_f:
        bboxes_dict = json.load(bbox_f)
    frames_names = os.listdir(video)
    
    frames_num = len(frames_names)
    
    frames = OrderedDict()
    for i in range(frames_num):
        frame_path = os.path.join(video, frames_names[i])
        frame = cv2.imread(frame_path)
        frames[frames_names[i]] = frame
    #frames = list(frames.values())
    
    counter = 0
    for i, key in enumerate(bboxes_dict.keys()):
        #if i % 2 != 0:
        #    continue
        frame = frames[key]
        id = os.path.splitext(os.path.basename(video))[0]
        crops = []
        bboxes = bboxes_dict[key]
        if bboxes is None:
            print(id, bboxes_path)
            continue
        else:
            counter += 1
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = 0
            p_w = 0
            
            #p_h = h // 3
            #p_w = w // 3
            
            #p_h = h // 6
            #p_w = w // 6

            if h > w:
                p_w = int((h-w)/2)
            elif h < w:
                p_h = int((w-h)/2)

            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            h, w = crop.shape[:2]
            crops.append(crop)

        tmp = video.split("release")[1]
        out_dir = opt.output_path + tmp
        os.makedirs(out_dir, exist_ok=True)
        frame_name = os.path.splitext(key)[0]
        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(out_dir, "{}_{}.png".format(frame_name, j)), crop)
            
    #if counter == 0:
        #print(video, counter)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default="../data/forgerynet/training_image_list.txt", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--data_path', default='../data/forgerynet/Training', type=str,
                        help='Videos directory')
    parser.add_argument('--output_path', default='../data/forgerynet/Training/crops', type=str,
                        help='Output directory')

    opt = parser.parse_args()
    print(opt)
    

    
    os.makedirs(opt.output_path, exist_ok=True)
    #excluded_videos = os.listdir(os.path.join(opt.output_dir)) # Useful to avoid to extract from already extracted videos
    #excluded_videos = os.listdir(opt.output_path)
    
    df = pd.read_csv(opt.list_file, sep=' ', usecols = [0, 3])
    
    df = df.loc[(df["16cls_label"] == 11)]
    df = df.drop(['16cls_label'], axis=1)
    videos_paths = df.values.tolist()
    videos_paths = list(dict.fromkeys([os.path.join(opt.data_path, "image", os.path.dirname(row[0].split(" ")[0])) for row in videos_paths]))
    
    with Pool(processes=cpu_count()-2) as p:
        with tqdm(total=len(videos_paths)) as pbar:
            for v in p.imap_unordered(partial(extract_video, data_path=opt.data_path), videos_paths):
                pbar.update()