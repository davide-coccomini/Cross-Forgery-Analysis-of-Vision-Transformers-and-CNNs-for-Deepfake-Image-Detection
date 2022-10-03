import argparse
import json
import os
import numpy as np
from typing import Type

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
from utils import get_video_paths, get_method
import argparse


def process_videos(videos, detector_cls: Type[VideoFaceDetector], opt):
    detector = face_detector.__dict__[detector_cls](device="cuda:0")
    dataset = VideoDataset(videos)

    loader = DataLoader(dataset, shuffle=False, num_workers=40, batch_size=1, collate_fn=lambda x: x)
    missed_videos = []
    for item in tqdm(loader): 
        result = {}
        video, indices, frames = item[0]
        id = os.path.splitext(os.path.basename(video))[0]
        tmp = video.split("release")[1]
        out_dir = opt.output_path + tmp
        out_dir = out_dir.replace(id, '')

        #if os.path.exists(out_dir) and "{}.json".format(id) in os.listdir(out_dir):
        #    continue
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
      
        for j, frames in enumerate(batches):
            result.update({i : b for i, b in zip(indices, detector._detect_faces(frames))})
        
       
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
            json.dump(result, f)
        
        found_faces = False
        for key in result:
            if type(result[key]) == list:
                found_faces = True
                break
        if not found_faces:
            print("Faces not found", video)
            missed_videos.append(video)

    if len(missed_videos) > 0:
        print("The detector did not find faces inside the following videos:")
        print(missed_videos)
        print(len(missed_videos))
        print("We suggest to re-run the code decreasing the detector threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default="../data/forgerynet/training_image_list.txt", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--data_path', type=str,
                        help='Data directory', default='../data/forgerynet/Training/image')
    parser.add_argument('--output_path', type=str,
                        help='Output directory', default='../data/forgerynet/Training/boxes')
    parser.add_argument("--detector_type", help="type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
                        
    opt = parser.parse_args()
    print(opt)
    forgery_methods = [10,12,13,14]
    dataframes = []
    df = pd.read_csv(opt.list_file, sep=' ', usecols = [0, 3])
    for forgery_method in forgery_methods:
        df1 = df.loc[(df["16cls_label"] == forgery_method)]
        #if opt.max_videos > -1:
        #    df1 = df1.head(opt.max_videos * 4)
        dataframes.append(df1)


    
    df = pd.concat(dataframes)
    df = df.drop(['16cls_label'], axis=1)
    videos_paths = df.values.tolist()
    videos_paths = list(dict.fromkeys([os.path.join(opt.data_path, os.path.dirname(row[0].split(" ")[0])) for row in videos_paths]))

    
    process_videos(videos_paths, opt.detector_type, opt)
   
if __name__ == "__main__":
    main()