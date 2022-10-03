
import torch
import random
import yaml
import argparse
import pandas as pd
import os
from os import cpu_count
import shutil
import cv2
import numpy as np
import math
import tensorflow as tf
from multiprocessing import Manager
from multiprocessing.pool import Pool
from progress.bar import Bar
from tqdm import tqdm
from functools import partial
from utils import custom_round

from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
from sklearn.model_selection import train_test_split

from timm.models.efficientnet import tf_efficientnet_b7_ns
from efficientnetv2.effnetv2 import effnetv2_m
from vit_pytorch import ViT
from coatnet.coatnet import coatnet_2
from mlp_mixer_pytorch import MLPMixer

from sklearn.metrics import accuracy_score
from statistics import mean

import timm
from transformers import ViTForImageClassification, ViTConfig


def save_confusion_matrix(confusion_matrix):
  fig, ax = plt.subplots()
  im = ax.imshow(confusion_matrix, cmap="Blues")

  threshold = im.norm(confusion_matrix.max())/2.
  textcolors=("black", "white")

  ax.set_xticks(np.arange(2))
  ax.set_yticks(np.arange(2))
  ax.set_xticklabels(["original", "fake"])
  ax.set_yticklabels(["original", "fake"])
  
  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

  for i in range(2):
      for j in range(2):
          text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", 
                         fontsize=12, color=textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)])

  fig.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, "confusion.jpg"))


  
def read_frames(paths, dataset, opt, image_size):
    transform = create_base_transform(image_size)
    for path in paths:
        video_path_tmp = os.path.dirname(path[0]).split("release")[1]
        video_path = opt.data_path + os.path.sep + "crops" + video_path_tmp
        if not os.path.exists(video_path):
            return
        frames_names = os.listdir(video_path)
        image_name = os.path.splitext(os.path.basename(path[0]))[0]
        frame_names = [frame_name for frame_name in frames_names if image_name in frame_name]
        label = path[1]
        for frame_name in frame_names:
            image = transform(image=cv2.imread(os.path.join(video_path, frame_name)))['image']
            row = (image, label, os.path.join(video_path, frame_name))
            dataset.append(row)


def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])


# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name.')
    parser.add_argument('--data_path', default='data/forgerynet/Validation', type=str,
                        help='Videos directory')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for validation (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--list_file', default="data/forgerynet/validation_image_list.txt", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--use_pretrained', type=bool, default=True, 
                        help="Use pretrained models")
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained (0: ViT, 1: EfficientNet B7, 2: Hybrid)")
    parser.add_argument('--forgery_method', type=int, default=1, 
                        help="Forgery method used for training")
    parser.add_argument('--save_errors', type=bool, default=True, 
                        help="Save errors in directory?")
    opt = parser.parse_args()
    print(opt)

    
    # Model Loading
    if opt.config != '':
        with open(opt.config, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
    
    batch_size = 1

    if opt.model == 0: # ViT-Base: 85.771.777 parameters / 86.859.496 pretrained
        if opt.use_pretrained:
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', ignore_mismatched_sizes=True, num_labels=1)
            
        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
            '''
            model = ViT(
                image_size = config['model']['image-size'],
                patch_size = config['model']['patch-size'],
                num_classes = config['model']['num-classes'],
                dim = config['model']['dim'],
                depth = config['model']['depth'],
                heads = config['model']['heads'],
                mlp_dim = config['model']['mlp-dim'],
                dropout = config['model']['dropout'],
                emb_dropout = config['model']['emb-dropout']
            )
            '''
    elif opt.model == 1: # EfficientNet-V2-M: 52.859.637 parameters | EfficientNet B7 63.789.521  
        if opt.use_pretrained:
            model = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=1)
        else:
            model = tf_efficientnet_b7_ns(num_classes=1, pretrained=False)
            #model = effnetv2_m()
    elif opt.model == 2: # CoAtNet2: 54.742.590 parameters
        model = coatnet_2()
    elif opt.model == 3: # MLP-Mixer: 60.998.641  parameters
        model = MLPMixer(
            image_size = config['model']['image-size'],
            channels = config['model']['channels'],
            patch_size = config['model']['patch-size'],
            dim = config['model']['dim'],
            depth = config['model']['depth'],
            num_classes = config['model']['num-classes']
        )

    model.load_state_dict(torch.load(opt.model_path))


    df = pd.read_csv(opt.list_file, sep=' ', usecols=[0, 1, 3])
     
    results_df = pd.DataFrame(columns=[i for i in range(16)], index=[0])
    fake_accuracy = 0
    real_accuracy = 0
    values = []
    print("FORGERY METHOD", opt.forgery_method, "MODEL", opt.model_path)
    for forgery_method in range(0, 16):
        f = open(os.path.join("results", opt.model_name + "_" + str(opt.forgery_method) + "_" + str(forgery_method) + "_labels.txt"), "w+")
        df_tmp = df.loc[(df["16cls_label"] == forgery_method)]
        if opt.max_videos > -1:
            df = df.head(opt.max_videos)
        df_tmp = df_tmp.sort_values(by=['image_name'])
        paths = df_tmp.to_numpy()
        paths = np.array_split(paths, cpu_count()) # Split the paths in chunks for processing
        
        mgr = Manager()
        dataset = mgr.list()

        with Pool(processes=cpu_count()) as p:
            with tqdm(total=len(paths)) as pbar:
                for v in p.imap_unordered(partial(read_frames, dataset=dataset, opt=opt, image_size = config['model']['image-size']), paths):
                    pbar.update()

        labels = [float(row[1]) for row in dataset]
        face_paths = [row[2] for row in dataset]
        dataset = [row[0] for row in dataset]
        model = model.cuda()
        faces_preds = []
        bar = Bar('Predicting', max=int(len(dataset)/batch_size))

        for i in range(0, len(dataset), batch_size):
        
            faces = dataset[i:i+batch_size]
            faces = np.transpose(faces, (0, 3, 1, 2))            
            faces = torch.tensor(np.asarray(faces))
            if faces.shape[0] == 0:
                continue
            
            faces = faces.cuda().float()
            pred = model(faces)
            if opt.model == 0:
                pred = pred.logits
            pred = pred.cpu().detach()
            faces = faces.cpu().detach()
            scaled_pred = []
            for idx, p in enumerate(pred):
                scaled_pred.append(torch.sigmoid(p))
            faces_preds.extend(scaled_pred)
            bar.next()

        final_preds = []
        correct_labels = []

        for i in range(len(labels)):
            current_label = labels[i]
            tmp_path = face_paths[i].split("_")[0]
            tmp_preds = []
            current_preds = np.asarray(faces_preds[i])
            tmp_preds.append(current_preds[0])
            f.write(" " + str(current_preds[0]))
            while i+1 < len(face_paths) and face_paths[i+1].split("_")[0] == tmp_path: # Same frame, different faces
                current_preds = np.asarray(faces_preds[i+1])
                tmp_preds.append(current_preds[0])
                f.write(" " + str(current_preds[0]))
                i += 1
            frame_pred = max(tmp_preds)
            final_preds.append(custom_round(frame_pred))
            correct_labels.append(current_label)
            f.write(" --> " + str(frame_pred) + "(CORRECT: " + str(current_label) + ")" +"\n")
            if opt.save_errors:
                if custom_round(frame_pred) != current_label:
                    output_path = os.path.join("errors", str(opt.model_name), str(opt.forgery_method), str(forgery_method))
                    os.makedirs(output_path, exist_ok = True)
                    shutil.copy(face_paths[i], os.path.join(output_path, os.path.basename(face_paths[i])))                    
            

        current_accuracy = accuracy_score(correct_labels, final_preds)
        if forgery_method == 0:
            real_accuracy = current_accuracy
        else:
            fake_accuracy += current_accuracy

        string = "ACCURACY: " + str(current_accuracy)
        print("Method", forgery_method, string)
        f.write(string)
        results_df[forgery_method] = round(current_accuracy, 3)
        values.append(round(current_accuracy, 3))
    
    fake_accuracy /= 14
    global_accuracy = mean([fake_accuracy, real_accuracy])

    results_df.insert(16, "real_accuracy", real_accuracy)
    results_df.insert(17, "fake_accuracy", fake_accuracy)
    results_df.insert(18, "global_accuracy", global_accuracy)
    results_df.insert(19, "variance", np.var(values))
    print(results_df)
    f.close()
    results_df.to_csv(os.path.join("results", opt.model_name + "_" + str(opt.forgery_method) + "_metrics.csv"))
    bar.finish()

    