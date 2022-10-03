import torch
import random
import yaml
import argparse
import pandas as pd
import os
from os import cpu_count
import cv2
import numpy as np
import math
import tensorflow as tf
from multiprocessing import Manager
from multiprocessing.pool import Pool
from progress.bar import Bar
from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import timm
from timm.models.efficientnet import tf_efficientnet_b7_ns
from efficientnetv2.effnetv2 import effnetv2_m
from vit_pytorch import ViT
from coatnet.coatnet import coatnet_2
from mlp_mixer_pytorch import MLPMixer


from sklearn.model_selection import train_test_split
import collections
from deepfakes_dataset import DeepFakesDataset
from torch.optim import lr_scheduler
from progress.bar import ChargingBar
from utils import check_correct, resize, get_n_params
from transformers import ViTForImageClassification, ViTConfig

from torch.utils.tensorboard import SummaryWriter


IMAGE_SIZE = 224

def read_frames(paths, dataset, opt):
    fail = 0
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
            image = cv2.imread(os.path.join(video_path, frame_name))
            row = (image, label)
            dataset.append(row)
    if fail > 0:
        print(fail)

# Main body
if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--data_path', default='data/forgerynet/Training', type=str,
                        help='Videos directory')
    parser.add_argument('--list_file', default="data/forgerynet/training_image_list.txt", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name.')
    parser.add_argument('--model_path', type=str, default='models',
                        help='Path to save checkpoints.')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, default='',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained (0: ViT, 1: EfficientNet B7, 2: Hybrid)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    parser.add_argument('--forgery_methods', type=list, default=[10], 
                        help="Which forgery method for training?")
    parser.add_argument('--fake_multiplier', type=int, default=1, 
                        help="Oversampling factor for fake faces")
    parser.add_argument('--real_multiplier', type=int, default=1, 
                        help="Oversampling factor for real faces")
    parser.add_argument('--use_pretrained', type=bool, default=True, 
                        help="Use pretrained models")
    parser.add_argument('--show_stats', type=bool, default=True, 
                        help="Show stats")
    parser.add_argument('--logger_name', default='runs/train',
                        help='Path to save the model and Tensorboard log.')
                        
    opt = parser.parse_args()
    print(opt)

    # Model Loading
    if opt.config != '':
        with open(opt.config, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
    if opt.model == 0: 
        if opt.use_pretrained:
            '''
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)
            
            '''
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', ignore_mismatched_sizes=True, num_labels=1)
            for index, (name, param) in enumerate(model.named_parameters()):
                if "layer.11" in name or "layer.10" in name or index > len(list(model.parameters()))-10:
                    param.requires_grad = True
                else:                    
                    param.requires_grad = False
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
    elif opt.model == 1: 
        if opt.use_pretrained:
            model = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=1)
            for index, (name, param) in enumerate(model.named_parameters()):
                if "blocks.6" in name or "blocks.5" in name or index > len(list(model.parameters()))-10:
                    param.requires_grad = True
                else:                    
                    param.requires_grad = False
        else:
            model = tf_efficientnet_b7_ns(num_classes=1, pretrained=False)
            #model = effnetv2_m()

    elif opt.model == 2: 
        if opt.use_pretrained:
            print("Pretrained network not found for this model, working from scratch.")

        model = coatnet_2()
    elif opt.model == 3: 
        if opt.use_pretrained:
            print("Pretrained network not found for this model, working from scratch.")

        model = MLPMixer(
            image_size = config['model']['image-size'],
            channels = config['model']['channels'],
            patch_size = config['model']['patch-size'],
            dim = config['model']['dim'],
            depth = config['model']['depth'],
            num_classes = config['model']['num-classes']
        )


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters:", params)
    # Read dataset
    df = pd.read_csv(opt.list_file, sep=' ', usecols=[0, 1, 3])
    df = shuffle(df, random_state = 42)
    if opt.show_stats:
        print("**** STATS ****")
        string = ""
        for i in range(0, 16):
            string += "Class " + str(i) + ": " + str(len(df.loc[df["16cls_label"] == i])) + " | "
        print(string)
    
    dataframes = []
    for forgery_method in opt.forgery_methods:
        forgery_method = int(forgery_method)
        df1 = df.loc[(df["16cls_label"] == forgery_method)]
        if opt.max_videos > -1:
            df1 = df1.head(opt.max_videos*4)
        dataframes.append(df1)

    df2 = df.loc[(df["16cls_label"] == 0)]
    if opt.max_videos > -1:
        df2 = df2.head(opt.max_videos * (len(opt.forgery_methods)) *2)
    dataframes.append(df2)
    df = pd.concat(dataframes)
    df = df.drop(df[(df['16cls_label'] == 0) & (df.index % 2 == 0)].index)
    df = df.drop(['16cls_label'], axis=1)

    df = df.sort_values(by=['image_name'])
    paths = df.to_numpy()
    paths = np.array_split(paths, cpu_count()) # Split the paths in chunks for processing
    mgr = Manager()
    dataset = mgr.list()

    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, dataset=dataset, opt=opt), paths):
                pbar.update()
    
    dataset = sorted(dataset, key=lambda tup: tup[1])
    labels = [float(row[1]) for row in dataset]
    dataset = [row[0] for row in dataset]
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(dataset, labels, test_size=0.10, random_state=42)

    train_samples = len(train_dataset)
    validation_samples = len(validation_dataset)

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(train_labels)
    print(train_counters)
    
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(validation_labels)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Create the data loaders
    if opt.config != '':
        batch_size = config['training']['bs']
    else:
        batch_size = 8

    train_dataset = DeepFakesDataset(np.asarray(train_dataset), np.asarray(train_labels), IMAGE_SIZE)
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=0, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(np.asarray(validation_dataset), np.asarray(validation_labels), IMAGE_SIZE, mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset
    
    
    # TRAINING
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    experiment_path = tb_logger.get_logdir()
    
    model.train()   
    #if opt.model == 0:
    #    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    #else:
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("_")[1]) + 1
    else:
        print("No checkpoint loaded.")
        
    model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*batch_size)+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0
        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()
            
            y_pred = model(images)
            if opt.model == 0:
                y_pred = y_pred.logits

            y_pred = y_pred.cpu()
            
            loss = loss_fn(y_pred, labels)
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            for i in range(batch_size):
                bar.next()

            if index%1200 == 0:
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*batch_size), "Train 0s: ", negative, "Train 1s:", positive)  


        
        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0
       
        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dl):
    
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            
            val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            
            if opt.model == 0:
                val_pred = val_pred.logits

            val_pred = val_pred.cpu()
            val_loss = loss_fn(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_negative += negative_class
            val_counter += 1
            bar.next()
            
        scheduler.step()
        bar.finish()
        

        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
        
        tb_logger.add_scalar("Training/Accuracy", train_correct, t)
        tb_logger.add_scalar("Training/Loss", total_loss, t)
        tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], t)
        tb_logger.add_scalar("Validation/Loss", total_loss, t)
        tb_logger.add_scalar("Validation/Accuracy", val_correct, t)

        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(val_counters[0]) + " val_1s:" + str(val_positive) + "/" + str(val_counters[1]))
    
        
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)

        forgery_methods_string = 'm'.join([str(method) for method in opt.forgery_methods])
        torch.save(model.state_dict(), os.path.join(opt.model_path, opt.model_name + "_" + str(t) + "_" + forgery_methods_string))

    #training_set = list(dict.fromkeys([os.path.join(opt.data_path, os.path.dirname(row[0].split(" "))) for row in training_set]))
  
