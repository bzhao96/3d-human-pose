import os
import time
import yaml
import argparse
import numpy as np 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

from dataset import *
from model import *
from utils import total_accuracy_keypoints, soft_argmax

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

trans_to_tensor = torchvision.transforms.ToTensor()
trans_to_image = torchvision.transforms.ToPILImage()

def save_npz(joints, image_names, output_path, is_train):
    output = {}
    subjects = subjects_train if is_train else subjects_test
    for subject in subjects:
        output[subject] = {}
        for action in actions:
            for subaction in subactions:
                output[subject][action+'-'+subaction] = {}
                for camera in cameras:
                    if (subject, action, subaction, camera) == black_list:
                        continue
                    output[subject][action+'-'+subaction][camera] = []                  
    for i in range(len(joints)):
        s = image_names[i].split('/')
        subject = s[-4]
        action_full = s[-3]
        camera = s[-2]
        output[subject][action_full][camera].append(np.expand_dims(joints[i], 0))
    for subject in output.keys():
        for action in output[subject].keys():
            for camera in output[subject][action].keys():
                output[subject][action][camera] = np.concatenate(output[subject][action][camera], 0)
    np.savez_compressed(output_path, pose2d=output)
    return



def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1]
    
    train_loader = DataLoader(
        H36M_Loader(cfg['data']['input'], img_size=cfg['model']['image_size']), 
        batch_size=cfg['training']['batch_size'], 
        num_workers=cfg['training']['n_workers'],
        shuffle=False,
        )
    test_loader = DataLoader(
        H36M_Loader(cfg['data']['test_all'], False, img_size=cfg['model']['image_size']), 
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['n_workers'],
        shuffle=False,
        )

    model = get_pose_net(cfg).to(device)
    checkpoint = torch.load(cfg['model']['restore'])
    model.load_state_dict(checkpoint['model'])
    model = nn.DataParallel(model, device_ids)
    
    model.eval()
    with torch.no_grad():
        ###############################################
        ### predict 2d keypoints of test set ##########
        ###############################################
        acc_test_1 = 0.0
        acc_test_2 = 0.0
        acc_test_4 = 0.0
        num_test = 0
        joints = []
        image_names = []
        joint_flip = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        for image, image_flip, keypoints, scale, image_name in tqdm(test_loader):
            image = image.to(device)
            output = model(image).cpu().numpy()
            image_flip = image_flip.to(device)
            output_flip = model(image_flip).cpu().numpy()
            output_flip = output_flip[..., ::-1]
            output_flip = output_flip[:, joint_flip, ...]
            output = (output + output_flip) / 2.0
            output_keypoints = soft_argmax(torch.from_numpy(output))
            output_keypoints = output_keypoints.numpy()

            acc_test_1 += total_accuracy_keypoints(output_keypoints, keypoints.numpy(), 1) * image.shape[0]
            acc_test_2 += total_accuracy_keypoints(output_keypoints, keypoints.numpy(), 2) * image.shape[0]
            acc_test_4 += total_accuracy_keypoints(output_keypoints, keypoints.numpy(), 4) * image.shape[0]
            num_test += image.shape[0]

            if cfg['testing']['save']:
                joints.append(output_keypoints * 4.0 / scale.numpy())
                image_names += image_name

        print("Acc Threshold 1 {:.4f}".format(acc_test_1/num_test), flush=True)
        print("Acc Threshold 2 {:.4f}".format(acc_test_2/num_test), flush=True)
        print("Acc Threshold 4 {:.4f}".format(acc_test_4/num_test), flush=True)
        joints = np.concatenate(joints, 0)
        print(joints.shape)
        print(len(image_names))
        print("save test dataset prediction results...", flush=True)
        save_npz(joints, image_names, os.path.join(cfg['data']['output'], 'h36m_pred2d_cropped_test_all.npz'), False)
        
        ###############################################
        ### predict 2d keypoints of training set    ###
        ###############################################
        # if cfg['testing']['save']:
        #     acc_test_1 = 0.0
        #     acc_test_2 = 0.0
        #     acc_test_4 = 0.0
        #     num_test = 0
        #     joints = []
        #     image_names = []
        #     joint_flip = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        #     for image, image_flip, keypoints, scale, image_name in tqdm(train_loader):
        #         image = image.to(device)
        #         output = model(image).cpu().numpy()
        #         image_flip = image_flip.to(device)
        #         output_flip = model(image_flip).cpu().numpy()
        #         output_flip = output_flip[..., ::-1]
        #         output_flip = output_flip[:, joint_flip, ...]
        #         output = (output + output_flip) / 2.0
        #         output_keypoints = soft_argmax(torch.from_numpy(output))
        #         output_keypoints = output_keypoints.numpy()

        #         acc_test_1 += total_accuracy_keypoints(output_keypoints, keypoints.numpy(), 1) * image.shape[0]
        #         acc_test_2 += total_accuracy_keypoints(output_keypoints, keypoints.numpy(), 2) * image.shape[0]
        #         acc_test_4 += total_accuracy_keypoints(output_keypoints, keypoints.numpy(), 4) * image.shape[0]
        #         num_test += image.shape[0]

        #         joints.append(output_keypoints * 4.0 / scale.numpy())
        #         image_names += image_name

        #     print("Acc Threshold 1 {:.4f}".format(acc_test_1/num_test), flush=True)
        #     print("Acc Threshold 2 {:.4f}".format(acc_test_2/num_test), flush=True)
        #     print("Acc Threshold 4 {:.4f}".format(acc_test_4/num_test), flush=True)
        #     joints = np.concatenate(joints, 0)
        #     print(joints.shape)
        #     print(len(image_names))
        #     print("save train dataset prediction results...", flush=True)
        #     save_npz(joints, image_names,  os.path.join(cfg['data']['output'], 'h36m_pred2d_cropped_train.npz'), True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, help="Configuration file to use")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    main(cfg)