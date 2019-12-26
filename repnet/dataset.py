import os
import numpy as np
import torch
from torch.utils.data import Dataset

subjects_train = ('S1', 'S5', 'S6', 'S7', 'S8') 
subjects_test = ('S9', 'S11')
actions = ('Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog', 'WalkingTogether')
subactions = ('1', '2')
cameras = ('54138969', '55011271', '58860488', '60457274')
black_list = ("S11", "Directions", "2", "54138969")


class Human36M(Dataset):
    def __init__(self, data_path, is_train=True):
        file2d_name = 'pose_resnet/h36m_pred2d_train.npz' if is_train else 'pose_resnet/h36m_pred2d_test.npz'
        file3d_name = 'h36m_gt_train.npz' if is_train else 'h36m_gt_test.npz'
        self.data2d_path = os.path.join(data_path, file2d_name)
        self.data3d_path = os.path.join(data_path, file3d_name)
        self.subjects = subjects_train if is_train else subjects_test
        self.pose2d, self.pose3d = self.read_npz()
        print(self.pose2d.shape)
        print(self.pose3d.shape)
        
        if is_train:
            rand_ind = torch.randperm(self.pose2d.shape[0])
            self.pose2d = self.pose2d[rand_ind, :]

    def __getitem__(self, index):
        return self.pose2d[index], self.pose3d[index]
    
    def __len__(self):
        return len(self.pose2d)
    
    def read_npz(self):
        pose2d = []
        pose3d = []
        pose2d_dict = np.load(self.data2d_path, encoding='latin1')['pose2d'].item()
        pose3d_dict = np.load(self.data3d_path, encoding='latin1')['pose3d'].item()
        for subject in self.subjects:
            for action in actions:
                for subaction in subactions:
                    for camera in cameras:
                        if (subject, action, subaction, camera) == black_list:
                            continue
                        pose2d.append(pose2d_dict[subject][action+'-'+subaction][camera])
                        pose3d.append(pose3d_dict[subject][action+'-'+subaction][camera])
        
        pose2d = np.concatenate(pose2d, 0)
        pose2d = pose2d / 1000 * 2 - 1
        pose2d = pose2d[:, 1:, :] - pose2d[:, :1, :]
        pose2d = pose2d.reshape(-1, 16 * 2)
        pose2d = torch.from_numpy(pose2d).float()

        pose3d = np.concatenate(pose3d, 0)
        pose3d = pose3d / 1000
        pose3d = pose3d[:, 1:, :] - pose3d[:, :1, :]
        pose3d = pose3d.reshape(-1, 16 * 3)
        pose3d = torch.from_numpy(pose3d).float()
        return pose2d, pose3d

