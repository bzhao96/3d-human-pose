import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

subjects_train = ('S1', 'S5', 'S6', 'S7', 'S8')
subjects_test = ('S9', 'S11')
actions = ('Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog', 'WalkingTogether')
subactions = ('1', '2')
cameras = {'54138969':0, '55011271':1, '58860488':2, '60457274':3}
black_list = ('S11', 'Directions', '2', '54138969')

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
    },
]

class Human36M(Dataset):
    pose3d_root_mean = torch.tensor([0.09512221, -0.39757583,  5.1474943]).float()
    def __init__(self, data_path, pad, is_train=True):
        self.pad = pad
        self.is_train = is_train
        pose2d_file = 'pose_resnet/h36m_pred2d_train.npz' if self.is_train else 'pose_resnet/h36m_pred2d_test_ds.npz'
        pose3d_file = 'h36m_gt_train.npz' if self.is_train else 'h36m_gt_test_ds.npz'
        subjects = subjects_train if self.is_train else subjects_test
        pose2d_dict = np.load(os.path.join(data_path, pose2d_file), encoding='latin1')['pose2d'].item()
        pose3d_dict = np.load(os.path.join(data_path, pose3d_file), encoding='latin1')['pose3d'].item()
        self.pose2d = []
        self.pose3d = []
        self.camera = []
        self.sequence = []
        shift = 0
        intrinsics = []
        for intrinsic in h36m_cameras_intrinsic_params:
            intrinsics += intrinsic['focal_length']
            intrinsics += intrinsic['center']
        intrinsics = np.array(intrinsics).reshape(4, 4)
        intrinsics[:, :2] = intrinsics[:, :2] / 1000 * 2
        intrinsics[:, 2:] = intrinsics[:, 2:] / 1000 * 2 - 1

        for subject in subjects:
            for action in actions:
                for subaction in subactions:
                    for camera in cameras.keys():
                        if (subject, action, subaction, camera) == black_list:
                            continue
                        self.pose2d.append(pose2d_dict[subject][action+'-'+subaction][camera])
                        self.pose3d.append(pose3d_dict[subject][action+'-'+subaction][camera])
                        clip_length = len(self.pose2d[-1])
                        start = [shift] * self.pad + list(range(shift, shift+clip_length-self.pad))
                        mid = list(range(shift, shift+clip_length))
                        end = list(range(shift+self.pad+1, shift+clip_length+1)) + [shift+clip_length] * self.pad
                        self.sequence += list(zip(start, mid, end))
                        self.camera.append(np.repeat(np.expand_dims(intrinsics[cameras[camera]], 0), clip_length, axis=0))
                        shift += clip_length
        
        self.pose2d = np.concatenate(self.pose2d, 0)
        self.pose2d = self.pose2d / 1000 * 2 - 1
        self.pose2d[:,1:,:] = self.pose2d[:,1:,:] - self.pose2d[:,:1,:] 
        self.pose2d = self.pose2d.reshape(-1, 17 * 2)
        self.pose2d = torch.from_numpy(self.pose2d).float()

        self.pose3d = np.concatenate(self.pose3d, 0)
        self.pose3d = self.pose3d / 1000
        self.pose3d[:,1:,:] = self.pose3d[:,1:,:] - self.pose3d[:,:1,:] 
        self.pose3d = self.pose3d.reshape(-1, 17 * 3)
        self.pose3d = torch.from_numpy(self.pose3d).float()

        self.camera = np.concatenate(self.camera, 0)
        self.camera = torch.tensor(self.camera).float()
        print(self.camera.shape)

        self.sequence = np.array(self.sequence)
        self.sequence_shift = np.copy(self.sequence)
        if is_train:
            rand_ind = torch.randperm(self.sequence.shape[0])
            self.sequence_shift = self.sequence_shift[rand_ind]
        print(self.sequence.shape)

    def __getitem__(self, index):
        pose2d = self.pose2d[self.sequence[index, 0] : self.sequence[index, 2]]
        if self.sequence[index, 1] - self.sequence[index, 0] != self.pad:
            pose_pad = torch.unsqueeze(self.pose2d[self.sequence[index, 0]], 0)
            pad_num = self.pad - (self.sequence[index, 1] - self.sequence[index, 0])
            pose_pad = torch.repeat_interleave(pose_pad, torch.tensor(pad_num), 0)
            pose2d = torch.cat((pose_pad, pose2d), 0)
        elif self.sequence[index, 2] - self.sequence[index, 1] != self.pad + 1:
            pose_pad = torch.unsqueeze(self.pose2d[self.sequence[index, 2] - 1], 0)
            pad_num = self.pad - (self.sequence[index, 2] - self.sequence[index, 1]) + 1
            pose_pad = torch.repeat_interleave(pose_pad, torch.tensor(pad_num), 0)
            pose2d = torch.cat((pose2d, pose_pad), 0)

        pose3d = self.pose3d[self.sequence_shift[index, 1]]
        return pose2d, self.camera[index], pose3d
    
    def __len__(self):
        return len(self.sequence)

class Human36M_Test(Dataset):
    def __init__(self, data_path):
        self.pose2d_dict = np.load(os.path.join(data_path, 'h36m_pred2d_test.npz'), encoding='latin1')['pose2d'].item()
        self.pose3d_dict = np.load(os.path.join(data_path, 'h36m_gt3d_test.npz'), encoding='latin1')['pose3d'].item()
        self.pose2d = []
        self.pose3d = []
        for subject in subjects_test:
            for action in actions:
                for subaction in subactions:
                    for camera in cameras:
                        if (subject, action, subaction, camera) == black_clip:
                            continue
                        self.pose2d.append(self.pose2d_dict[subject][action+'-'+subaction][camera])
                        self.pose3d.append(self.pose3d_dict[subject][action+'-'+subaction][camera])
        
        self.pose2d = np.concatenate(self.pose2d, 0)
        self.pose2d = self.pose2d / 1000 * 2 - 1
        self.pose2d = self.pose2d[:,1:,:] - self.pose2d[:,:1,:] 
        self.pose2d = self.pose2d.reshape(-1, 17 * 2)
        self.pose2d = torch.from_numpy(self.pose2d).float()

        self.pose3d = np.concatenate(self.pose3d, 0)
        self.pose3d = self.pose3d / 1000
        self.pose3d = self.pose3d[:,1:,:] - self.pose3d[:,:1,:] 
        self.pose3d = self.pose3d.reshape(-1, 17 * 3)
        self.pose3d = torch.from_numpy(self.pose3d).float()
    
    def __getitem__(self, index):
        return self.pose2d[index], self.pose3d[index]
    
    def __len__(self):
        return len(self.pose2d)

if __name__ == '__main__':
    train_D_fake = DataLoader(
        dataset=Train_Generator('h36m_train.npz', True, 4),
        batch_size=256,
        shuffle=True,
    )
                        
                        