import os
import h5py
import torch
import torchvision
import numpy as np 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import glob
from PIL import Image
import random

subjects_train = ('S1', 'S5', 'S6', 'S7', 'S8')
subjects_test = ('S9', 'S11')
actions = ('Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog', 'WalkingTogether')
subactions = ('1', '2')
cameras = ('54138969', '55011271', '58860488', '60457274')
black_list = ("S11", "Directions", "2", "54138969")

class H36M_Loader(Dataset):
    def __init__(self, root_path, is_train=True, img_size=(256, 256)):
        self.mean = [0.44487878, 0.27743985, 0.2616888]
        self.std = [0.25466519, 0.26579411, 0.24354595]
        self.joint_flip = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        self.root_path = root_path
        self.is_train = is_train
        self.img_size = img_size
        self.heatmap_size = (img_size[0]/4, img_size[1]/4)
        self.sigma = 2.0
        self.subjects = subjects_train if is_train else subjects_test
        self.trans_to_tensor = torchvision.transforms.ToTensor()
        self.trans_to_image = torchvision.transforms.ToPILImage()

        self.file = []
        self.pose = []
        for subject in self.subjects:
            for action in actions:
                for subaction in subactions:
                    for camera in cameras:
                        if (subject, action, subaction, camera) == black_list:
                            continue
                        self.file += glob.glob(os.path.join(self.root_path, subject, action+'-'+subaction, camera, "*.jpg"))
                    pose_file = h5py.File(os.path.join(self.root_path, subject, action+'-'+subaction, "annot.h5"))
                    self.pose.append(pose_file["pose2d"])
        self.pose = np.concatenate(self.pose, 0)
        self.pose = torch.from_numpy(self.pose).float()
        print(self.pose.shape)
        print(len(self.file))

    def __getitem__(self, index):
        image = Image.open(self.file[index])
        scale = [self.img_size[0] / image.size[0], self.img_size[1] / image.size[1]]
        scale = torch.tensor(scale).view(1, 2).float()
        image = image.resize(self.img_size, Image.ANTIALIAS)
        pose = self.pose[index] * scale
        
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                pose[:, 0] = self.img_size[0] - 1 - pose[:, 0]
                pose = pose[self.joint_flip]
            heatmap = self.gen_heatmap(pose)
            # random rotate
            # angle = random.gauss(0.0, 5.0)
            # image = TF.rotate(image, angle)
            # heatmap = [self.trans_to_image(hm) for hm in heatmap]
            # heatmap = [TF.rotate(hm, angle) for hm in heatmap]
            # heatmap = [self.trans_to_tensor(hm) for hm in heatmap]

            image = self.trans_to_tensor(image).float()
            image = TF.normalize(image, self.mean, self.std)
            heatmap = torch.cat(heatmap, 0).float()
            return image, heatmap, scale, self.file[index]
        
        else:
            image_flip = TF.hflip(image)
            image_flip = self.trans_to_tensor(image_flip).float()
            image_flip = TF.normalize(image_flip, self.mean, self.std)
            
            image = self.trans_to_tensor(image).float()
            image = TF.normalize(image, self.mean, self.std)
            
            return image, image_flip, pose / 4.0, scale, self.file[index]

    def __len__(self):
        return len(self.file)
    
    def gen_heatmap(self, keypoints):
        heatmap = []
        for keypoint in keypoints:
            x = keypoint[1] / 4.0
            y = keypoint[0] / 4.0
            heatmap.append(self.gaussian(self.sigma, (x,y)))
        return heatmap

    def gaussian(self, sigma, center):
        x, y = torch.meshgrid(torch.arange(self.heatmap_size[0]), torch.arange(self.heatmap_size[1]))
        c_x, c_y = center
        hm = torch.exp(-((x-c_x)**2 + (y-c_y)**2)/(2*sigma**2))
        return hm.unsqueeze(0)


if __name__ == "__main__":
    root_path = "/home/users/bin.zhao/h36m_data/cropped"
    torch.manual_seed(1333)
    train_loader = DataLoader(H36M_Loader(root_path), batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    # for image, heatmap, _ in train_loader:
    #     image = (image + 1.0) / 2.0
    #     for i in range(16):
    #         img = TF.to_pil_image(image[i])
    #         plt.figure()
    #         plt.imshow(img)
    #         for hm in heatmap[i].numpy():
    #             x, y = peak(hm)
    #             plt.scatter(4 * y, 4 * x, c='red', marker='.')
    #         plt.savefig('output_{}.jpg'.format(i))
    #         plt.close()
    #     exit()

