import os
import glob
import numpy as np 
from PIL import Image

subjects_train = ('S1', 'S5', 'S6', 'S7', 'S8')
subjects_test = ('S9', 'S11')
actions = ('Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog', 'WalkingTogether')
subactions = ('1', '2')
cameras = ('54138969', '55011271', '58860488', '60457274')
black_list = ("S11", "Directions", "2", "54138969")

root_path = '/home/users/bin.zhao/h36m_data/downsample'
file_list = []
for subject in subjects_train:
    for action in actions:
        for subaction in subactions:
            for camera in cameras:
                a = glob.glob(os.path.join(root_path, subject, action+'-'+subaction, 'imageSequence', camera, "*.jpg"))
                file_list += a[100:101]
print(len(file_list))
img_all = []
for f in file_list:
    img = Image.open(f)
    img = np.array(img, dtype=np.float) / 255.0
    img_all.append(img.reshape(-1, 3))
img_all = np.concatenate(img_all, 0)
print(img_all.shape)
mean = np.mean(img_all, 0)
std = np.std(img_all, 0)
print(mean) # 0.44487878 0.27743985 0.2616888
print(std) # 0.25466519 0.26579411 0.24354595

