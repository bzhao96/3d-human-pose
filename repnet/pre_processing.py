import os
import h5py
import numpy as np

subjects_train = ('S1', 'S5', 'S6', 'S7', 'S8')
subjects_test = ('S9', 'S11')
actions = ('Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog', 'WalkingTogether')
subactions = ('1', '2')
h36m_index = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

def absolute_to_relative(pose3d):
    root_position = pose3d[:, 0:1, :]
    return pose3d - root_position

def normalize_train(path):
    pose2d_all = []
    pose3d_all = []
    root_path = path
    for subject in subjects_train:
        for action in actions:
            for subaction in subactions:
                annot_path = os.path.join(root_path, subject, action+'-'+subaction, 'annot.h5')
                print(annot_path)
                annot_file = h5py.File(annot_path, 'r')
                pose2d = annot_file['pose/2d'][:, h36m_index, :]
                pose2d_all.append(pose2d)
                pose3d = annot_file['pose/3d'][:, h36m_index, :]
                pose3d_all.append(pose3d)
                annot_file.close()

    pose2d_all = np.concatenate(pose2d_all, axis=0)
    # pose2d_all = pose2d_all - pose2d_all[:,:1,:]
    # pose2d_head = pose2d_all[:, 10, :]
    # scale2d = np.mean(np.sqrt(np.sum(pose2d_head ** 2, axis=1)))
    # pose2d_normlized = pose2d_all / scale2d
    # pose2d_normlized = pose2d_normlized[:, 1:, :]

    pose2d_normlized = pose2d_all / 1000 * 2 - 1
    pose2d_normlized = pose2d_normlized[:,1:,:] - pose2d_normlized[:,:1,:]
    print(pose2d_normlized.shape)
    np.savetxt('data/pose2d_train_norm.txt', pose2d_normlized.reshape(-1, 16 * 2))

    pose3d_all = np.concatenate(pose3d_all, axis=0)
    # pose3d_all = pose3d_all - pose3d_all[:,:1,:]
    # pose3d_head = pose3d_all[:, 10, :]
    # scale3d = np.mean(np.sqrt(np.sum(pose3d_head ** 2, axis=1)))
    # pose3d_normlized = pose3d_all / scale3d
    # pose3d_normlized = pose3d_normlized[:, 1:, :]
    
    pose3d_normlized = pose3d_all / 1000
    pose3d_normlized = pose3d_normlized[:,1:,:] - pose3d_normlized[:,:1,:]
    print(pose3d_normlized.shape)
    np.savetxt('data/pose3d_train_norm.txt', pose3d_normlized.reshape(-1, 16 * 3))

    # scale = np.array([scale2d, scale3d])
    # np.savetxt('data/norm_scale.txt', scale)

def normalize_test(path):
    pose2d_all = []
    pose3d_all = []
    root_path = path
    for subject in subjects_test:
        for action in actions:
            for subaction in subactions:
                annot_path = os.path.join(root_path, subject, action+'-'+subaction, 'annot.h5')
                print(annot_path)
                annot_file = h5py.File(annot_path, 'r')
                pose2d = annot_file['pose/2d'][:, h36m_index, :]
                pose2d_all.append(pose2d)
                pose3d = annot_file['pose/3d'][:, h36m_index, :]
                pose3d_all.append(pose3d)
                annot_file.close()
    
    # scale2d, scale3d = np.loadtxt('data/norm_scale.txt')

    pose2d_all = np.concatenate(pose2d_all, axis=0)
    # pose2d_all = pose2d_all - pose2d_all[:,:1,:]
    # pose2d_normlized = pose2d_all / scale2d
    # pose2d_normlized = pose2d_normlized[:, 1:, :]

    pose2d_normlized = pose2d_all / 1000 * 2 - 1
    pose2d_normlized = pose2d_normlized[:,1:,:] - pose2d_normlized[:,:1,:]
    print(pose2d_normlized.shape)
    np.savetxt('data/pose2d_test_norm.txt', pose2d_normlized.reshape(-1, 16 * 2))

    pose3d_all = np.concatenate(pose3d_all, axis=0)
    # pose3d_all = pose3d_all - pose3d_all[:,:1,:]
    # pose3d_normlized = pose3d_all / scale3d
    # pose3d_normlized = pose3d_normlized[:, 1:, :]

    pose3d_normlized = pose3d_all / 1000
    pose3d_normlized = pose3d_normlized[:,1:,:] - pose3d_normlized[:,:1,:]
    print(pose3d_normlized.shape)
    np.savetxt('data/pose3d_test_norm.txt', pose3d_normlized.reshape(-1, 16 * 3))


if __name__ == "__main__":
    path = '/home/users/bin.zhao/h36m_data/downsample'
    normalize_train(path)
    normalize_test(path)
    