import os
import glob
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers

from repnet.model import Regression as single_reg
from video_gan_absolute.model import Regression as video_reg_absolute
from repnet.post_processing import unnormalize, get_transformation

h36m_index = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
h36m_joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 
               'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
cameras = ('54138969', '55011271', '58860488', '60457274')
joints_right_2d = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
PAD = 13

def main(args):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model_G_single = single_reg().to(device)
    model_G_root = video_reg_absolute(pad=PAD, input_size=34, output_size=3).to(device)
    model_G_video = video_reg_absolute(pad=PAD, input_size=32, output_size=48).to(device)
    print("load saved model...")
    checkpoint_single = torch.load('saved_model/repnet_resnet.tar')
    model_G_single.load_state_dict(checkpoint_single['model_G'])
    checkpoint_video = torch.load('saved_model/video_absolute_resnet.tar')
    model_G_video.load_state_dict(checkpoint_video['model_G_pose'])
    model_G_root.load_state_dict(checkpoint_video['model_G_root'])

    print("load data...")
    img_path = glob.glob(os.path.join(args.viz_data, 'test_all', args.viz_subject, args.viz_action, "imageSequence", cameras[int(args.viz_camera)], "*.jpg"))
    img_path = img_path[300:800]
    pose2d_dict = np.load('data/pose_resnet/h36m_pred2d_test_all.npz', encoding='latin1')['pose2d'].item()
    pose2d = pose2d_dict[args.viz_subject][args.viz_action][cameras[int(args.viz_camera)]]
    length = len(pose2d)
    pose3d_dict = np.load('data/h36m_gt_test_all.npz', encoding='latin1')['pose3d'].item()
    pose3d = pose3d_dict[args.viz_subject][args.viz_action][cameras[int(args.viz_camera)]]

    pose2d_input =  pose2d / 1000 * 2 - 1
    pose2d_input_absolute = np.copy(pose2d_input)
    pose2d_input_absolute = torch.from_numpy(pose2d_input_absolute.reshape(-1, 17*2)).float()
    pose2d_absolute_squence = gen_sequence(pose2d_input_absolute)
    pose2d_absolute_squence = pose2d_absolute_squence[300:800].to(device)

    pose2d_input_root = pose2d_input[300:800,:1,:]
    pose2d_input = pose2d_input[:,1:,:] - pose2d_input[:,:1,:]
    pose2d_input = torch.from_numpy(pose2d_input.reshape(-1, 16*2)).float()
    pose2d_sequence = gen_sequence(pose2d_input)

    pose2d = pose2d[300:800]
    pose2d_input = pose2d_input[300:800].to(device)
    pose2d_sequence = pose2d_sequence[300:800].to(device)

    pose3d = pose3d / 1000
    # pose3d = pose3d[:,1:,:] - pose3d[:,:1,:]
    pose3d = pose3d[300:800]

    print("run model, predict...")
    with torch.no_grad():
        # single frame model predict
        pose3d_pred = model_G_single(pose2d_input)[:, :48]
        outputs_unnorm = unnormalize(pose3d_pred.cpu().numpy())
        labels_unnorm = pose3d.reshape(-1, 17*3)*1000
        output_single = outputs_unnorm

        # video gan model predict
        pose3d_pred_root = model_G_root(pose2d_absolute_squence).view(-1, 1, 3)
        pose3d_pred = model_G_video(pose2d_sequence).view(-1, 16, 3)
        pose3d_pred = torch.cat((pose3d_pred_root, pose3d_pred_root+pose3d_pred), 1)
        output_video = pose3d_pred.cpu().numpy()

        camera = np.array([[1145.51, 1144.77, 514.96, 501.88]])
        camera = np.expand_dims(np.repeat(camera, 17, axis=0), 0)
        pose2d_rep = output_video[..., :2] / output_video[..., 2:]
        pose2d_rep = pose2d_rep * camera[..., :2] + camera[..., 2:]
        pose2d_rep = pose2d_rep.astype(np.int32)

        print(output_video[0])
        
        render_animation(img_path, pose2d_rep, labels_unnorm.reshape(-1, 17, 3)/1000, output_single.reshape(-1, 17, 3)/1000, output_video.reshape(-1, 17, 3))

def gen_sequence(pose2d):
    print(pose2d.shape)
    start_frame = pose2d[0]
    start_pad = torch.repeat_interleave(start_frame.unsqueeze(0), PAD, 0)
    end_frame = pose2d[-1]
    end_pad = torch.repeat_interleave(end_frame.unsqueeze(0), PAD, 0)
    pose2d_pad = torch.cat((start_pad, pose2d, end_pad), 0)
    pose2d_sequence = []
    for i in range(pose2d.shape[0]):
        pose2d_sequence.append(pose2d_pad[range(i, i+(2*PAD+1), 1)])
    pose2d_sequence = torch.stack(pose2d_sequence, 0)
    print(pose2d_sequence.shape)
    return pose2d_sequence


def render_animation(img_path, pose2d, pose3d, pose3d_single, pose3d_video):
    pose3d_single = pose3d_single - pose3d_single[:, :1, :]
    # pose3d_video = pose3d_video - pose3d_video[:, :1, :]
    print("render animation...")
    plt.ioff()
    size = 10
    limit = len(img_path)
    fig = plt.figure(figsize=(size*4, size))
    ax_in = fig.add_subplot(1, 4, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Reprojection', fontsize=30)

    ax_3d = []
    lines_3d = []
    radius = 1.7
    title = ['single frame', 'video absolute', 'ground truth']
    
    for i in range(3):
        ax = fig.add_subplot(1, 4, i+2, projection='3d')
        ax.view_init(elev=15., azim=-90)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius/2, radius/2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.set_xlabel('x label', fontsize=20)
        # ax.set_ylabel('y label', fontsize=20)
        # ax.set_zlabel('z label', fontsize=20)
        ax.dist = 7.5
        ax.set_title(title[i], fontsize=30)
        ax_3d.append(ax)
        lines_3d.append([])

    initialized = False
    imgplot = None
    points = None
    lines = []

    def update_video(i):
        nonlocal initialized, imgplot, points, lines
        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2, radius/2])
            ax.set_ylim3d([-radius/2, radius/2])
            ax.set_zlim3d([-radius/2, radius/2])
            if n == 1 or n == 2:
                ax.set_xlim3d([-radius/2-0.3, radius/2-0.3])
                ax.set_ylim3d([-radius/2+5.0, radius/2+5.0])
                ax.set_zlim3d([-radius/2, radius/2])

        colors_2d = np.full(pose2d.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        img=mpimg.imread(img_path[i])
        if not initialized:
            imgplot = ax_in.imshow(img, aspect='equal')
            points = ax_in.scatter(*pose2d[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                lines.append(ax_in.plot([pose2d[i, j, 0], pose2d[i, j_parent, 0]],
                                            [pose2d[i, j, 1], pose2d[i, j_parent, 1]], color='pink', linewidth=3))
                
                col = 'red' if joints_right_2d[j] else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = None
                    if n == 0:
                        pos = pose3d_single[i]
                    elif n == 1:
                        pos = pose3d_video[i]
                        ax.set_zlim3d([-radius/2-pos[0, 1], radius/2-pos[0, 1]])
                    else:
                        pos = pose3d[i]
                        ax.set_zlim3d([-radius/2-pos[0, 1], radius/2-pos[0, 1]])
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 2], pos[j_parent, 2]],
                                               [-pos[j, 1], -pos[j_parent, 1]], zdir='z', c=col, linewidth=3))

            initialized = True
        else:
            imgplot.set_data(img)
            points.set_offsets(pose2d[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                lines[j-1][0].set_data([pose2d[i, j, 0], pose2d[i, j_parent, 0]],
                                        [pose2d[i, j, 1], pose2d[i, j_parent, 1]])
                
                for n, ax in enumerate(ax_3d):
                    pos = None
                    if n == 0:
                        pos = pose3d_single[i]
                    elif n == 1:
                        pos = pose3d_video[i]
                        ax.set_zlim3d([-radius/2-pos[0, 1], radius/2-pos[0, 1]])
                    else:
                        pos = pose3d[i] 
                        ax.set_zlim3d([-radius/2-pos[0, 1], radius/2-pos[0, 1]])
                    lines_3d[n][j-1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j-1][0].set_ydata([pos[j, 2], pos[j_parent, 2]])
                    lines_3d[n][j-1][0].set_3d_properties([-pos[j, 1], -pos[j_parent, 1]], zdir='z')
    
    fig.tight_layout()
    anim = FuncAnimation(fig, update_video, frames=np.arange(1, 301, 1), interval=1000/50, repeat=False)
    Writer = writers['ffmpeg']
    writer = Writer(fps=50, metadata={}, bitrate=3000)
    anim.save("output.mp4", writer=writer)
    # anim.save("output.gif", dpi=80, writer='imagemagick')
    plt.close() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz-data", help="data path")
    parser.add_argument("--viz-subject", help="subject to render")
    parser.add_argument("--viz-action", help="action to render")
    parser.add_argument("--viz-camera", help="camera to render")

    args = parser.parse_args()
    main(args)
