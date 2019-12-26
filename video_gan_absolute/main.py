import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Human36M, Human36M_Test
from model import Regression, Critic, reprojection
from loss import gradient_penalty_loss, camera_loss
from post_processing import unnormalize, get_transformation

CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64
PAD = 13
ITERS = 50000 # How many generator iterations to train for

h36m_joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 
               'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

def inf_data(data_loader):
    while True:
        for data in data_loader:
            yield data

def main(args):
    np.random.seed(1335)
    torch.manual_seed(1335)
    torch.cuda.manual_seed(1335)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    train_D_loader = DataLoader(dataset=Human36M(args.data, PAD, True), batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
    train_G_loader = DataLoader(dataset=Human36M(args.data, PAD, True), batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
    test_loader = DataLoader(dataset=Human36M(args.data, PAD, False), batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
    train_D_loader = inf_data(train_D_loader)
    train_G_loader = inf_data(train_G_loader)

    model_G_pose = Regression(pad=PAD, input_size=32, output_size=48).to(device)
    model_G_root = Regression(pad=PAD, input_size=34, output_size=3).to(device)
    model_G_parameters = list(model_G_root.parameters()) + list(model_G_pose.parameters())
    model_D = Critic().to(device)
    model_D_parameters = model_D.parameters()
    optimizer_G = optim.Adam(model_G_parameters, lr=1e-4, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(model_D_parameters, lr=1e-4, betas=(0.5, 0.9))
    criterion = nn.MSELoss().to(device)
    best_error = float('inf')
    iterations = 0

    output_path = os.path.join(args.output, 'video_absolute_resnet.tar')
    if args.restore == True and os.path.exists(output_path):
        print("load saved model")
        checkpoint = torch.load(output_path)
        model_G_pose.load_state_dict(checkpoint['model_G_pose'])
        model_G_root.load_state_dict(checkpoint['model_G_root'])
        model_D.load_state_dict(checkpoint['model_D'])
        # optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        # optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        iterations = checkpoint['iteration']
        best_error = checkpoint['best_error']

    running_rep_loss = 0.0
    running_gan_loss = 0.0
    
    for iteration in range(iterations, ITERS):
        ############################
        # (1) Update D network
        ###########################
        # print("updata D network")
        model_G_pose.train()
        model_G_root.train()
        for p in model_D_parameters:
            p.requires_grad = True
        for critic_iter in range(CRITIC_ITERS):
            fake, _, real, = next(train_D_loader)
            fake = fake.to(device)
            real = real.to(device)
            model_D.zero_grad()

            output_real = -1.0 * model_D(real[..., 3:]).mean()
            output_G_pose_fake = model_G_pose(fake[..., 2:])
            output_fake = model_D(output_G_pose_fake).mean()
            gradient_penalty = gradient_penalty_loss(model_D, real[..., 3:], output_G_pose_fake, device)
            disc_cost = output_real + output_fake + gradient_penalty
            disc_cost.backward()
            optimizer_D.step()

        ############################
        # (2) Update G network
        ###########################
        # print("updata G network")
        for p in model_D_parameters:
            p.requires_grad = False
        fake, camera, _ = next(train_G_loader)
        fake = fake.to(device)
        camera = camera.to(device)
        b, p, _ = fake.shape
        fake_absolute = fake.clone().view(b, p, 17, 2)
        fake_absolute[..., 1:, :] = fake_absolute[..., 1:, :] + fake_absolute[..., :1, :]
        model_G_pose.zero_grad()
        model_G_root.zero_grad()

        output_G_pose_fake = model_G_pose(fake[..., 2:])
        output_G_root_fake = model_G_root(fake_absolute.view(b, p, -1))
        output_fake = -1 * model_D(output_G_pose_fake).mean()

        output_G_pose_fake = output_G_pose_fake.view(-1, 16, 3)
        output_G_root_fake = output_G_root_fake.view(-1, 1, 3)
        output_G_absolute = torch.cat((output_G_root_fake, output_G_root_fake + output_G_pose_fake), 1)
        pose2d_rep = reprojection(output_G_absolute, camera)
        rep_loss = 100 * criterion(pose2d_rep, fake_absolute[:, PAD, ...] * output_G_absolute[..., 2:3])
        gen_loss = output_fake + rep_loss

        running_gan_loss += output_fake.item()
        running_rep_loss += rep_loss.item()

        gen_loss.backward()
        optimizer_G.step()

        if iteration % 50 == 49:
            print('Iteration: {:d}, reprojection loss: {:.4f}, gan loss {:.4f}, '
                'total loss {:.4f}'.format(iteration + 1, running_rep_loss / 50, running_gan_loss / 50,
                 (running_rep_loss + running_gan_loss) / 50), flush=True)
            running_rep_loss = 0.0
            running_gan_loss = 0.0

        ######################
        # evaluation
        ######################
        if iteration % 500 == 499:
            model_G_root.eval()
            model_G_pose.eval()
            all_dist = []
            with torch.no_grad():
                for i, (fake, camera, real) in tqdm(enumerate(test_loader)):
                    fake = fake.to(device)
                    b, p, _ = fake.shape
                    fake_absolute = fake.clone().view(b, p, 17, 2)
                    fake_absolute[..., 1:, :] = fake_absolute[..., 1:, :] + fake_absolute[..., :1, :]
                    output_G_pose_fake = model_G_pose(fake[..., 2:]).view(-1, 16, 3)
                    output_G_root_fake = model_G_root(fake_absolute.view(b, p, -1)).view(-1, 1, 3)
                    output_G_absolute = torch.cat((output_G_root_fake, output_G_root_fake + output_G_pose_fake,), 1)
                    outputs = output_G_absolute.cpu().numpy() * 1000.0
                    real = real.view(b, 17, 3)
                    real[:, 1:, :] = real[:, 1:, :] + real[:, :1, :]
                    labels = real.numpy() * 1000.0
                    
                    # for ba in range(labels.shape[0]):
                    #     gt = labels[ba].reshape(-1, 3)
                    #     out = outputs[ba].reshape(-1, 3)
                    #     _, Z, T, b, c = get_transformation(gt, out, True)
                    #     out = (b * out.dot(T)) + c
                    #     outputs[ba, :] = out.reshape(1, 17, 3)
                    
                    distance = ((outputs - labels) ** 2)
                    distance = np.sqrt(np.sum(distance, axis=2))
                    all_dist.append(distance)
            all_dist = np.vstack(all_dist)
            joint_error = np.mean(all_dist, axis=0)
            total_error = np.mean(all_dist)
            # for joint_index in range(len(h36m_joints)):
            #     print("{}:{:.3f}".format(h36m_joints[joint_index], joint_error[joint_index]))
            print("evalutation loss: {:.4f} mm".format(total_error))

            if total_error < best_error:
                best_error = total_error
                print('saving the model...')
                torch.save(
                    {
                        'model_G_root': model_G_root.state_dict(),
                        'model_G_pose': model_G_pose.state_dict(),
                        'model_D': model_D.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'iteration': iteration + 1,
                        'best_error': best_error
                    },
                    output_path
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="data path")
    parser.add_argument("-r", "--restore", help="retore the model", action="store_true")
    parser.add_argument("-o", "--output", help="output path")
    args = parser.parse_args()
    main(args)
