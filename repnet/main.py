import os
import numpy as np 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Human36M
from model import Regression, Critic, reprojection
from loss import gradient_penalty_loss, camera_loss
from post_processing import unnormalize, get_transformation

CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64
ITERS = 100000 # How many generator iterations to train for

h36m_joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 
               'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']


def inf_data(data_loader):
    while True:
        for data in data_loader:
            yield data

def main(args):
    np.random.seed(1333)
    torch.manual_seed(1333)
    torch.cuda.manual_seed(1333)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)
    train_D_loader = DataLoader(dataset=Human36M(args.data), batch_size=BATCH_SIZE, shuffle=True)
    train_G_loader = DataLoader(dataset=Human36M(args.data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=Human36M(args.data, False), batch_size=BATCH_SIZE, shuffle=False)
    train_D_loader = inf_data(train_D_loader)
    train_G_loader = inf_data(train_G_loader)

    model_G = Regression().to(device)
    model_D = Critic().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer_G = optim.Adam(model_G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(model_D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    best_error = float('inf')
    iterations = 0

    output_path = os.path.join(args.output, 'repnet_resnet.tar')
    if args.restore and os.path.exists(output_path):
        print('load saved model')
        checkpoint = torch.load(output_path)
        model_G.load_state_dict(checkpoint['model_G'])
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
        model_G.train()
        for p in model_D.parameters():
            p.requires_grad = True
        for critic_iter in range(CRITIC_ITERS):
            fake, real = next(train_D_loader)
            fake = fake.to(device)
            real = real.to(device)
            model_D.zero_grad()

            output_real = -1.0 * model_D(real).mean()
            output_G_fake = model_G(fake)
            output_fake = model_D(output_G_fake[:, :48]).mean()
            gradient_penalty = gradient_penalty_loss(model_D, real, output_G_fake[:, :48], device)
            disc_cost = output_real + output_fake + gradient_penalty
            disc_cost.backward()
            optimizer_D.step()

        ############################
        # (2) Update G network
        ###########################
        # print("updata G network")

        for p in model_D.parameters():
            p.requires_grad = False
        fake, _ = next(train_G_loader)
        fake = fake.to(device)
        model_G.zero_grad()
        output_G_fake = model_G(fake)
        pose2d_rep = reprojection(output_G_fake)
        rep_loss = 100 * 100 * criterion(pose2d_rep, fake)  #increase weight gradually
        camera_error = camera_loss(output_G_fake[:, 48:], device).mean()
        output_fake = -1.0 * model_D(output_G_fake[:, :48]).mean()
        gen_loss = rep_loss + output_fake + camera_error

        running_rep_loss += rep_loss.item()
        running_gan_loss += output_fake.item()

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
            model_G.eval()
            all_dist = []
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(test_loader):
                    outputs = model_G(inputs.to(device))
                    outputs = outputs[:,:48].cpu().numpy()
                    outputs_unnorm = unnormalize(outputs)
                    labels_unnorm = unnormalize(labels.numpy())

                    # for ba in range(labels_unnorm.shape[0]):
                    #     gt = labels_unnorm[ba].reshape(-1, 3)
                    #     out = outputs_unnorm[ba].reshape(-1, 3)
                    #     _, Z, T, b, c = get_transformation(gt, out, True)
                    #     out = (b * out.dot(T)) + c
                    #     outputs_unnorm[ba, :] = out.reshape(1, 51)
                    
                    distance = ((outputs_unnorm - labels_unnorm) ** 2).reshape(-1, 17, 3)
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
                torch.save(
                    {
                        'model_G': model_G.state_dict(),
                        'model_D': model_D.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'iteration': iteration + 1,
                        'best_error': best_error
                    },
                    output_path
                )
                print('model saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="data path")
    parser.add_argument("-r", "--restore", help="retore the model", action="store_true")
    parser.add_argument("-o", "--output", help="output path")
    args = parser.parse_args()
    main(args)








