import os
import time
import yaml
import argparse
import numpy as np 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

from dataset import H36M_Loader
from model import *
from utils import total_accuracy_keypoints, total_accuracy_heatmap, soft_argmax


def main(cfg):
    np.random.seed(cfg['seeds'])
    torch.manual_seed(cfg['seeds'])
    torch.cuda.manual_seed(cfg['seeds'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1, 2, 3]
    
    train_loader = DataLoader(
        H36M_Loader(cfg['data']['input'], img_size=cfg['model']['image_size']), 
        batch_size=cfg['training']['batch_size'], 
        num_workers=cfg['training']['n_workers'],
        shuffle=True,
        )
    test_loader = DataLoader(
        H36M_Loader(cfg['data']['input'], False, img_size=cfg['model']['image_size']), 
        batch_size=cfg['training']['batch_size'], 
        num_workers=cfg['training']['n_workers'],
        )
    
    model = get_pose_net(cfg).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 7], gamma=0.2)
    best_acc = 0.0

    print("load saved model", flush=True)
    checkpoint = torch.load('pose_resnet_101_384x384.pth.tar')
    del checkpoint['final_layer.weight']
    del checkpoint['final_layer.bias']
    model.load_state_dict(checkpoint, strict=False)

    # checkpoint = torch.load('../saved_model/pose_resnet_101_384x384.tar')
    # model.load_state_dict(checkpoint['model'])


    model = nn.DataParallel(model, device_ids)
    model.train()

    for epoch in range(cfg['training']['epochs']):
        print_interval = cfg['training']['print_interval']
        val_interval = cfg['training']['val_interval']
        accuracy = 0.0
        running_loss = 0.0
        total_time = 0.0
        end_iter = len(train_loader.dataset) // cfg['training']['batch_size']
        print("Epoch {}".format(epoch+1), flush=True)
        for param_group in optimizer.param_groups:
            print('learning rate:{:.8f}'.format(param_group['lr']))
        for i, (image, heatmap) in enumerate(train_loader):
            start_time = time.time()
            image = image.to(device)
            heatmap = heatmap.to(device)

            model.zero_grad()
            output = model(image)
            loss = criterion(output, heatmap)
            running_loss += loss.item()
            accuracy_iter = total_accuracy_heatmap(output.cpu().detach().numpy(), heatmap.cpu().numpy())
            accuracy += accuracy_iter
            loss.backward()
            optimizer.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            if i % print_interval == (print_interval - 1):
                print("Epoch: {}, Batch: {},  Avg Acc: {:.4f}, Avg Loss: {:.6f}, Avg Time: {:.4f}s".format(epoch+1, 
                       i+1, accuracy/print_interval, running_loss/print_interval, total_time/print_interval), flush=True)
                accuracy = 0.0
                running_loss = 0.0
                total_time = 0.0

            if i % val_interval == (val_interval - 1) or i == end_iter:
                model.eval()
                acc_test_1 = 0.0
                acc_test_2 = 0.0
                acc_test_4 = 0.0
                num_test = 0
                with torch.no_grad():
                    joint_flip = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
                    for image, image_flip, keypoints in test_loader:
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
                    print("Epoch {}, Batch {}".format(epoch+1, i+1), flush=True)
                    print("Acc Threshold 1 {:.4f}".format(acc_test_1/num_test), flush=True)
                    print("Acc Threshold 2 {:.4f}".format(acc_test_2/num_test), flush=True)
                    print("Acc Threshold 4 {:.4f}".format(acc_test_4/num_test), flush=True)
                    if acc_test_4/num_test > best_acc:
                        best_acc = acc_test_4/num_test
                        torch.save(
                            {
                                'model': model.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'best_acc': best_acc,
                            },
                            os.path.join(cfg['data']['output'], 'pose_resnet_101_384x384.tar')
                        )
                        print("model saved", flush=True)
                    print("")
                model.train()
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, help="Configuration file to use")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    main(cfg)
