import numpy as np 
import torch
import torch.nn as nn

def peak(heatmap):
    argmax = np.argmax(heatmap)
    return argmax // heatmap.shape[0], argmax % heatmap.shape[0]

def total_accuracy_heatmap(pred, gt, threshold=1):
    '''
    Arguments:
        pred: N, C, H, W
        gt: N, C, H, W
    '''
    total_correct = 0
    _, _, h, w = pred.shape
    pred = pred.reshape(-1, h, w)
    gt = gt.reshape(-1, h, w)
    for i in range(pred.shape[0]):
        pred_x, pred_y = peak(pred[i])
        # pred_x += 0.25
        # pred_y += 0.25
        gt_x, gt_y = peak(gt[i])
        correct = np.sqrt(np.square(pred_x - gt_x) + np.square(pred_y - gt_y)) <= threshold
        total_correct += correct
    return total_correct / pred.shape[0]

def total_accuracy_keypoints(pred, gt, threshold=1):
    '''
    Arguments:
        pred: N, C, 2
        gt: N, C, 2
    '''
    dist = np.sqrt(np.sum((pred - gt) ** 2, axis=2))
    total_correct = np.sum(dist < threshold)
    total_num = pred.shape[0] * pred.shape[1]
    return total_correct / total_num

def soft_argmax(heatmap):
    N, C, H, W = heatmap.shape
    alpha = 1000.0
    marginal_x = nn.functional.softmax(heatmap.sum(3) * alpha, dim=2)
    indices_x = torch.arange(H).float()
    x = marginal_x * indices_x
    x = x.sum(2)

    marginal_y = nn.functional.softmax(heatmap.sum(2) * alpha, dim=2)
    indices_y = torch.arange(W).float()
    y = marginal_y * indices_y
    y = y.sum(2)

    coords = torch.stack([y, x], dim=2)
    return coords

