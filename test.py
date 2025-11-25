#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, accuracy_score
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import model
from detection_layers.modules import MultiBoxLoss
from dataset import DeepfakeDataset
from lib.util import load_config, update_learning_rate, my_collate, get_video_auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from torchvision.transforms.functional import resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test.cfg')
    args = parser.parse_args()
    return args


def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module." + k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net

def valiation(cfg, net, dataset, name, ld_path, csv_path):
    cfg['dataset']['img_path'] = dataset
    cfg['dataset']['name'] = name
    cfg['dataset']['ld_path'] = ld_path
    cfg['dataset']['img_csv_list'] = csv_path
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    test_dataset = DeepfakeDataset('test', cfg)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=False, num_workers=4,
                             )

    # start testing.
    frame_pred_list = list()
    frame_label_list = list()

    for batch_data, batch_labels in tqdm(test_loader):
        labels = batch_labels
        labels = labels.long()

        outputs = net(batch_data)
        outputs = outputs[:, 1]

        frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
        frame_label_list.extend(labels.detach().cpu().numpy().tolist())

    f_auc = roc_auc_score(frame_label_list, frame_pred_list)
    pred_list = [0 if i < 0.5 else 1 for i in frame_pred_list]
    f_acc = accuracy_score(frame_label_list, pred_list)

    print(f"Frame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"Frame-ACC of {cfg['dataset']['name']} is {f_acc:.4f}")
    # print(f"Video-AUC of {cfg['dataset']['name']} is {v_auc:.4f}")

    wandb.log({
        '{}_auc'.format(name): f_auc,
        '{}_acc'.format(name): f_acc,
    })

    return f_auc, f_acc

def test():
    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)
    run = wandb.init(project=cfg['general']['exp_name'], name=cfg['general']['exp_id'])

    # init model.
    net = model.get(backbone=cfg['model']['backbone'], svd=cfg['general']['svd'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net)
    net.eval()
    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)

    valiation(cfg, net, cfg['dataset']['name_dfdc'], cfg['dataset']['name_dfdc'], cfg['dataset']['ld_path_dfdc'], cfg['dataset']['img_csv_list_dfdc'])
    valiation(cfg, net, cfg['dataset']['name_dfd'], cfg['dataset']['name_dfd'], cfg['dataset']['ld_path_dfd'], cfg['dataset']['img_csv_list_dfd'])
    valiation(cfg, net, cfg['dataset']['name_celeb'], cfg['dataset']['name_celeb'], cfg['dataset']['ld_path_celeb'], cfg['dataset']['img_csv_list_celeb'])


if __name__ == "__main__":
    test()

# vim: ts=4 sw=4 sts=4 expandtab
