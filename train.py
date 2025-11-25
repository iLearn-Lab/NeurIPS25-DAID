#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import model
from detection_layers.modules import MultiBoxLoss
from dataset import DeepfakeDataset
from lib.util import load_config, update_learning_rate, my_collate
from tqdm import tqdm
from test import valiation


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_train.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.', default=None)
    args = parser.parse_args()
    return args


def save_checkpoint(net, opt, save_path, epoch_num):
    os.makedirs(save_path, exist_ok=True)
    module = net.module
    model_state_dict = OrderedDict()
    for k, v in module.state_dict().items():
        model_state_dict[k] = torch.tensor(v, device="cpu")

    opt_state_dict = {}
    opt_state_dict['param_groups'] = opt.state_dict()['param_groups']
    opt_state_dict['state'] = OrderedDict()
    for k, v in opt.state_dict()['state'].items():
        opt_state_dict['state'][k] = {}
        opt_state_dict['state'][k]['step'] = v['step']
        if 'exp_avg' in v:
            opt_state_dict['state'][k]['exp_avg'] = torch.tensor(v['exp_avg'], device="cpu")
        if 'exp_avg_sq' in v:
            opt_state_dict['state'][k]['exp_avg_sq'] = torch.tensor(v['exp_avg_sq'], device="cpu")

    checkpoint = {
        'network': model_state_dict,
        'opt_state': opt_state_dict,
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}.pkl')


def load_checkpoint(ckpt, net, opt, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'].items():
        name = "module." + k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    opt.load_state_dict(checkpoint['opt_state'])
    base_epoch = int(checkpoint['epoch']) + 1
    return net, opt, 0


def validation_epo(epoch, net):
    cfg_path = './configs/caddm_test.cfg'
    cfg = load_config(cfg_path)

    net.eval()
    dfdc_auc, dfdc_acc = valiation(cfg, net, cfg['dataset']['img_path_dfdc'], cfg['dataset']['name_dfdc'],
                                   cfg['dataset']['ld_path_dfdc'],
                                   cfg['dataset']['img_csv_list_dfdc'])
    dfd_auc, dfd_acc = valiation(cfg, net, cfg['dataset']['img_path_dfd'], cfg['dataset']['name_dfd'],
                                 cfg['dataset']['ld_path_dfd'],
                                 cfg['dataset']['img_csv_list_dfd'])
    celeb_auc, celeb_acc = valiation(cfg, net, cfg['dataset']['img_path_celeb'], cfg['dataset']['name_celeb'],
                                     cfg['dataset']['ld_path_celeb'],
                                     cfg['dataset']['img_csv_list_celeb'])

    wandb.log({
        'epoch': epoch,
        'dfdc_auc': dfdc_auc,
        'dfdc_acc': dfdc_acc,
        'celeb_auc': celeb_auc,
        'celeb_acc': celeb_acc,
        'dfd_auc': dfd_auc,
        'dfd_acc': dfd_acc,
    })


def train():
    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)

    run = wandb.init(project=cfg['general']['exp_name'], name=cfg['general']['exp_id'])

    # init model.
    net = model.get(backbone=cfg['model']['backbone'], svd=cfg['general']['svd'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net)

    # loss init
    det_criterion = MultiBoxLoss(
        cfg['det_loss']['num_classes'],
        cfg['det_loss']['overlap_thresh'],
        cfg['det_loss']['prior_for_matching'],
        cfg['det_loss']['bkg_label'],
        cfg['det_loss']['neg_mining'],
        cfg['det_loss']['neg_pos'],
        cfg['det_loss']['neg_overlap'],
        cfg['det_loss']['encode_target'],
        cfg['det_loss']['use_gpu']
    )
    criterion = torch.nn.CrossEntropyLoss(reduction=cfg['train']['reduction'])
    criterion_cos = torch.nn.CosineEmbeddingLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

    # load checkpoint if given
    base_epoch = 0
    # args.ckpt = "./checkpoints/effi_4_w_aug/epoch_10.pkl"
    if args.ckpt:
        print('Load ckpt from %s' % args.ckpt)
        net, optimzer, base_epoch = load_checkpoint(args.ckpt, net, optimizer, device)

    # get training data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    train_dataset = DeepfakeDataset('train', cfg)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=4,
                              collate_fn=my_collate
                              )

    # start trining.
    if cfg['train']['strategy'] != 'remove':
        for epoch in range(0, cfg['train']['epoch_num']):
            net.train()
            index = 0
            for batch_data, batch_labels in tqdm(train_loader):
                index += 1
                lr = update_learning_rate(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                labels, location_labels, confidence_labels, gender, race = batch_labels
                labels = labels.long().to(device)
                location_labels = location_labels.to(device)
                confidence_labels = confidence_labels.long().to(device)

                optimizer.zero_grad()
                locations, confidence, outputs, return_final_cls_feat = net(batch_data)
                loss_end_cls = criterion(outputs, labels)
                loss_l, loss_c = det_criterion(
                    (locations, confidence),
                    confidence_labels, location_labels
                )
                acc = sum(outputs.max(-1).indices == labels).item() / labels.shape[0]
                det_loss = 0.1 * (loss_l + loss_c)
                loss = det_loss + loss_end_cls
                loss.backward()

                torch.nn.utils.clip_grad_value_(net.parameters(), 2)
                optimizer.step()
                outputs = [
                    "e:{},iter: {}".format(epoch, index),
                    "acc: {:.2f}".format(acc),
                    "loss: {:.8f} ".format(loss.item()),
                    "lr:{:.4g}".format(lr),
                ]
                if index % 100 == 0:
                    print(" ".join(outputs))
                    wandb.log({
                        'epoch': epoch,
                        'ite': index,
                        'loss': loss.item(),
                        'acc': acc,
                    })
            validation_epo(epoch, net)
            if epoch % 5 == 0:
                save_checkpoint(net, optimizer,
                                cfg['model']['save_path'],
                                epoch)
    else:
        print('using remove strategy')
        validation_epo(0, net)
        for epoch in range(0, cfg['train']['epoch_num']):
            net.train()
            index = 0
            for batch_data, batch_labels, contras_batch_data, contras_batch_labels in tqdm(train_loader):
                index += 1
                lr = update_learning_rate(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                labels, location_labels, confidence_labels, gender, race, weight, group_id = batch_labels
                labels = labels.long().to(device)
                location_labels = location_labels.to(device)
                confidence_labels = confidence_labels.long().to(device)
                weight = weight.to(device)

                contras_labels, contras_location_labels, contras_confidence_labels, contras_gender, contras_race, contras_weight, contras_group_id = contras_batch_labels
                contras_labels = contras_labels.long().to(device)
                contras_location_labels = contras_location_labels.to(device)
                contras_weight = contras_weight.to(device)
                contras_confidence_labels = contras_confidence_labels.long().to(device)

                optimizer.zero_grad()
                locations, confidence, outputs, final_cls_feat = net(batch_data, group_id)
                contras_locations, contras_confidence, contras_outputs, contras_final_cls_feat = net(contras_batch_data, contras_group_id)

                if cfg['train']['reduction'] == 'none':
                    loss_end_cls = criterion(outputs, labels)
                    loss_end_cls_weighted = loss_end_cls * weight
                    loss_end_cls_weighted = loss_end_cls_weighted.mean()

                    loss_end_cls_contras = criterion(contras_outputs, contras_labels)
                    loss_end_cls_contras_weighted = loss_end_cls_contras * contras_weight
                    loss_end_cls_contras_weighted = loss_end_cls_contras_weighted.mean()

                    loss_end_cls = loss_end_cls_weighted + loss_end_cls_contras_weighted

                else:
                    loss_end_cls = criterion(outputs, labels) + criterion(contras_outputs, contras_labels)

                loss_l, loss_c = det_criterion(
                    (locations, confidence),
                    confidence_labels, location_labels
                )

                contras_loss_l, contras_loss_c = det_criterion(
                    (contras_locations, contras_confidence),
                    contras_confidence_labels, contras_location_labels
                )

                target_contras = torch.ones(final_cls_feat.size(0)).to(final_cls_feat.device)
                loss_cos = criterion_cos(final_cls_feat, contras_final_cls_feat, target_contras)
                acc = sum(outputs.max(-1).indices == labels).item() / labels.shape[0]

                det_loss = 0.05 * (loss_l + loss_c + contras_loss_c + contras_loss_l)

                ortho_loss = net.module.orthogonality_regularization()

                loss = det_loss + loss_end_cls + 0.7 * loss_cos + ortho_loss * 0.2
                loss.backward()

                torch.nn.utils.clip_grad_value_(net.parameters(), 2)
                optimizer.step()
                outputs = [
                    "e:{},iter: {}".format(epoch, index),
                    "acc: {:.2f}".format(acc),
                    "loss: {:.8f} ".format(loss.item()),
                    "lr:{:.4g}".format(lr),
                ]
                print(" ".join(outputs))
                if index % 100 == 0:
                    print(" ".join(outputs))
                    wandb.log({
                        'epoch': epoch,
                        'ite': index,
                        'loss': loss.item(),
                        'loss_cos': loss_cos.item(),
                        'loss_det': det_loss.item(),
                        'loss_ortho': ortho_loss.item(),
                        'acc': acc,
                    })
            validation_epo(epoch, net)
            if epoch % 5 == 0:
                save_checkpoint(net, optimizer,
                                cfg['model']['save_path'],
                                epoch)


if __name__ == "__main__":
    train()

# vim: ts=4 sw=4 sts=4 expandtab
