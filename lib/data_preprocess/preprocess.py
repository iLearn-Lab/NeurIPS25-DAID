#!/usr/bin/env python3
import argparse

import os
import cv2
import torch
import random
import numpy as np

from functools import lru_cache
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from .mfs import multi_scale_facial_swap
from .augmentor import add_noise, resize_aug, image_h_mirror
from .cropface import get_align5p, align_5p

from detection_layers.box_utils import match
from detection_layers import PriorBox

Prior = None


def get_prior(config):
    global Prior
    if Prior is None:
        Prior = PriorBox(config['adm_det'])


def label_assign(bboxs, config, genuine=False):

    global Prior
    get_prior(config)

    labels = torch.zeros(bboxs.shape[0],)
    defaults = Prior.forward().data  # information of priors

    if genuine:
        return np.zeros(defaults.shape), np.zeros(defaults.shape[0], )

    # anchor matching by iou.

    loc_t = torch.zeros(1, defaults.shape[0], 4)
    conf_t = torch.zeros(1, defaults.shape[0])

    match(
        0.5, torch.Tensor(bboxs), defaults,
        [0.1, 0.2], labels, loc_t, conf_t, 0)

    loc_t, conf_t = np.array(loc_t)[0, ...], np.array(conf_t)[0, ...]

    if loc_t.max() > 10**5:
        return None, 'prior bbox match err. bias is inf!'

    return loc_t, conf_t

def add_mask_a(image, mask):
    # empty = Image.new('RGBA', (image.size), 255)
    # mask = Image.fromarray(np.uint8(mask*255), 'L')
    # new_image = PIL.Image.composite(image, image, mask)
    image = np.array(image)
    new_image = image
    for k in mask:
        new_image[k[0][1]][k[0][0]][0] = random.randint(0, 255)
        new_image[k[0][1]][k[0][0]][1] = random.randint(0, 255)
        new_image[k[0][1]][k[0][0]][2] = random.randint(0, 255)
        # new_image[k[0][0]][k[0][1]][0] = 0
        # new_image[k[0][0]][k[0][1]][1] = 0
        # new_image[k[0][0]][k[0][1]][2] = 0
    return new_image


def des_mask(progress, mask):
    mask_dict = {}
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] > 0.00:
                mask_dict[(i, j)] = mask[i][j]
    mask_per = sorted(mask_dict.items(), key=lambda kv: (kv[1], kv[0]))
    mask_per.reverse()
    alpha = progress[0]
    mask_num = int(alpha * len(mask_per))
    mask_point_alpha = mask_per[:mask_num]
    return mask_point_alpha

def finetune_mask(fusion_heatmap, progress):
    mask_finetune = des_mask(progress, fusion_heatmap)
    return mask_finetune

def add_mask(img, path):
    img_name = os.path.basename(path)
    video_name = os.path.basename(os.path.dirname(path))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    classes = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    mask_path = '/root/autodl-tmp/Deepfake_Dataset/heat_map_fusion'
    m_path = os.path.join(mask_path, 'train', classes, dataset_name, video_name,
                          "%s.npy" % img_name)
    #
    mask = np.load(m_path)
    sample_resize = cv2.resize(img, mask.shape)
    progress = [0.0 if random.randint(0, 100) < 50 else random.uniform(0.0, 1.0)]
    # print(progress)
    if progress[0] != 0.0:
        mask_finetune = finetune_mask(fusion_heatmap=mask, progress=progress)
    else:
        mask_finetune = {}
    m = mask_finetune

    mask_image = add_mask_a(sample_resize, m)
    mask_image = cv2.resize(mask_image, (224, 224))
    return mask_image


def prepare_train_input(targetRgb, sourceRgb, landmark, label, config, training=True, img_path=None):
    '''Prepare model input images.

    Arguments:
    targetRgb: original images or fake images.
    sourceRgb: source images.
    landmark: face landmark.
    label: deepfake labels. genuine: 0, fake: 1.
    config: deepfake config dict.
    training: return processed image with aug or not.
    '''

    rng = np.random

    images = [targetRgb, sourceRgb]

    if training and rng.rand() >= 0.7:
        images, landmark = resize_aug(images, landmark)

    # multi-scale facial swap.

    targetRgb, sourceRgb = images
    # if input image is genuine.
    mfs_result, bbox = targetRgb, np.zeros((1, 4))
    # if input image is fake image. generate new fake image with mfs.
    if label:
        blending_type = 'poisson' if rng.rand() >= 0.5 else 'alpha'

        if rng.rand() >= 0.2:
            # global facial swap.
            sliding_win = targetRgb.shape[:2]

            if rng.rand() > 0.5:
                # fake to source global facial swap.
                mfs_result, bbox = multi_scale_facial_swap(
                    targetRgb, sourceRgb, landmark, config,
                    sliding_win, blending_type, training
                )
            elif rng.rand() >= 0.5:
                # source to fake global facial swap.
                mfs_result, bbox = multi_scale_facial_swap(
                    sourceRgb, targetRgb, landmark, config,
                    sliding_win, blending_type, training
                )
            else:
                mfs_result, bbox = targetRgb, np.array([[0, 0, 224, 224]])
                cropMfs, landmark = get_align5p(
                    [mfs_result], landmark, rng, config, training
                )
                mfs_result = cropMfs[0]
        else:
            # parial facial swap.
            prior_bbox = config['sliding_win']['prior_bbox']
            sliding_win = prior_bbox[np.random.choice(len(prior_bbox))]
            mfs_result, bbox = multi_scale_facial_swap(
                sourceRgb, targetRgb, landmark, config,
                sliding_win, blending_type, training
            )
    else:
        # crop face with landmark.
        cropMfs, landmark = get_align5p(
            [mfs_result], landmark, rng, config, training
        )
        mfs_result = cropMfs[0]

    if mfs_result is None:
        return None, 'multi scale facial swap err.'

    if training:  # and rng.rand() >= 0.5:
        mfs_result, bbox = image_h_mirror(mfs_result, bbox)
        # mfs_result = add_mask(mfs_result, img_path)
        mfs_result = add_noise(rng, mfs_result)

    genuine = True if not label else False

    location_label, confidence_label = label_assign(
        bbox.astype('float32') / config['crop_face']['output_size'],
        config, genuine
    )

    return mfs_result, {'label': label, 'location_label': location_label,
                        'confidence_label': confidence_label}


def prepare_test_input(img, ld, label, config):
    config = config['crop_face']

    img, ld = align_5p(
        img, ld=ld,
        face_width=config['face_width'], canvas_size=config['output_size'],
        scale=config['scale']
    )
    return img, {'label': label}

# vim: ts=4 sw=4 sts=4 expandtab
