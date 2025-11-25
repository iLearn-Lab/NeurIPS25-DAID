#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import csv
from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input
import random

class DeepfakeDataset(Dataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        mode: train or test.
        config: hypter parameters for processing images.
    """

    def __init__(self, mode: str, config: dict):
        super().__init__()

        self.config = config
        self.mode = mode
        if self.mode == "train":
            self.training_strategy = self.config['train']['strategy']
            self.remove = True if self.training_strategy == 'remove' else False

        self.root = self.config['dataset']['img_path']
        self.csv_list = self.config['dataset']['img_csv_list']
        self.landmark_path = self.config['dataset']['ld_path']
        self.rng = np.random
        assert mode in ['train', 'test']
        self.do_train = True if mode == 'train' else False
        self.info_meta_dict = self.load_landmark_json(self.landmark_path)
        self.class_dict = self.collect_class()

        """
        if read from file directly, use this
        """
        # self.samples_dict = self.collect_samples()
        """
        if read from csv with race info, use this
        """
        self.samples_dict = self.collect_samples_from_csv()
        self.samples = self.samples_dict['samples']
        self.samples_male = self.samples_dict['samples_male']
        self.samples_female = self.samples_dict['samples_female']
        self.samples_white = self.samples_dict['samples_white']
        self.samples_black = self.samples_dict['samples_black']
        self.samples_asian = self.samples_dict['samples_asian']

        self.bio_images = {
            'male': self.samples_male,
            'female': self.samples_female,
            'black': self.samples_black,
            'white': self.samples_white,
            'asian': self.samples_asian,
        }

        self.bio_acc = {
            'male': {
                'black': 0,
                'white': 0,
                'asian': 0,
                'other': 0,
            },
            'female': {
                'black': 0,
                'white': 0,
                'asian': 0,
                'other': 0,
            },
        }

        self.bio_ratio = {
            'male': {
                'black': 0,
                'white': 0,
                'asian': 0,
                'other': 0,
            },
            'female': {
                'black': 0,
                'white': 0,
                'asian': 0,
                'other': 0,
            },
        }

        print('dataset samples:', len(self.samples))
        if self.mode == 'train':
            spe_labels = self.samples_male.keys()
            for spe_label in spe_labels:
                for img_path, img_label in self.samples_male[spe_label]:
                # if img_label['race'] == 'white':
                    self.bio_acc['male'][img_label['race']] += 1
            spe_labels = self.samples_female.keys()
            for spe_label in spe_labels:
                for img_path, img_label in self.samples_female[spe_label]:
                    self.bio_acc['female'][img_label['race']] += 1

        for gender in ['male', 'female']:
            for race in ['black', 'white', 'asian']:
                self.bio_ratio[gender][race] = self.bio_acc[gender][race] / len(self.samples)
        # print('dataset images_bio_choose:', bio_choose)
        print(self.bio_ratio)


    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict

    def collect_samples(self) -> dict:
        samples = []
        directory = os.path.expanduser(self.root)
        for key in sorted(self.class_dict.keys()):
            d = os.path.join(directory, key)
            if not os.path.isdir(d):
                continue
            for r, _, filename in sorted(os.walk(d, followlinks=True)):
                for name in sorted(filename):
                    path = os.path.join(r, name)
                    info_key = path
                    video_name = '/'.join(path.split('/')[:-1])
                    if self.info_meta_dict.get(info_key):
                        info_meta = self.info_meta_dict[info_key]
                        landmark = info_meta['landmark']
                        class_label = int(info_meta['label'])
                        source_path = info_meta['source_path']
                        samples.append(
                            (path, {'labels': class_label, 'landmark': landmark,
                                    'source_path': source_path,
                                    'video_name': video_name})
                        )

        return {"samples": samples}

    def collect_samples_from_csv(self) -> dict:
        samples = []
        samples_male = {}
        samples_female = {}
        samples_white = {}
        samples_black = {}
        samples_asian = {}
        class_to_idx = {'1': 0, '0': 1}
        group_id_to_idx = {
            'male': {
                'black': 1,
                'white': 2,
                'asian': 3,
                'other': 8,
            },
            'female': {
                'black': 4,
                'white': 5,
                'asian': 6,
                'other': 9,
            },
        }
        for csv_file in self.csv_list:
            with open(csv_file, newline='', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    # print(row)
                    # if self.do_train:
                    img_path = row[0]
                    label = class_to_idx[row[1]]
                    is_male = row[2]
                    is_asian = row[3]
                    is_white = row[4]
                    is_black = row[5]
                    if self.mode == 'train' and label == 0:
                        spe_label = int(row[7])
                    else:
                        spe_label = 0

                    img_name = img_path.split('/')[-1]
                    img_path = os.path.join(os.path.dirname(csv_file), 'crop_img', img_name)
                    gender = 'male' if int(is_male) == 1 else 'female'
                    if int(is_asian) == 1:
                        race = 'asian'
                    elif int(is_white) == 1:
                        race = 'white'
                    elif int(is_black) == 1:
                        race = 'black'
                    else:
                        race = 'other'

                    if self.info_meta_dict.get(img_path) is None:
                        continue
                    # print(len(self.info_meta_dict))

                    info_meta = self.info_meta_dict[img_path]
                    landmark = info_meta['landmark']
                    class_label = int(label)
                    if self.do_train:
                        if label == 1:
                            source_path = info_meta['source_path']
                        else:
                            img_name = img_path.split('/')[-1].replace('.png', '')
                            cls, spe_name, source_video, target_video, cnt_frame = img_name.split('_')
                            source_path = os.path.join(os.path.dirname(img_path), f'{cls}_youtube_{source_video}_{cnt_frame}.png')
                    else:
                        source_path = img_path

                    if self.info_meta_dict.get(source_path) is None:
                        continue

                    samples.append(
                        (img_path, {'labels': class_label, 'landmark': landmark,
                                    'source_path': source_path,
                                    'video_name': img_path,
                                    'gender': gender,
                                    'race': race,
                                    'group_id': group_id_to_idx[gender][race],
                                    'spe_label': spe_label})
                    )

                    if int(is_male) == 1:
                        if samples_male.get(spe_label) is None:
                            samples_male[spe_label] = []
                        samples_male[spe_label].append((img_path, {'labels': class_label, 'landmark': landmark,
                                                                   'source_path': source_path,
                                                                   'video_name': img_path,
                                                                   'gender': gender,
                                                                   'race': race,
                                                                   'group_id': group_id_to_idx[gender][race],
                                                                   'spe_label': spe_label,}))
                    if int(is_male) != 1:
                        if samples_female.get(spe_label) is None:
                            samples_female[spe_label] = []
                        samples_female[spe_label].append((img_path, {'labels': class_label, 'landmark': landmark,
                                                                     'source_path': source_path,
                                                                     'video_name': img_path,
                                                                     'gender': gender,
                                                                     'race': race,
                                                                     'group_id': group_id_to_idx[gender][race],
                                                                     'spe_label': spe_label}))
                    if int(is_white) == 1:
                        if samples_white.get(spe_label) is None:
                            samples_white[spe_label] = []
                        samples_white[spe_label].append((img_path, {'labels': class_label, 'landmark': landmark,
                                                                    'source_path': source_path,
                                                                    'video_name': img_path,
                                                                    'gender': gender,
                                                                    'race': race,
                                                                    'group_id': group_id_to_idx[gender][race],
                                                                    'spe_label': spe_label}))
                    if int(is_black) == 1:
                        if samples_black.get(spe_label) is None:
                            samples_black[spe_label] = []
                        samples_black[spe_label].append((img_path, {'labels': class_label, 'landmark': landmark,
                                                                    'source_path': source_path,
                                                                    'video_name': img_path,
                                                                    'gender': gender,
                                                                    'race': race,
                                                                    'group_id': group_id_to_idx[gender][race],
                                                                    'spe_label': spe_label}))
                    if int(is_asian) == 1:
                        if samples_asian.get(spe_label) is None:
                            samples_asian[spe_label] = []
                        samples_asian[spe_label].append((img_path, {'labels': class_label, 'landmark': landmark,
                                                                    'source_path': source_path,
                                                                    'video_name': img_path,
                                                                    'gender': gender,
                                                                    'race': race,
                                                                    'group_id': group_id_to_idx[gender][race],
                                                                    'spe_label': spe_label}))

        return {'samples': samples,
                'samples_male': samples_male,
                'samples_female': samples_female,
                'samples_white': samples_white,
                'samples_black': samples_black,
                "samples_asian": samples_asian}

    def collect_class(self) -> Dict:
        # classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        # classes.sort(reverse=True)
        # return {classes[i]: np.int32(i) for i in range(len(classes))}
        return {'fake': np.int32(0), 'real': np.int32(1)}

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        source_path = label_meta['source_path']
        gender = label_meta['gender']
        race = label_meta['race']
        group_id = label_meta['group_id']
        spe_label = label_meta['spe_label']
        print(label_meta['video_name'])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if self.mode == "train":
            if self.remove:
                img, label_dict = prepare_train_input(
                    img, source_img, ld, label, self.config, self.do_train, img_path=path,
                )
                if isinstance(label_dict, str):
                    return None, label_dict

                location_label = torch.Tensor(label_dict['location_label'])
                confidence_label = torch.Tensor(label_dict['confidence_label'])
                img = torch.Tensor(img.transpose(2, 0, 1))

                waiting_list = []
                male_list = ['male', 'female']
                race_list = ['black', 'white', 'asian']

                for item in male_list:
                    if item != gender:
                        waiting_list.append(item)
                # for item in race_list:
                #     if item != race:
                #         waiting_list.append(item)

                bio_contras_choose = random.choice(waiting_list)
                new_ind = random.randint(0, len(self.bio_images[bio_contras_choose][spe_label]) - 1)
                contras_img_path, contras_label_meta = self.bio_images[bio_contras_choose][spe_label][new_ind]

                contras_ld = np.array(contras_label_meta['landmark'])
                contras_label = contras_label_meta['labels']
                contras_source_path = contras_label_meta['source_path']
                # print(contras_img_path, contras_source_path)
                contras_gender = contras_label_meta['gender']
                contras_race = contras_label_meta['race']
                contras_group_id = contras_label_meta['group_id']
                spe_label = contras_label_meta['spe_label']

                contras_img = cv2.imread(contras_img_path, cv2.IMREAD_COLOR)
                contras_source_img = cv2.imread(contras_source_path, cv2.IMREAD_COLOR)

                contras_img, contras_img_label_dict = prepare_train_input(
                    contras_img, contras_source_img, contras_ld, contras_label, self.config, self.do_train
                )
                if isinstance(contras_img_label_dict, str):
                    return None, contras_img_label_dict

                contras_location_label = torch.Tensor(contras_img_label_dict['location_label'])
                contras_confidence_label = torch.Tensor(contras_img_label_dict['confidence_label'])
                contras_img = torch.Tensor(contras_img.transpose(2, 0, 1))

                return img, (label, location_label, confidence_label, gender, race, 1-self.bio_ratio[gender][race], group_id), contras_img, (contras_label, contras_location_label, contras_confidence_label, contras_gender, contras_race, 1-self.bio_ratio[contras_gender][contras_race], contras_group_id)

            else:
                img, label_dict = prepare_train_input(
                    img, source_img, ld, label, self.config, self.do_train, img_path=path,
                )
                if isinstance(label_dict, str):
                    return None, label_dict

                location_label = torch.Tensor(label_dict['location_label'])
                confidence_label = torch.Tensor(label_dict['confidence_label'])
                img = torch.Tensor(img.transpose(2, 0, 1))

                return img, (label, location_label, confidence_label, gender, race)

        elif self.mode == 'test':
            ori_img = img
            img, label_dict = prepare_test_input(
                [img], ld, label, self.config
            )
            img = torch.Tensor(img[0].transpose(2, 0, 1))
            video_name = label_meta['video_name']
            video_name = os.path.basename(video_name).split('/')[-1].split('.')[0].split('_')
            video_name = video_name[0:-1]
            video_name = '_'.join(video_name)
            return img, (label, video_name, gender, race, group_id), ori_img

        else:
            raise ValueError("Unsupported mode of dataset!")

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from lib.util import load_config

    config = load_config('./configs/caddm_train.cfg')
    d = DeepfakeDataset(mode="test", config=config)
    for index in range(len(d)):
        res = d[index]
# vim: ts=4 sw=4 sts=4 expandtab
