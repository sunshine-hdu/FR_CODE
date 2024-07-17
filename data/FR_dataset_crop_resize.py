import os
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from FR_config import config

class IQA_datset(torch.utils.data.Dataset):
    def __init__(self, config, scene_list, transform, mode='train'):
        super(IQA_datset, self).__init__()
        self.config = config
        self.scene_list = scene_list
        self.transform = transform
        self.mode = mode
        self.dis_path = self.config.db_path
        self.ref_path = self.config.Ref_path
        self.txt_file_name = self.config.text_path
        self.aug_num = self.config.aug_num

        idx_data, dis_files_data, score_data, ref_file_data, ref_data = [], [], [], [], []

        if config.db_name == 'win5':
            # win5 begin
            name_list_heng = [['5_1', '5_2', '5_3'], ['5_4', '5_5', '5_6'], ['5_7', '5_8', '5_9']]   #0
            name_list_shu = [['1_5', '2_5', '3_5'], ['4_5', '5_5', '6_5'], ['7_5', '8_5', '9_5']]   # 90
            name_list_pie = [['1_1', '2_2', '3_3'], ['4_4', '5_5', '6_6'], ['7_7', '8_8', '9_9']]   #45
            name_list_na = [['9_1', '8_2', '7_3'], ['6_4', '5_5', '4_6'], ['3_7', '2_8', '1_9']]    #135

            name_list_sel = [
                name_list_heng,
                name_list_shu,
                name_list_pie,
                name_list_na
            ]

            ref_name = ['Bikes', 'dishes', 'Flowers', 'greek', 'museum', 'Palais_du_Luxembourg', 'rosemary', 'Sphynx',
                        'Swans_1', 'Vespa']

            ref_heng = [['51', '52', '53'], ['54', '55', '56'], ['57', '58', '59']]  # 0
            ref_shu = [['15', '25', '35'], ['45', '55', '65'], ['75', '85', '95']]  # 90
            ref_pie = [['11', '22', '33'], ['44', '55', '66'], ['77', '88', '99']]  # 45
            ref_na = [['91', '82', '73'], ['64', '55', '46'], ['37', '28', '19']]  # 135

            ref_list = [
                ref_heng,
                ref_shu,
                ref_pie,
                ref_na
            ]
            # win5 end
        elif config.db_name == 'NBU':

            # nbu begin
            name_list_heng = [['004_000', '004_001', '004_002'], ['004_003', '004_004', '004_005'],
                              ['004_006', '004_007', '004_008']]
            name_list_shu = [['000_004', '001_004', '002_004'], ['003_004', '004_004', '005_004'],
                             ['006_004', '007_004', '008_004']]
            name_list_pie = [['000_000', '001_001', '002_002'], ['003_003', '004_004', '005_005'],
                             ['006_006', '007_007', '008_008']]
            name_list_na = [['008_000', '007_001', '006_002'], ['005_003', '004_004', '003_005'],
                            ['002_006', '001_007', '000_008']]

            name_list_sel = [
                name_list_heng,
                name_list_shu,
                name_list_pie,
                name_list_na
            ]

            ref_name = ['I01R0', 'I02R0', 'I03R0', 'I04R0', 'I05R0', 'I06R0', 'I07R0', 'I08R0', 'I09R0', 'I10R0', 'I11R0',
                        'I12R0', 'I13R0', 'I14R0']
            ref_heng = [['004_000', '004_001', '004_002'], ['004_003', '004_004', '004_005'],
                        ['004_006', '004_007', '004_008']]
            ref_shu = [['000_004', '001_004', '002_004'], ['003_004', '004_004', '005_004'],
                       ['006_004', '007_004', '008_004']]
            ref_pie = [['000_000', '001_001', '002_002'], ['003_003', '004_004', '005_005'],
                       ['006_006', '007_007', '008_008']]
            ref_na = [['008_000', '007_001', '006_002'], ['005_003', '004_004', '003_005'],
                      ['002_006', '001_007', '000_008']]

            ref_list = [
                ref_heng,
                ref_shu,
                ref_pie,
                ref_na
            ]
            # nbu end
        else:
            # SHU begin
            name_list_heng = [['8_4', '8_5', '8_6'], ['8_7', '8_8', '8_9'], ['8_10', '8_11', '8_12']]   #0
            name_list_shu = [['4_8', '5_8', '6_8'], ['7_8', '8_8', '9_8'], ['10_8', '11_8', '12_8']]   # 90
            name_list_pie = [['4_4', '5_5', '6_6'], ['7_7', '8_8', '9_9'], ['10_10', '11_11', '12_12']]   #45
            name_list_na = [['4_12', '5_11', '6_10'], ['7_9', '8_8', '9_7'], ['10_6', '11_5', '12_4']]    #135

            name_list_sel = [
                name_list_heng,
                name_list_shu,
                name_list_pie,
                name_list_na
            ]

            ref_name = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']

            ref_heng = [['8_4', '8_5', '8_6'], ['8_7', '8_8', '8_9'], ['8_10', '8_11', '8_12']]   #0
            ref_shu = [['4_8', '5_8', '6_8'], ['7_8', '8_8', '9_8'], ['10_8', '11_8', '12_8']]   # 90
            ref_pie = [['4_4', '5_5', '6_6'], ['7_7', '8_8', '9_9'], ['10_10', '11_11', '12_12']]   #45
            ref_na = [['4_12', '5_11', '6_10'], ['7_9', '8_8', '9_7'], ['10_6', '11_5', '12_4']]    #135

            ref_list = [
                ref_heng,
                ref_shu,
                ref_pie,
                ref_na
            ]
            # SHU end

        for ref in range(len(ref_name)):
            for i in range(len(ref_list)):
                ref_sai_each = []
                ref_f_cat = []
                ref_count = 0
                for j in range(len(ref_list[i])):
                    for n in range(len(ref_list[i][j])):
                        if config.db_name == 'SHU':
                            ref_each = '{}/{}.bmp'.format(ref_name[ref], ref_list[i][j][n])
                        else:
                            ref_each = '{}/{}.png'.format(ref_name[ref], ref_list[i][j][n])
                        ref_sai_each.append(ref_each)
                        ref_count += 1
                    if ref_count >= len(ref_list[i][j]):
                        ref_f_cat.append(ref_sai_each)
                        ref_sai_each = []
                        ref_count = 0
                ref_data.append(ref_f_cat)


        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                idx, dis, score = line.split()
                idx = int(idx)
                score = float(score)

                if idx in self.scene_list:
                    for aug_num in range(self.aug_num):
                        for i in range(len(name_list_sel)):
                            sai_each = []
                            f_cat = []
                            count = 0

                            for j in range(len(name_list_sel[i])):  # [3,3,3]

                                for n in range(len(name_list_sel[i][j])):
                                    if config.db_name == 'SHU':
                                        each = '{}/{}.bmp'.format(dis, name_list_sel[i][j][n])
                                    else:
                                        each = '{}/{}.png'.format(dis, name_list_sel[i][j][n])
                                    sai_each.append(each)
                                    count += 1
                                if count >= len(name_list_sel[i][j]):  # 3
                                    f_cat.append(sai_each)
                                    sai_each = []
                                    count = 0
                            dis_files_data.append(f_cat)
                            idx_data.append(idx)
                            score_data.append(score)

                            for i in range(len(ref_name)):
                                if config.db_name == 'NBU':
                                    if ref_name[i][0:3] in dis:
                                        ref_file_data.append(ref_data[i])
                                        break
                                else:
                                    if ref_name[i] in dis:
                                        ref_file_data.append(ref_data[i])
                                        break

        # reshape list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        idx_data = np.array(idx_data)
        idx_data = idx_data.reshape(-1, 1)

        self.data_dict = {
            'd_img_list': dis_files_data,
            'score_list': score_data,
            'idx_list': idx_data,
            'ref_img_list' : ref_file_data,
        }

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        h, w = self.config.input_size
        top = random.randint(0, h - config.crop_size)
        left = random.randint(0, w - config.crop_size)
        bottom = top + config.crop_size
        right = left + config.crop_size

        if_flip = random.random()
        resize_percent = 0.8

        angle = random.randint(-60, 60)

        cat_all = []
        '''1-'''
        for n in range(len(self.data_dict['d_img_list'][idx])):
            dis = []
            for i in range(len(self.data_dict['d_img_list'][idx][n])):
                d_img_name = self.data_dict['d_img_list'][idx][n][i]
                d_img = Image.open(Path(self.config.db_path) / d_img_name).convert("RGB")

                # origin
                if self.mode == 'train':
                    if if_flip < resize_percent:
                        d_img = d_img.resize(self.config.new_size)
                    else:
                        d_img = d_img.resize(self.config.input_size)       
                        d_img = d_img.rotate(angle)
                        d_img = d_img.crop((left, top, right, bottom))
                if self.mode == 'test':
                    d_img = d_img.resize(self.config.new_size)

                if self.transform:
                    d_img = self.transform(d_img)

                dis.append(d_img)

            dis = torch.cat(dis, dim=0)
            cat_all.append(dis)

        ref_cat_all = []
        '''1-'''
        for n in range(len(self.data_dict['ref_img_list'][idx])):
            ref = []
            for i in range(len(self.data_dict['ref_img_list'][idx][n])):
                ref_img_name = self.data_dict['ref_img_list'][idx][n][i]
                ref_img = Image.open(Path(self.ref_path) / ref_img_name).convert("RGB")

                # origin
                ref_img = ref_img.resize(self.config.input_size)

                if self.mode == 'train':
                    if if_flip < resize_percent:
                        ref_img = ref_img.resize(self.config.new_size)
                    else:
                        ref_img = ref_img.resize(self.config.input_size)
                        ref_img = ref_img.rotate(angle)
                        ref_img = ref_img.crop((left, top, right, bottom))
                if self.mode == 'test':
                    ref_img = ref_img.resize(self.config.new_size)

                if self.transform:
                    ref_img = self.transform(ref_img)

                ref.append(ref_img)

            ref= torch.cat(ref, dim=0)

            ref_cat_all.append(ref)

        score = self.data_dict['score_list'][idx]
        idx = self.data_dict['idx_list'][idx]

        sample = {
            'd_img_org': cat_all,
            'score': score,
            'idx': idx,
            'ref_img_org': ref_cat_all
        }

        return sample
