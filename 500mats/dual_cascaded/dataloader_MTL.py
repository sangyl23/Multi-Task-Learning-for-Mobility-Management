import os
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import torch
import math
from collections import Counter


class Dataloader():
    def __init__(self, path = '', batch_size = 32, his_len = 9, pre_len = 1, BS_num = 4, beam_num = 32, device = 'cpu'):
        self.batch_size = batch_size
        self.his_len = his_len
        self.pre_len = pre_len
        self.BS_num = BS_num
        self.beam_num = beam_num
        self.device = device
        
        self.files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        for i, f in enumerate(self.files):
            if not f.split('.')[-1] == 'mat':
                del (self.files[i])
        self.reset()

    def reset(self):
        self.done = False
        self.unvisited_files = [f for f in self.files]

        self.buffer = np.zeros((0, 2, self.his_len + self.pre_len, self.BS_num, self.beam_num))
        self.buffer_BS_label = np.zeros((0, self.his_len + self.pre_len))
        self.buffer_beam_label = np.zeros((0, self.his_len + self.pre_len))
        self.buffer_beam_power = np.zeros((0, self.his_len + self.pre_len, self.BS_num, self.beam_num))
        # positioning for UE and BS
        self.buffer_UE_loc = np.zeros((0, self.his_len + self.pre_len, 2))
        self.buffer_BS_loc = np.zeros((0, self.his_len + self.pre_len, self.BS_num, 2))

    def load(self, file):
        data = sio.loadmat(file)
        channels = data['MM_data'] # beam training received signal
        BS_labels = data['BS_label'] - 1 # optimal BS index label
        beam_labels = data['beam_label'] - 1 # optimal beam index label
        beam_power = data['beam_power'] # beam amplitude
        UE_loc = data['UE_loc_data'] # UE positioning
        BS_loc = data['BS_loc_data'] # BS positioning
        return channels, BS_labels, beam_labels, beam_power, UE_loc, BS_loc

    def next_batch(self):
        done = False

        # sequentially load data
        while self.buffer.shape[0] < self.batch_size:
            if len(self.unvisited_files) == 0:
                done = True
                break
            channels, BS_labels, beam_labels, beam_power, UE_loc, BS_loc = self.load(
                self.unvisited_files.pop(0))

            self.buffer = np.concatenate((self.buffer, channels), axis = 0)
            self.buffer_BS_label = np.concatenate((self.buffer_BS_label, BS_labels), axis = 0)
            self.buffer_beam_label = np.concatenate((self.buffer_beam_label, beam_labels), axis = 0)
            self.buffer_beam_power = np.concatenate((self.buffer_beam_power, beam_power), axis = 0)
            self.buffer_UE_loc = np.concatenate((self.buffer_UE_loc, UE_loc), axis = 0)
            self.buffer_BS_loc = np.concatenate((self.buffer_BS_loc, BS_loc), axis = 0)

        out_size = min(self.batch_size, self.buffer.shape[0])

        batch_channels = self.buffer[0 : out_size, :, :, :, :]
        batch_BS_labels = np.squeeze(self.buffer_BS_label[0 : out_size, :])
        batch_beam_labels = np.squeeze(self.buffer_beam_label[0 : out_size, :])
        batch_beam_power = np.squeeze(self.buffer_beam_power[0 : out_size, :, :, :])
        batch_UE_loc = np.squeeze(self.buffer_UE_loc[0:out_size, :, :])
        batch_BS_loc = np.squeeze(self.buffer_BS_loc[0:out_size, :, :, :])

        self.buffer = np.delete(self.buffer, np.s_[0 : out_size], 0)
        self.buffer_BS_label = np.delete(self.buffer_BS_label, np.s_[0 : out_size], 0)
        self.buffer_beam_label = np.delete(self.buffer_beam_label, np.s_[0 : out_size], 0)
        self.buffer_beam_power = np.delete(self.buffer_beam_power, np.s_[0 : out_size], 0)
        self.buffer_UE_loc = np.delete(self.buffer_UE_loc, np.s_[0:out_size], 0)
        self.buffer_BS_loc = np.delete(self.buffer_BS_loc, np.s_[0:out_size], 0)

        batch_channels = np.float32(batch_channels)
        batch_BS_labels = batch_BS_labels.astype(int)
        batch_beam_labels = batch_beam_labels.astype(int)
        batch_beam_power = np.float32(batch_beam_power)
        batch_UE_loc = np.float32(batch_UE_loc)
        batch_BS_loc = np.float32(batch_BS_loc)

        return torch.from_numpy(batch_channels).to(self.device), torch.from_numpy(batch_BS_labels).to(
            self.device), torch.from_numpy(batch_beam_labels).to(
            self.device), torch.from_numpy(batch_beam_power).to(
            self.device), torch.from_numpy(batch_UE_loc).to(
            self.device), torch.from_numpy(batch_BS_loc).to(
            self.device), done