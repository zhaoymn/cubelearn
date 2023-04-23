import numpy as np
import torch
from torch.utils.data import Dataset

class Gesture_Dataset(Dataset):
    def __init__(self, mode):
        super(Gesture_Dataset, self).__init__()
        self.mode = mode

        self.data_dir = "" #path to data dir
        #'G:/imwut_data/hand/'
        self.validation_sample_file = ""
        #'validation_samples.csv'
        self.train_sample_file = ""
        #'train_samples.csv'
        self.test_sample_file = ""#
        #'test_samples.csv'
        
        self.validation_samples = np.genfromtxt(self.validation_sample_file)
        self.train_samples = np.genfromtxt(self.train_sample_file)
        self.test_samples = np.genfromtxt(self.test_sample_file)
        users = [0,1,2,3,4,5] #training users
        if mode == 0:#train
            self.total_samples = 15
            self.sample_list = []
            for user in users:
                for gesture in range(12):
                    for sample in range(15):
                        self.sample_list.append([int(user), gesture, int(self.train_samples[sample])])
        elif mode == 1: #validation
            self.total_samples = 5
            self.sample_list = []
            for user in users:
                for gesture in range(12):
                    for sample in range(5):
                        self.sample_list.append([int(user), gesture, int(self.validation_samples[sample])])

    def __len__(self):
        return self.total_samples * 6 * 12

    def __getitem__(self, idx):
        user = self.sample_list[idx][0]
        gesture = self.sample_list[idx][1]
        sample = self.sample_list[idx][2]
        label = gesture
        data = np.load(self.data_dir + str(user)+'_' + str(gesture)+'_' + str(sample) + '.npy')
        #frame, chirp, virtual antenna, sample
        data = data[0, :, :, :, :] + data[1,:,:,:,:] * 1j
        #for HGR, AGR data (1 sec = 10 frames)
        data = data.reshape(10, 128, 12, 256)
        #for HAR data (2 sec = 20 frames)
        #data = data.reshape(20, 128, 12, 256)
        #DAT and RDAT models
        #data = data[:,:64,:,:128]

        data = np.array(data, dtype=np.complex64)
        return data, label

