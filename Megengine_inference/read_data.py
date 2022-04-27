# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import random
from megengine.data.dataset import Dataset
import numpy as np

class MyData(Dataset):
    def __init__(self, train='train'):
        super().__init__()
        self.istrain = train
        ## split the dataset to ['train','val'] = [8000,192]
        sample_index = random.sample(range(0, 8192), 192)
        if self.istrain == 'train':
            sample_index = list(set(range(0, 8192)) - set(sample_index))

        ## load data
        if self.istrain in ['train', 'val']:
            content_input = open('/home/megstudio/dataset/burst_raw/competition_train_input.0.2.bin', 'rb').read()
            content_gt = open('/home/megstudio/dataset/burst_raw/competition_train_gt.0.2.bin', 'rb').read()

            self.data = np.frombuffer(content_input, dtype='uint16').reshape((-1, 256, 256))[sample_index]
            self.gt = np.frombuffer(content_gt, dtype='uint16').reshape((-1, 256, 256))[sample_index]
            ## norm to [0,1]
            self.data = np.float32(self.data) * np.float32(1 / 65536)
            self.gt = np.float32(self.gt) * np.float32(1 / 65536)

        elif self.istrain == 'test':
            content_input = open('/home/megstudio/dataset/burst_raw/competition_test_input.0.2.bin', 'rb').read()
            self.data = np.frombuffer(content_input, dtype='uint16').reshape((-1, 256, 256))
            self.data = np.float32(self.data) * np.float32(1 / 65536)
        self.len = len(self.data)

    def __getitem__(self, i):
        if self.istrain in ['train', 'val']:
            gt = self.gt[i]
            sample = self.data[i]
            return sample[None], gt[None]

        elif self.istrain == 'test':
            sample = self.data[i]
            return sample[None]

    def __len__(self):
        return self.len
