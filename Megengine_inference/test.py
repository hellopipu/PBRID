# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

from read_data import MyData
import megengine as mge
from HQSNet import HQSNet
import megengine.data as Data
from megengine.data import SequentialSampler
import pickle

# 0. data path
# /home/megstudio/dataset/burst_raw/

# 1. load dataset
dataset_test = MyData('test')
num_train_image = len(dataset_test)
print('num of image: ', num_train_image)

loader_test = Data.DataLoader(dataset_test,SequentialSampler(dataset_test, batch_size=16, drop_last=False), num_workers=4)

# 2. load model
model = HQSNet(buffer_size=7, n_iter=10)

with open('weight/hqs_torch.pkl', 'rb') as f:
    w = pickle.load(f)
weights = {}
for k, v in w.items():
    if k.endswith('bias') and v.ndim == 1:
        v = v.reshape(1, -1, 1, 1)
    if k=='ee.2.weight':
        v = v[:,None]
    weights[k] = v
model.load_state_dict(weights, strict=False)

# 3. test & save
model.eval()
fout = open('pred/result.bin', 'wb')

for input_data in loader_test:
    pred = model(mge.tensor(input_data))
    pred = (pred.numpy()[:, 0] * 65536).clip(0, 65535).astype('uint16')
    fout.write(pred.tobytes())
fout.close()
