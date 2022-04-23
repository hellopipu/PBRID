# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import argparse
import os
import time
import torch
import pathlib
import yaml
import random
import torch.utils.data as Data
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from os.path import join
import numpy as np

from read_data import MyData
from utils import cal_score
from HQSNet import HQSNet


class Solver():
    def __init__(self, args):
        # load exp settings from yaml file
        self.load_yaml_setting(args.yaml)
        # make it reproducible
        self.fix_seed(self.seed)
        # arg settings for running
        self.mode = args.mode
        self.resume = args.resume

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'val':
            self.val()
        elif self.mode == 'test':
            self.test()

    def fix_seed(self, seed=33):
        ## make it reproducible
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = True  # should be False if need reproducible, but slow the training
        # torch.use_deterministic_algorithms(True) # should uncomment this line if need reproducible

    def load_yaml_setting(self, path):
        path_config = pathlib.Path(path)
        with open(path_config, "r") as f:
            yaml_data = yaml.safe_load(f)
        for k in yaml_data:
            self.__setattr__(k, yaml_data[k])

    def train(self):

        if self.loss == 'l1':
            criterion = nn.L1Loss()
        else:
            raise NotImplementedError(f"{self.loss} not yet supported.")

        model = HQSNet(buffer_size=self.buffer_size, n_iter=self.n_iter)
        if self.ema:
            model_E = HQSNet(buffer_size=self.buffer_size, n_iter=self.n_iter)
            model_E.cuda()
            model_E.eval()

        dataset_train = MyData('train')
        dataset_val = MyData('val')

        num_train_image = len(dataset_train)
        num_val_image = len(dataset_val)

        loader_train = Data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                       num_workers=self.num_works, pin_memory=self.pin_memory)

        loader_val = Data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=self.num_works, pin_memory=self.pin_memory)

        optim = torch.optim.Adam(model.parameters(), lr=self.lr)

        if self.resume:
            checkpoint = torch.load(join(self.weight_dir, self.exp + '.pth'))
            start_epoch = checkpoint['epoch'] + 1
            best_val_score = checkpoint['val_score']
            model.load_state_dict(checkpoint['net'])
            if self.ema:
                model_E.load_state_dict(checkpoint['ema'])
                model_E.cuda()
                model_E.eval()

            optim.load_state_dict(checkpoint['optim'])
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            train_iter = num_train_image * (start_epoch - 1) // self.batch_size
            print('resume: ', 'epoch: ', start_epoch, 'val_score: ', best_val_score)
        else:
            start_epoch = 0
            best_val_score = 0
            train_iter = 1

        print('exp: ', self.exp, '\nnum of train image: ', num_train_image, '\nnum of val image: ', num_val_image,
              '\nmodel param: %.2f K' % (sum(p.numel() for p in model.parameters()) / 1e3))
        writer = SummaryWriter(self.log_dir + self.exp)
        model.cuda()
        for epoch in range(start_epoch, self.epoch):
            ############# train
            model.train()
            with tqdm(loader_train, unit='batch') as tepoch:
                for input, gt in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    input = input.cuda()
                    gt = gt.cuda()

                    pred = model(input)

                    loss = criterion(pred, gt)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    if self.ema:
                        #################### update ema
                        netG_params = dict(model.named_parameters())
                        netE_params = dict(model_E.named_parameters())
                        for k in netG_params.keys():
                            netE_params[k].data.mul_(self.ema_decay).add_(netG_params[k].data, alpha=1 - self.ema_decay)
                        ####################

                    tepoch.set_postfix(loss=loss.item())
                    writer.add_scalar("loss/loss_g", loss, train_iter)
                    train_iter += 1

            ############## val
            if epoch % 1 == 0:

                model.eval()
                val_score = 0
                base_score = 0

                with torch.no_grad():
                    for input, gt in loader_val:
                        input = input.cuda()
                        gt = gt.cuda()
                        if self.ema:
                            pred = model_E(input)
                        else:
                            pred = model(input)

                        pred = (pred.cpu().numpy() * 65536).clip(0, 65535).astype('uint16')
                        input = (input.cpu().numpy() * 65536).clip(0, 65535).astype('uint16')
                        gt = (gt.cpu().numpy() * 65536).clip(0, 65535).astype('uint16')

                        base_score += self.batch_size * cal_score(np.float32(input), np.float32(gt))
                        val_score += self.batch_size * cal_score(np.float32(pred), np.float32(gt))

                    val_score /= num_val_image
                    val_score = np.log10(100 / val_score) * 5
                    writer.add_scalar("loss/val_score", val_score, epoch)

                    base_score /= num_val_image
                    base_score = np.log10(100 / base_score) * 5
                    writer.add_scalar("loss/base_score", base_score, epoch)

                print("Epoch {}/{}".format(epoch, self.epoch))
                print(" base score:\t\t{:.6f}".format(base_score))
                print(" val score:\t\t{:.6f}".format(val_score))

                if val_score > best_val_score:
                    best_val_score = val_score
                    state = {'net': model.state_dict(), 'epoch': epoch, 'val_score': best_val_score,
                             'optim': optim.state_dict()}
                    if self.ema:
                        state['ema'] = model_E.state_dict()
                    if not os.path.isdir(self.weight_dir):
                        os.makedirs(self.weight_dir)
                    torch.save(state, join(self.weight_dir, self.exp + '.pth'))
                    print('-- best model updated! --')
        writer.close()

    def val(self):
        model = HQSNet(buffer_size=self.buffer_size, n_iter=self.n_iter)

        #################################################333
        dataset_train = MyData('train')
        dataset_test = MyData('val')

        num_train_image = len(dataset_test)
        print('exp: ', self.exp, ' :num of image: ', num_train_image)

        loader_test = Data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                      num_workers=self.num_works, pin_memory=self.pin_memory)

        checkpoint = torch.load(join(self.weight_dir, self.exp + '.pth'))
        if self.ema:
            model.load_state_dict(checkpoint['ema'])
        else:
            model.load_state_dict(checkpoint['net'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_score = checkpoint['val_score']
        print('resume: ', 'epoch: ', start_epoch, 'val_score: ', best_val_score)

        model.cuda()

        ############## val
        model.eval()
        val_score = 0
        base_score = 0

        with torch.no_grad():
            for input, gt in loader_test:
                input = input.cuda()
                pred = model(input)

                pred = (pred.cpu().numpy() * 65536).clip(0, 65535).astype('uint16')
                input = (input.cpu().numpy() * 65536).clip(0, 65535).astype('uint16')
                gt = (gt.numpy() * 65536).clip(0, 65535).astype('uint16')

                base_score += self.batch_size * cal_score(np.float32(input), np.float32(gt))
                val_score += self.batch_size * cal_score(np.float32(pred), np.float32(gt))

            val_score /= num_train_image
            val_score = np.log10(100 / val_score) * 5

            base_score /= num_train_image
            base_score = np.log10(100 / base_score) * 5

            print(" base score:\t\t{:.6f}".format(base_score))
            print(" val score:\t\t{:.6f}".format(val_score))

    def test(self):
        model = HQSNet(buffer_size=self.buffer_size, n_iter=self.n_iter)

        dataset_test = MyData('test')
        num_test_image = len(dataset_test)
        print('exp: ',self.exp,' num of image: ', num_test_image)

        loader_test = Data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                      num_workers=self.num_works, pin_memory=self.pin_memory)

        checkpoint = torch.load(join(self.weight_dir, self.exp + '.pth'))
        if self.ema:
            model.load_state_dict(checkpoint['ema'])
        else:
            model.load_state_dict(checkpoint['net'])
        model.cuda()

        model.eval()
        if not os.path.isdir(self.pred_dir):
            os.makedirs(self.pred_dir)

        fout = open(join(self.pred_dir, self.exp + '_pred.0.2.bin'), 'wb')
        t0=time.time()
        with torch.no_grad():
            for input in loader_test:
                input = input.cuda()
                pred = model(input)

                pred = (pred.cpu().numpy()[:, 0] * 65536).clip(0, 65535).astype('uint16')
                fout.write(pred.tobytes())
            fout.close()
        print('inference time: %.6f ms'%(1000*(time.time()-t0)/num_test_image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'],
                        help='mode for the program')
    parser.add_argument('--resume', default=0, choices=[1, 0],
                        help='resume training')
    parser.add_argument('--yaml', default='exp.yaml',
                        help='yaml file path')
    args = parser.parse_args()
    s = Solver(args)
    s.run()
