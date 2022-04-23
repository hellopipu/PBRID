# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import torch
from torch import nn
from BasicModule import UNetRes


class HQSNet(nn.Module):
    def __init__(self, buffer_size=5, n_iter=8):
        '''
        HQS-Net modified from paper " Learned Half-Quadratic Splitting Network for MR Image Reconstruction "
        ( https://openreview.net/pdf?id=h7rXUbALijU ) ( https://github.com/hellopipu/HQS-Net )
        :param buffer_size:  buffer_size m
        :param n_iter:  iterations n
        '''

        super().__init__()
        self.m = buffer_size
        self.n_iter = n_iter

        self.rec_blocks = UNetRes(in_nc=1 * (self.m + 1), out_nc=1 * self.m, nc=[10, 14, 14, 16], nb=4)

        # estimate noise level for each image
        self.ee = nn.Sequential(
            nn.Conv2d(1, 16, 8, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 8, 8, groups=16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, 1, 1),
            nn.Sigmoid()
        )

    def update_opration(self, pred, img):
        update = pred + self.mu * (img - pred)
        return update

    def forward(self, img):
        '''
        :param img: noisy raw img, (batch,1,h,w)
        :return: de-noised img
        '''
        self.mu = self.ee(img)

        ## initialize buffer f : the concatenation of m copies of noisy images
        f = torch.cat([img] * self.m, 1).to(img.device)

        ## n reconstruction blocks
        for i in range(self.n_iter):
            f_1 = f[:, 0:1].clone()
            updated_f_1 = self.update_opration(f_1, img)
            f = f + self.rec_blocks(torch.cat([f, updated_f_1], 1))
        return f[:, 0:1]


if __name__ == '__main__':
    x = torch.rand(16, 1, 256, 256)
    net = HQSNet(buffer_size=7, n_iter=10)
    print(net)
    net.eval()
    with torch.no_grad():
        y = net(x)
    print('output shape: ', y.shape)
    print('    Total params: %.5fKB' % (sum(p.numel() for p in net.parameters()) / 1e3))
