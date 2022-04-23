# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.res(x)

class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[10, 14, 14, 16], nb=4):
        super(UNetRes, self).__init__()
        self.m_head = nn.Conv2d(in_nc, nc[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0], nc[0]) for _ in range(nb)],
            nn.Conv2d(nc[0], nc[1], kernel_size=2, stride=2, bias=False)
        )
        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1], nc[1]) for _ in range(nb)],
            nn.Conv2d(nc[1], nc[2], kernel_size=2, stride=2, bias=False)
        )
        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2], nc[2]) for _ in range(nb)],
            nn.Conv2d(nc[2], nc[3], kernel_size=2, stride=2, bias=False)
        )

        self.m_body = nn.Sequential(
            *[ResBlock(nc[3], nc[3]) for _ in range(nb)])

        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], kernel_size=2, stride=2, bias=False),
                                   *[ResBlock(nc[2], nc[2]) for _ in range(nb)])
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], kernel_size=2, stride=2, bias=False),
                                   *[ResBlock(nc[1], nc[1]) for _ in range(nb)])
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], kernel_size=2, stride=2, bias=False),
                                   *[ResBlock(nc[0], nc[0]) for _ in range(nb)])

        self.m_tail = nn.Conv2d(nc[0], out_nc, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x


if __name__ == '__main__':
    x = torch.rand(16, 1, 256, 256)
    net = UNetRes(in_nc=1, out_nc=1, nc=[10, 14, 14, 16], nb=4)
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(net)
    print('output shape:', y.shape)
    print('    Total params: %.5fKB' % (sum(p.numel() for p in net.parameters()) / 1e3))
