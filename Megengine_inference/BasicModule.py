# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import megengine.module as M

class ResBlock(M.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(ResBlock, self).__init__()
        self.res = M.Sequential(
            M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            M.ReLU(),
            M.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.res(x)


class UNetRes(M.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[10, 14, 14, 16], nb=4):
        super(UNetRes, self).__init__()
        self.m_head = M.Conv2d(in_nc, nc[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.m_down1 = M.Sequential(
            *[ResBlock(nc[0], nc[0]) for _ in range(nb)],
            M.Conv2d(nc[0], nc[1], kernel_size=2, stride=2, bias=False)
        )
        self.m_down2 = M.Sequential(
            *[ResBlock(nc[1], nc[1]) for _ in range(nb)],
            M.Conv2d(nc[1], nc[2], kernel_size=2, stride=2, bias=False)
        )
        self.m_down3 = M.Sequential(
            *[ResBlock(nc[2], nc[2]) for _ in range(nb)],
            M.Conv2d(nc[2], nc[3], kernel_size=2, stride=2, bias=False)
        )

        self.m_body = M.Sequential(
            *[ResBlock(nc[3], nc[3]) for _ in range(nb)])

        self.m_up3 = M.Sequential(M.ConvTranspose2d(nc[3], nc[2], kernel_size=2, stride=2, bias=False),
                                   *[ResBlock(nc[2], nc[2]) for _ in range(nb)])
        self.m_up2 = M.Sequential(M.ConvTranspose2d(nc[2], nc[1], kernel_size=2, stride=2, bias=False),
                                   *[ResBlock(nc[1], nc[1]) for _ in range(nb)])
        self.m_up1 = M.Sequential(M.ConvTranspose2d(nc[1], nc[0], kernel_size=2, stride=2, bias=False),
                                   *[ResBlock(nc[0], nc[0]) for _ in range(nb)])

        self.m_tail = M.Conv2d(nc[0], out_nc, kernel_size=3, stride=1, padding=1, bias=False)

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
    import numpy as np
    import megengine as mge
    from megengine.utils.module_stats import module_stats

    input_data = np.random.rand(1, 1, 256, 256).astype("float32")
    # input_data = mge.tensor(input_data)
    #
    #
    net = UNetRes(in_nc=1, out_nc=1, nc=[10, 14, 14, 16], nb=4)

    total_stats, stats_details = module_stats(
        net,
        inputs=(input_data,),
        cal_params=True,
        cal_flops=True,
        logging_to_stdout=True,
    )
    print("params %.3fK MAC/pixel %.0f" % (
    total_stats.param_dims / 1e3, total_stats.flops / input_data.shape[2] / input_data.shape[3]))

