# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

from BasicModule import UNetRes
import megengine.module as M
import megengine.functional as F

class HQSNet(M.Module):
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
        self.ee = M.Sequential(
            M.Conv2d(1, 16, 8, 8),
            M.ReLU(),
            M.Conv2d(16, 16, 8, 8, groups=16),
            M.ReLU(),
            M.AdaptiveAvgPool2d((1,1)),
            M.Conv2d(16, 1, 1, 1),
            M.Sigmoid()
        )

    def update_opration(self, pred, img):
        update = pred + self.mu * (img - pred)
        return update

    def forward(self, img):
        '''import megengine.functional as F
        :param img: noisy raw img, (batch,1,h,w)
        :return: de-noised img
        '''
        self.mu = self.ee(img)

        ## initialize buffer f : the concatenation of m copies of noisy images
        f = F.concat([img] * self.m, 1).to(img.device)

        ## n reconstruction blocks
        for i in range(self.n_iter):
            f_1 = f[:, 0:1]#.clone()
            updated_f_1 = self.update_opration(f_1, img)
            f = f + self.rec_blocks(F.concat([f, updated_f_1], 1))
        return f[:, 0:1]


if __name__ == '__main__':
    import numpy as np
    import megengine as mge
    from megengine.utils.module_stats import module_stats

    input_data = np.random.rand(1, 1, 256, 256).astype("float32")
    
    net = HQSNet(buffer_size=7, n_iter=10)

    total_stats, stats_details = module_stats(
        net,
        inputs=(input_data,),
        cal_params=True,
        cal_flops=True,
        logging_to_stdout=True,
    )
    print("params %.3fK MAC/pixel %.0f" % (
    total_stats.param_dims / 1e3, total_stats.flops / input_data.shape[2] / input_data.shape[3]))

