# A Light-weight model for Practical Blind Raw Image Denoising (PBRID)

![visitors](https://visitor-badge.glitch.me/badge?page_id=hellopipu/PBRID) 

A baseline light-weight model for practical blind raw image denoising (PBRID)

Raw (RGGB) image denoising results on test set:

<table>
  <tr>
    <td><img src="/figs/22_noise.png?raw=true" width="500"></td>
    <td><img src="/figs/22_pred.png?raw=true" width="500"></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="/figs/42_noise.png?raw=true" width="500"></td>
    <td><img src="/figs/42_pred.png?raw=true" width="500"></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="/figs/58_noise.png?raw=true" width="500"></td>
    <td><img src="/figs/58_pred.png?raw=true" width="500"></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="/figs/140_noise.png?raw=true" width="500"></td>
    <td><img src="/figs/140_pred.png?raw=true" width="500"></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="/figs/204_noise.png?raw=true" width="500"></td>
    <td><img src="/figs/204_pred.png?raw=true" width="500"></td>
  </tr>
</table>


# model Param |  inference speed (img 256x256, single V100)
-------|-------
97.2K| 10.1ms

## Dataset & pretrained model
```shell
## download dataset
wget -nc https://rutgers.box.com/shared/static/tx3s87qdcx3g8vc62ukf4hk56h0fwn2f.zip -O burst_raw.zip
unzip burst_raw.zip
## download pretrained model file (includes model, ema model and optimizer)
wget -nc https://rutgers.box.com/shared/static/5cluf6gdevwsi7samjkytlt15f3zziaa.pth -P weight/
mv weight/5cluf6gdevwsi7samjkytlt15f3zziaa.pth weight/hqs.pth
```

## Run scripts
```shell
## train:
CUDA_VISIBLE_DEVICES=0 python main.py --mode 'train'
## val:
CUDA_VISIBLE_DEVICES=1 python main.py --mode 'val'
## test:
CUDA_VISIBLE_DEVICES=1 python main.py --mode 'test'
```
