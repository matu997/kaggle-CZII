import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from decoder import *


# Tversky損失の実装
def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1e-6):
    """
    Tversky損失を計算する関数

    Parameters:
    - pred: モデルの出力 (B, C, D, H, W) -> 確率分布 (softmax適用済み)
    - target: 正解ラベル (B, D, H, W) -> int形式のラベル
    - alpha: False Positive (FP) に対するペナルティ係数
    - beta: False Negative (FN) に対するペナルティ係数
    - smooth: 数値安定化のための小さな値

    Returns:
    - Tversky損失
    """
    pred = F.softmax(pred, dim=1)  # 確率分布に変換
    num_classes = pred.size(1)
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    TP = (pred * target_onehot).sum(dim=(2, 3, 4))  # True Positive
    FP = ((1 - target_onehot) * pred).sum(dim=(2, 3, 4))  # False Positive
    FN = (target_onehot * (1 - pred)).sum(dim=(2, 3, 4))  # False Negative

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss = 1 - tversky.mean()
    return loss


#------------------------------------------------
# processing
def encode_for_resnet(e, x, B, depth_scaling=[2, 2, 2, 2, 1]):
    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode = []
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x, x1 = pool_in_depth(x, depth_scaling[0])
    encode.append(x1)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)

    x = e.layer1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])
    encode.append(x1)

    x = e.layer2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])
    encode.append(x1)

    x = e.layer3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)

    x = e.layer4(x)
    x, x1 = pool_in_depth(x, depth_scaling[4])
    encode.append(x1)

    return encode


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))

        num_class = 6 + 1
        self.arch = 'resnet34d' if cfg is None else cfg.arch

        encoder_dim = {
            'resnet34d': [64, 64, 128, 256, 512],
            'resnet50d': [64, 256, 512, 1024, 2048],
        }.get(self.arch, [768])

        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )

        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
        )
        self.mask = nn.Conv3d(decoder_dim[-1], num_class, kernel_size=1)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)
        B, D, H, W = image.shape
        image = image.reshape(B * D, 1, H, W)

        x = (image.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2, 2, 2, 2, 1])

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None], depth_scaling=[1, 2, 2, 2, 2]
        )

        logit = self.mask(last)

        output = {}
        if 'loss' in self.output_type:
            output['mask_loss'] = tversky_loss(
                pred=logit, target=batch['label'].to(device), alpha=0.5, beta=0.95
            )

        if 'infer' in self.output_type:
            output['particle'] = F.softmax(logit, 1)

        return output


#------------------------------------------------------------------------
def run_check_net():
    B = 4
    image_size = 640
    mask_size = 640
    num_slice = 32
    num_class = 6 + 1

    batch = {
        'image': torch.from_numpy(np.random.uniform(0, 1, (B, num_slice, image_size, image_size))).float(),
        'label': torch.from_numpy(np.random.choice(num_class, (B, num_slice, mask_size, mask_size))).long(),
    }
    net = Net(pretrained=True, cfg=None).cuda()

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            output = net(batch)

    print('batch')
    for k, v in batch.items():
        print(f'{k:>32} : {v.shape} ')

    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print(f'{k:>32} : {v.shape} ')
    print('loss')
    for k, v in output.items():
        if 'loss' in k:
            print(f'{k:>32} : {v.item()} ')


# main #################################################################
if __name__ == '__main__':
    run_check_net()
