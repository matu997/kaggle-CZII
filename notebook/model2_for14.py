import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import timm
#from timm.models.convnext import *

from decoder import *

#------------------------------------------------
# Jensen-Shannon ダイバージェンス損失の定義
def jensen_shannon_divergence_loss(logits, mask, eps=1e-8):
    """
    ※この外側の関数でラベル(mask)をdetach()してfloat()に変換し、
      内側の本来の計算部分は『一切変更しない』形にしています。
    """

    # --- ここでラベルを浮動小数化＆勾配を切り離し ---
    mask_ = mask.detach().float()

    #--- 以下に「元の実装部分」を全く同じにして埋め込み ---
    def _original_js_loss(logits, mask, eps=1e-8):
        """
        logits: (B, 7, D, H, W) - ネットワーク出力 (生のロジット)
        mask:   (B, 7, D, H, W) - 連続値のラベル(確率分布を想定)
        eps:    log(0) 回避のための小さい値

        1) Q = softmax(logits, dim=1)
        2) P = mask を [eps,1] にクランプして確率分布とみなす
        3) M = 0.5*(P + Q)
        4) KL(P||M) = ∑ P * log(P/M) をチャネル方向で合計
        5) JS = 0.5*(KL(P||M) + KL(Q||M))
        6) 全ボクセル平均
        """
        # ラベルを確率分布に変換
        mask = F.softmax(mask, dim=1)  
        # ネットワーク予測確率 Q
        Q = F.softmax(logits, dim=1).clamp(min=eps, max=1.0) 
        # ラベル分布 P
        P = mask.clamp(min=eps, max=1.0)                     
        # 中間分布 M
        M = 0.5 * (P + Q)

        # KL(P||M), KL(Q||M) をチャネル方向(dim=1) で合計
        KL_PM = (P * (torch.log(P) - torch.log(M))).sum(dim=1)
        KL_QM = (Q * (torch.log(Q) - torch.log(M))).sum(dim=1)

        # JS = 0.5 * (KL(P||M) + KL(Q||M))
        JS = 0.5 * KL_PM + 0.5 * KL_QM

        # 全ボクセル平均
        #loss = JS.mean()
        #全ポクセルの和
        loss = JS.sum()
        loss = loss / 500000  # バッチサイズで割る
        return loss
    #--- ここまで ---

    # ラベルだけ処理したものを、上記の「元の実装」にそのまま渡す
    return _original_js_loss(logits, mask_, eps)


#------------------------------------------------
# processing
def encode_for_resnet(e, x, B, depth_scaling=[2,2,2,2,1]):

    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode=[]
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x, x1 = pool_in_depth(x, depth_scaling[0])
    encode.append(x1)
    #print(x.shape)
    #x = e.maxpool(x)
    x = F.avg_pool2d(x,kernel_size=2,stride=2)

    x = e.layer1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])
    encode.append(x1)
    #print(x.shape)

    x = e.layer2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])
    encode.append(x1)
    #print(x.shape)

    x = e.layer3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)
    #print(x.shape)

    x = e.layer4(x)
    x, x1 = pool_in_depth(x, depth_scaling[4])
    encode.append(x1)
    #print(x.shape)

    return encode


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss', ]
        self.register_buffer('D', torch.tensor(0))

        # 今回の例では 7 クラス (6 + 1) と仮定
        num_class = 6 + 1

        self.arch = 'resnet34d'
        if cfg is not None:
            self.arch = cfg.arch

        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
            'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
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
        image = image.reshape(B*D, 1, H, W)

        x = (image.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        # ResNet の出力 (マルチスケール特徴) を取得
        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2,1])
        
        # デコーダに入力（skip connection の長さを合わせるために逆順にしている）
        last, decode = self.decoder(
            feature=encode[-1], 
            skip=encode[:-1][::-1] + [None], 
            depth_scaling=[1,2,2,2,2]
        )

        # 3D Conv で最終出力(logits) を作る
        logit = self.mask(last)

        output = {}
        if 'loss' in self.output_type:
            # ここを JS ダイバージェンス損失に置き換え
            # 7クラス想定なので num_classes=7 を指定
            output['mask_loss'] = jensen_shannon_divergence_loss(
                logits=logit, 
                mask=batch['label'].to(device)
            )

        if 'infer' in self.output_type:
            # 予測確率 (softmax)
            output['particle'] = F.softmax(logit, 1)

        return output


#------------------------------------------------------------------------
def run_check_net():
    B = 4
    image_size = 640
    mask_size  = 640
    num_slice = 32  # 例: 3D方向に32スライス
    num_class = 6 + 1

    batch = {
        'image': torch.from_numpy(np.random.uniform(0,1, (B,num_slice, image_size, image_size))).float(),
        'label': torch.from_numpy(np.random.randint(0, num_class, (B,num_slice, image_size, image_size))).long(),
    }

    net = Net(pretrained=False, cfg=None).cuda()

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            output = net(batch)

    # --- 出力を確認
    print('batch')
    for k, v in batch.items():
        if k == 'D':
            print(f'{k:>32} : {v}')
        else:
            print(f'{k:>32} : {v.shape}')

    print('\noutput')
    for k, v in output.items():
        if 'loss' not in k:
            print(f'{k:>32} : {v.shape}')
    print('loss')
    for k, v in output.items():
        if 'loss' in k:
            print(f'{k:>32} : {v.item()}')


if __name__ == '__main__':
    run_check_net()
