import os

# NCCL環境変数を設定
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import timm

# データセットクラス
class RandomDataset(Dataset):
    def __init__(self, num_samples, image_size, mask_size, num_slices, num_classes):
        self.num_samples = num_samples
        self.image_size = image_size
        self.mask_size = mask_size
        self.num_slices = num_slices
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = torch.rand(self.num_slices, self.image_size, self.image_size).float()
        mask = torch.randint(0, self.num_classes, (self.num_slices, self.mask_size, self.mask_size)).long()
        return {'image': image, 'mask': mask}

# エンコーダのヘルパー関数
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

# ネットワーククラス
class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))

        num_class = 6 + 1
        self.arch = 'resnet34d'
        encoder_dim = {
            'resnet34d': [64, 64, 128, 256, 512],
        }.get(self.arch, [768])

        decoder_dim = [384, 192, 96, 32, 16]
        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(encoder_dim[-1], decoder_dim[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(decoder_dim[0], num_class, kernel_size=1)
        )

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)
        B, D, H, W = image.shape
        image = image.reshape(B * D, 1, H, W)
        x = (image.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2, 2, 2, 2, 1])
        last = encode[-1].mean(dim=1, keepdim=True)  # 簡略化
        logit = self.decoder(last)

        output = {}
        if 'loss' in self.output_type:
            output['mask_loss'] = F.cross_entropy(logit, batch['mask'].to(device))
        if 'infer' in self.output_type:
            output['particle'] = F.softmax(logit, 1)
        return output

# 分散処理の初期化
def setup_distributed(rank, world_size):
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank
    )

# 学習関数
def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)

    # デバイス設定
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # データセットとデータローダー
    dataset = RandomDataset(num_samples=100, image_size=640, mask_size=640, num_slices=32, num_classes=7)
    sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=4, sampler=sampler)

    # モデルと最適化
    net = Net(pretrained=False).to(device)
    net = DDP(net, device_ids=[rank], output_device=rank)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # 学習ループ
    net.train()
    for epoch in range(5):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(batch)
            loss = output['mask_loss']
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch [{epoch + 1}/5], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Rank {rank}, Epoch [{epoch + 1}/5], Average Loss: {epoch_loss / len(train_loader):.4f}")

    dist.destroy_process_group()

# エントリーポイント
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
