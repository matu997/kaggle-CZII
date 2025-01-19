import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.nn as nn
import torch.optim as optim
import subprocess

# カスタムデータセットの定義
class RandomDataset(Dataset):
    def __init__(self, input_size, output_size, data_size):
        self.data = torch.randn(data_size, input_size)
        self.labels = torch.randn(data_size, output_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# シンプルなモデルの定義
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# GPU使用状況の確認関数
def print_gpu_utilization(rank):
    # torch.cuda を使ったメモリ使用量の確認
    allocated = torch.cuda.memory_allocated(rank) / (1024 ** 3)  # GB単位
    reserved = torch.cuda.memory_reserved(rank) / (1024 ** 3)  # GB単位
    print(f"[Rank {rank}] Allocated GPU Memory: {allocated:.2f} GB")
    print(f"[Rank {rank}] Reserved GPU Memory: {reserved:.2f} GB")
    

# トレーニングループ
def train(rank, world_size, input_size, output_size, data_size, batch_size):
    # 分散プロセスグループの初期化
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # データセットとデータローダーの作成
    dataset = RandomDataset(input_size, output_size, data_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # モデルの定義
    model = Model(input_size, output_size).to(device)
    model = DDP(model, device_ids=[rank])

    # 損失関数とオプティマイザ
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # トレーニングループ
    for epoch in range(2):  # エポック数を指定
        sampler.set_epoch(epoch)  # サンプラのシャッフルシードを変更
        for batch_idx, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)

            # フォワードパス
            output = model(data)
            loss = loss_fn(output, label)

            # 勾配の更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # GPU使用状況のデバッグ
            print_gpu_utilization(rank)

            if rank == 0:  # 主プロセスのみ損失を表示
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}")

    # プロセスグループの終了
    dist.destroy_process_group()

# エントリポイント
def main():
    # NCCL設定
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_PROTO"] = "AUTO"
    os.environ["NCCL_ALGO"] = "AUTO"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

    # MASTER_ADDRとMASTER_PORTを設定
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 設定値
    world_size = torch.cuda.device_count()
    input_size = 2048
    output_size = 1024
    data_size = 10000
    batch_size = 256

    print(f"Using {world_size} GPUs for DDP training.")

    # マルチプロセス実行
    mp.spawn(train,
             args=(world_size, input_size, output_size, data_size, batch_size),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
