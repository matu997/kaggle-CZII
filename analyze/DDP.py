import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def main():
    # torch.distributed.run で起動した場合、環境変数からランクとワールドサイズを取得する
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # プロセスグループの初期化
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Rank {rank}: init_process_group 完了", flush=True)

    # デバイスの設定
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    
    # モデル作成＆DDPラップ
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # ダミー入力による計算
    input_tensor = torch.randn(20, 10).to(device)
    output = ddp_model(input_tensor)
    loss = output.sum()
    loss.backward()
    
    for name, param in ddp_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Rank {rank} - {name} の勾配ノルム: {grad_norm}", flush=True)

    dist.destroy_process_group()
    print(f"Rank {rank}: DDPプロセスを終了します。", flush=True)

if __name__ == '__main__':
    main()