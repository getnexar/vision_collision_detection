import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def cleanup():
    dist.destroy_process_group()

def main():
    rank = setup()
    
    # Simple model
    model = nn.Linear(10, 1).cuda()
    ddp_model = DDP(model, device_ids=[rank])
    
    print(f"Rank {rank}: DDP initialization successful!")
    
    cleanup()

if __name__ == "__main__":
    main()
