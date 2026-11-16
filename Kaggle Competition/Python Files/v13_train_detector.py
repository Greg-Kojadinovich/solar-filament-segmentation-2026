"""Distributed Mask R-CNN training for V13 (run with torchrun on two GPUs)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v13_lib as lib  # noqa: E402


def seed_worker(worker_id):
    import random

    import numpy as np
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()
    work = Path(args.work)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lib.seed_everything(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    manifest = json.loads((work / "manifest.json").read_text())
    train_rows = [r for r in manifest if r["split"] == "train"]
    val_rows = [r for r in manifest if r["split"] == "val"]
    train_dataset = lib.FilamentDataset(train_rows, args.annotations, args.images, augment=True)
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    options = dict(num_workers=2, pin_memory=True, persistent_workers=True,
                   collate_fn=lib.collate_fn, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, **options)
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(lib.FilamentDataset(val_rows, args.annotations, args.images),
                                batch_size=1, shuffle=False, **options)

    model = DDP(lib.build_detector(pretrained=True).to(device), device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, momentum=0.9, weight_decay=1e-4)
    milestones = sorted({max(1, round(args.epochs * 0.70)), max(2, round(args.epochs * 0.90))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    history, best, started_all = [], -1.0, time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        started = time.perf_counter()
        model.train()
        total = torch.zeros(2, device=device)
        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch:02d}", disable=rank != 0, leave=False):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = sum(model(images, targets).values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total[0] += loss.detach()
            total[1] += 1
        dist.all_reduce(total)
        train_seconds = time.perf_counter() - started
        metrics = lib.evaluate_detector(model.module, val_loader, device) if rank == 0 else {}
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            row = dict(epoch=epoch, train_loss=(total[0] / total[1]).item(), lr=optimizer.param_groups[0]["lr"],
                       train_minutes=train_seconds / 60, minutes=(time.perf_counter() - started) / 60, **metrics)
            history.append(row)
            if metrics["soft_dice"] > best:
                best = metrics["soft_dice"]
                torch.save({"epoch": epoch, "model": model.module.state_dict()}, work / "best_detector.pt")
            print(f"DETECTOR {epoch:02d}/{args.epochs} | loss {row['train_loss']:.4f} | val soft Dice "
                  f"{metrics['soft_dice']:.4f} | val PQ {metrics['pq']:.4f} | {metrics['predictions']:.1f} masks/img "
                  f"| train {row['train_minutes']:.2f} min | epoch {row['minutes']:.2f} min", flush=True)
        scheduler.step()
        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        torch.save({"epoch": args.epochs, "model": model.module.state_dict()}, work / "last_detector.pt")
        (work / "detector_history.json").write_text(json.dumps(history, indent=2))
        print(f"DETECTOR total {(time.perf_counter() - started_all) / 60:.1f} min", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
