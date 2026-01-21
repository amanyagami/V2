# train_mobilevit_webdataset.py
import os
import time
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

# webdataset streaming
import webdataset as wds

# mixed precision
from torch.cuda.amp import GradScaler, autocast

# ---- Hyperparameters ----
BATCH_SIZE = 256           # effective batch size; reduce for single GPU
NUM_WORKERS = 8
IMAGE_SIZE = 224
NUM_CLASSES = 1000
EPOCHS = 90
LR = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
MIXUP_ALPHA = 0.2         # set to 0 to disable mixup

# Checkpoint directory
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Transforms ----
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE/0.875)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ---- WebDataset helpers ----
def make_webdataset_pipeline(shard_urls: Iterable[str], transform, shuffle: bool = True, epoch_shuffle: bool = True):
    """
    shard_urls: list of "file://..." or "s3://..." or local tar paths (e.g., ["data/imagenet-000.tar", ...]).
    Expect tar members to have names: <key>.jpg and <key>.cls or similar. We'll parse image bytes and class index.
    """
    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=shuffle)             # read shards
           .decode("pil")                                            # decode JPEG -> PIL.Image
           .to_tuple("jpg", "cls")                                   # expects 'jpg' and 'cls' members
           .map_tuple(transform, lambda x: int(x))                   # apply transform to PIL image, convert class to int
    )
    return dataset

# collate for mixup support
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.tensor(targets, dtype=torch.long)
    return images, targets

# ---- Mixup util (simple) ----
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---- Build model, optimizer, scheduler ----
def build_model():
    # Choose model name available in timm; check timm model list for exact MobileViT variants
    model = timm.create_model("mobilevit_xxs", pretrained=False, num_classes=NUM_CLASSES)
    return model.cuda()

def build_optimizer(model):
    # AdamW is common for ViT / mobile transformers
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return opt

def build_scheduler(optimizer, total_steps_per_epoch):
    # Simple cosine scheduler with linear warmup
    def lr_lambda(current_step):
        current_epoch = current_step / total_steps_per_epoch
        if current_epoch < WARMUP_EPOCHS:
            return float(current_epoch / float(max(1, WARMUP_EPOCHS)))
        # cosine after warmup
        progress = (current_epoch - WARMUP_EPOCHS) / float(max(1, EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

# ---- Training / Validation loops ----
import numpy as np, math
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # optional: mixup
        if MIXUP_ALPHA > 0:
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            idx = torch.randperm(images.size(0)).to(device)
            images = lam * images + (1 - lam) * images[idx]
            targets_a, targets_b = targets, targets[idx]
            use_mixup = True
        else:
            use_mixup = False

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            if use_mixup:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        if use_mixup:
            # for accuracy, count using targets_a only as approximation or skip accuracy during mixup
            correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    pbar = tqdm(loader, desc="val", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
        correct5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()

    avg_loss = running_loss / total
    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    return avg_loss, top1, top5

# ---- Main training entrypoint ----
def main(shard_list_train, shard_list_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # streaming datasets
    train_ds = make_webdataset_pipeline(shard_list_train, train_tfms, shuffle=True)
    val_ds = make_webdataset_pipeline(shard_list_val, val_tfms, shuffle=False)

    # DataLoader: webdataset is iterable => use IterableDataset semantics
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = build_model()
    optimizer = build_optimizer(model)
    total_steps_per_epoch = 1281167 // BATCH_SIZE  # approximate; used only by scheduler's lambda if needed
    scheduler = build_scheduler(optimizer, total_steps_per_epoch)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    best_top1 = 0.0
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
              f"val_loss={val_loss:.4f} val1={val_top1:.2f} val5={val_top5:.2f} epoch_time={time.time()-t0:.1f}s")

        # checkpoint
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        ckpt_path = CKPT_DIR / f"mobilevit_epoch{epoch}.pth"
        torch.save(state, ckpt_path)

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(state, CKPT_DIR / "mobilevit_best.pth")

if __name__ == "__main__":
    # Example: local tar files
    # You must prepare shards where each .tar has pairs like: <sampleid>.jpg , <sampleid>.cls (text file with integer class)
    train_shards = ["data/imagenet-train-000.tar", "data/imagenet-train-001.tar", "..."]
    val_shards = ["data/imagenet-val-000.tar", "..."]
    main(train_shards, val_shards)
