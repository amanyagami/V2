# train_wds.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import webdataset as wds
from torchvision import transforms, models
from PIL import Image

# --------- USER CONFIG ----------
SHARD_PATTERN = os.environ.get("SHARD_PATTERN", "s3://my-bucket/imagenet/imagenet-{000000..000255}.tar")
BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 5
LR = 0.01
IMAGE_KEY = "jpg"    # key used inside the shards for the image bytes
LABEL_KEY = "cls"    # key used inside the shards for the label
SHUFFLE_BUFFER = 1000
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms (standard ImageNet)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def pil_decode(b):
    return Image.open(b).convert("RGB")

def make_dataset(shard_pattern):
    """
    Create a streaming WebDataset pipeline.
    Assumes each sample in shard has keys: IMAGE_KEY (jpg), LABEL_KEY (cls)
    where the label is an integer (or string convertible to int).
    """
    dataset = (
        wds.WebDataset(shard_pattern, handler=wds.handlers.warn_and_continue)
           .shuffle(SHUFFLE_BUFFER)        # sample-level shuffle buffer
           .decode("pil")                  # decode bytes -> PIL.Image
           .to_tuple(IMAGE_KEY, LABEL_KEY) # pick which keys to return
           .map_tuple(transform, lambda x: int(x))  # apply transforms & convert label
    )
    return dataset

def make_dataloader(shard_pattern):
    dataset = make_dataset(shard_pattern)
    # webdataset yields endless iterator semantics; wrap with DataLoader for batching
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return loader

def train_one_epoch(model, loader, opt, epoch):
    model.train()
    total = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()

        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        if i % 50 == 0:
            print(f"Epoch {epoch} Iter {i} Loss {loss.item():.4f} Acc {100*correct/total:.2f}%")

def main():
    # small model example - replace with your model / amp / optimizer
    model = models.resnet18(pretrained=False, num_classes=1000)
    model = model.to(device)

    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

    loader = make_dataloader(SHARD_PATTERN)

    # NOTE: webdataset produces an infinite stream; for epoch semantics we'll iterate a fixed
    # number of steps treated as one epoch. Alternatively, you can use .with_epoch() features.
    steps_per_epoch = 1000  # tune for your dataset and BATCH_SIZE
    for epoch in range(EPOCHS):
        # Here we break the DataLoader iterator after steps_per_epoch batches to emulate an epoch.
        it = iter(loader)
        for step in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step}/{steps_per_epoch} Loss {loss.item():.4f}")

        # Optionally adjust LR, save checkpoints, etc.
        torch.save(model.state_dict(), f"model_epoch{epoch}.pth")
        print(f"Saved checkpoint model_epoch{epoch}.pth")

if __name__ == "__main__":
    main()
