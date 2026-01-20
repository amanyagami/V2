import os
from torchvision import datasets

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

MARKER = os.path.join(DATA_DIR, ".download_complete")


if not os.path.exists(MARKER):
    os.makedirs(DATA_DIR, exist_ok=True)
    # MNIST
    datasets.MNIST(root=DATA_DIR, train=True, download=True)
    datasets.MNIST(root=DATA_DIR, train=False, download=True)

    # Fashion-MNIST
    datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)
    datasets.FashionMNIST(root=DATA_DIR, train=False, download=True)

    # KMNIST
    datasets.KMNIST(root=DATA_DIR, train=True, download=True)
    datasets.KMNIST(root=DATA_DIR, train=False, download=True)

    # QMNIST
    datasets.QMNIST(root=DATA_DIR, train=True, download=True)
    datasets.QMNIST(root=DATA_DIR, train=False, download=True)

    # CIFAR-10
    datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    # CIFAR-100
    datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
    datasets.CIFAR100(root=DATA_DIR, train=False, download=True)

    # SVHN
    datasets.SVHN(root=DATA_DIR, split="train", download=True)
    datasets.SVHN(root=DATA_DIR, split="test", download=True)
    datasets.SVHN(root=DATA_DIR, split="extra", download=True)

    # GTSRB
    datasets.GTSRB(root=DATA_DIR, split="train", download=True)
    datasets.GTSRB(root=DATA_DIR, split="test", download=True)

    #Places365
    datasets.Places365( DATA_DIR,split="train-standard",small=True, download=True)
    datasets.Places365( DATA_DIR,split="val", small=True, download=True)

    #DTD 
    datasets.DTD(DATA_DIR, split="train", download=True)
    datasets.DTD(DATA_DIR, split="val", download=True)
    datasets.DTD(DATA_DIR, split="test", download=True)

    #LSUN Resize
    datasets.LSUNResize(DATA_DIR, download=True)
    with open(MARKER, "w") as f:
        f.write("done")
    print("Datasets downloaded.")
else :
    print("Datasets already present. Skipping download.")
