
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Subset
import torch

def get_id_dataloaders(id_name,batch_size = 512):
    data_root='./data' 
    if id_name == 'svhn':
        input_size = (32, 32) 
        transform_id = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)) 
        ])
    elif id_name == 'gtsrb':
        input_size = (32, 32) 
        transform_id = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.34172177, 0.3125682 , 0.3215711 ), (0.21573113, 0.210904  , 0.21959049 )) 
        ])
    elif id_name == 'cifar10':
        input_size = (32, 32) 
        transform_id  = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.ToTensor(),   
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))   
        ])
    elif id_name in [ 'mnist' ]:
        input_size = (32, 32) 
        transform_id=  transforms.Compose([
            transforms.Resize(input_size), 
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
        ])
    elif id_name == 'cifar100':
        input_size = (32, 32) 
        transform_id  = transforms.Compose([
            transforms.Resize(input_size),   
            transforms.ToTensor(),   
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])        
    datasets_list = {
        'cifar10': (
            datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_id),
            datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_id)
        ),
        'cifar100': (
            datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_id),
            datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_id)
        ),
        'mnist': (
            datasets.MNIST(root=data_root, train=True, download=True, transform=transform_id),
            datasets.MNIST(root=data_root, train=False, download=True, transform=transform_id)
        ),
        'qmnist': (
            datasets.QMNIST(root=data_root, train=True, download=True, transform=transform_id),
            datasets.QMNIST(root=data_root, train=False, download=True, transform=transform_id)
        ),
        'kmnist': (
            datasets.KMNIST(root=data_root, train=True, download=True, transform=transform_id),
            datasets.KMNIST(root=data_root, train=False, download=True, transform=transform_id)
        ),
        'fmnist': (
            datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform_id),
            datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform_id)
        ),
        'svhn': (
            datasets.SVHN(root=data_root, split='train', download=True, transform=transform_id),
            datasets.SVHN(root=data_root, split='test', download=True, transform=transform_id)
        ),
        'gtsrb':(
            datasets.GTSRB(root=data_root, split='train', download=True, transform=transform_id),
            datasets.GTSRB(root=data_root, split='test', download=True, transform=transform_id)
        )
    }        
    train_dataset, test_dataset = datasets_list[id_name]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=4 )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
    return { "ID_train" : train_loader, "ID_test": test_loader }

def get_random_subset(dataset, num_samples, seed=99):
    generator = torch.Generator().manual_seed(seed)
    dataset_size = len(dataset)
    # print(dataset_size, " == dataset size")
    sample_size = min(num_samples, dataset_size)
    sampler = RandomSampler(dataset, num_samples=sample_size, generator=generator)
    indices = list(sampler)
    return Subset(dataset, indices)

def get_ADV_DataLoaders(m_name, id_name,seed = 99,batch_size = 512, num_samples = 2000 ):
    optim_type = "SGD"
    def create_adversarial_loader(file_path, num_samples, batch_size, seed=99):
        data = torch.load(file_path, weights_only=False)
        data_list, label_list = zip(*data)
        inputs_tensor = torch.stack([torch.from_numpy(data).float() for data in data_list])
        labels_tensor = torch.tensor(label_list, dtype=torch.long)
        dataset = TensorDataset(inputs_tensor, labels_tensor)
        subset = get_random_subset(dataset, num_samples, seed)
        return DataLoader(subset, batch_size=batch_size, shuffle=False)
    file_loc = f"./adv_samples2/{optim_type}/{m_name}/{m_name}_{id_name}"
    fgsm_test_loader = create_adversarial_loader(file_loc + '_fgsm.pt', num_samples, batch_size, seed = seed)
    pgd_test_loader = create_adversarial_loader(file_loc + '_pgd.pt', num_samples, batch_size, seed = seed)
    cw_test_loader = create_adversarial_loader(file_loc + '_cw.pt', num_samples, batch_size, seed = seed)
    deepfool_test_loader = create_adversarial_loader(file_loc + '_deepfool.pt', num_samples, batch_size, seed = seed)
    autoattack_test_loader =  create_adversarial_loader(file_loc + '_autoattack.pt', num_samples, batch_size, seed = seed)
    return { "fgsm":fgsm_test_loader, "pgd":pgd_test_loader, "cw":cw_test_loader, "deepfool":deepfool_test_loader, "autoattack":autoattack_test_loader }


def get_OOD_Dataloaders(m_name, id_name,seed = 99,batch_size= 512,type = "Standard", num_samples=2000 ):
    data_root='./data' 
    # print(" number of samples ajabawgwbrb = ",num_samples)
    ood_types = ['cifar10',"cifar100","isun","LSUNResize","lsun","svhn","dtd","inaturalist","mnist","kmnist","qmnist","fmnist","places365","gtsrb"]  
    if type !=  "Standard": 
        ood_types = ["GaussianNoise","UniformNoise"]
    elif type == "Standard":
        if id_name == "cifar10":
            ood_types = ["cifar100","places365","mnist","lsun","isun","gtsrb"]
        elif id_name == "cifar100":
            ood_types = ["places365","mnist","lsun","isun","gtsrb","cifar10"]
        elif id_name == "svhn":
            ood_types = ["cifar100","places365","lsun",'cifar10',"isun","cifar100"] 
        elif id_name == "mnist":
            ood_types = ["places365",'cifar10',"isun","lsun","gtsrb","cifar100"]
        elif id_name == "gtsrb":
            ood_types = ["cifar10","isun","lsun","mnist","cifar100","places365"]
    if id_name == 'svhn':
        input_size = (32, 32) 
        transform_test = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))   
        ])
        transform_test_bw = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  
        ])
    elif id_name == 'gtsrb':
        input_size = (32, 32)         
        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.34172177, 0.3125682 , 0.3215711 ), (0.21573113, 0.210904  , 0.21959049 )) 
        ])
        transform_test_bw = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.34172177, 0.3125682 , 0.3215711 ), (0.21573113, 0.210904  , 0.21959049 )) 
        ])    
    elif id_name == 'cifar10':
        input_size = (32, 32)  
        transform_test  = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.ToTensor(),   
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))   
        ])
        transform_test_bw  = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),   
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))   
        ])
    elif id_name in [ 'mnist' ]:
        input_size = (32, 32)          
        transform_test_bw  =  transforms.Compose([
            transforms.Resize(input_size), 
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
        ])
        transform_test  =  transforms.Compose([
            transforms.Resize(input_size), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
        ])
    elif id_name == 'cifar100':
        input_size = (32, 32)         
        transform_test = transforms.Compose([
            transforms.Resize(input_size),  
            transforms.ToTensor(),   
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test_bw = transforms.Compose([
            transforms.Resize(input_size), 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),   
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
    download_here=True
    colorfull_ood = {
    # "TinyImage300k": TinyImages300k(root=data_root,download= download_here, transform= transform_test),
    # "UniformNoise": UniformNoise(length=3000,size= (224,224,3), transform = transform_test),
    # "GaussianNoise": GaussianNoise(length=3000,size= (224,224,3), transform = transform_test),
    'cifar10': datasets.CIFAR10(root=data_root, train=False, download=download_here, transform=transform_test),
    'cifar100': datasets.CIFAR100(root=data_root, train=False, download=download_here, transform=transform_test),
    'svhn': datasets.SVHN(root=data_root, split='test', download=download_here, transform=transform_test),
    'gtsrb': datasets.GTSRB(root=data_root, split='test', download=download_here, transform=transform_test),
    "isun": datasets.ImageFolder(root=f"{data_root}/isun", transform=transform_test),
    "lsun": datasets.ImageFolder(root=f"{data_root}/lsun_resize", transform=transform_test),
    'pcam': datasets.PCAM(root=data_root, split='test', download=download_here, transform=transform_test),
    'dtd': datasets.DTD(root=data_root, split='test', download=download_here, transform=transform_test),
    "places365":datasets.Places365(root = data_root, split= "val",small=True, download=download_here, transform=transform_test),
    "inaturalist":datasets.INaturalist(root=data_root, version="2021_valid",download= download_here, transform=transform_test) 
    # "LSUNResize": LSUNResize(root = data_root, download= download_here,transform=transform_test),
    }
    bw_ood = {
    'mnist': datasets.MNIST(root=data_root, train=False, download=False, transform=transform_test_bw),
    'qmnist': datasets.QMNIST(root=data_root, train=False, download=False, transform=transform_test_bw),
    'kmnist': datasets.KMNIST(root=data_root, train=False, download=False, transform=transform_test_bw),
    'fmnist': datasets.FashionMNIST(root=data_root, train=False, download=False, transform=transform_test_bw),
    }
    # print(" number of samples ajaba = ",num_samples)
    ood_loaders = {}
    for ood in ood_types :
        if ood in colorfull_ood:
            if num_samples == -1:
                # print("--aevavav- ood = ",ood)
                ood_subset = colorfull_ood[ood]
            else:
                # print("-rgrsrrs- ood = ",ood)
                ood_subset = get_random_subset(colorfull_ood[ood],num_samples,seed)
            ood_loaders[ood] = DataLoader(ood_subset,batch_size,shuffle=True)
        else:
            if num_samples == -1:
                # print("--rgvaca- ood = ",ood)
                ood_subset = bw_ood[ood]
            else:
                # print("------- ood = ",ood)
                ood_subset = get_random_subset(bw_ood[ood],num_samples,seed)
            ood_loaders[ood] = DataLoader(ood_subset,batch_size,shuffle=True)     
    return ood_loaders,ood_types
