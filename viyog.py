import torch 
from pytorch_ood.utils import OODMetrics
import numpy as np 
from torch import nn
import cupy as cp
import copy
from datasets.get_loaders import get_id_dataloaders,get_OOD_Dataloaders,get_ADV_DataLoaders
import time


def ViyogT1(num, Temperature = 1):
    num = num/Temperature
    sign = torch.sign(num)
    num = torch.exp(torch.absolute(num))
    num = sign / (1.0 + torch.exp(-num))
    return num
def ViyogT1000(num, Temperature = 1000):
    num = num/Temperature
    sign = torch.sign(num)
    num = torch.exp(torch.absolute(num))
    num = sign / (1.0 + torch.exp(-num))
    return num
     
"""
viyog profiling 

"""
def Viyog_Seperate_profiling(model,id_loaders,device = "cuda:0",norm_val = np.inf): 
    train_loader,_ = id_loaders.values()
    with torch.no_grad():
        id_norm_score_mean = {}
        norm_scores_list = []
        total = 0
        count = 0
        for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                feats = model.Trishul_forward(inputs)
                logits = feats[-1]

                total += targets.size(0)
                feats = feats[:-1] 
                f = copy.deepcopy(feats[0])
                f=cp.asarray(f.reshape(f.shape[0], -1))
                norm_scores_list  += cp.linalg.norm(f,ord=norm_val, axis=1).tolist() 
    id_norm_score_mean = np.mean(norm_scores_list)
    return id_norm_score_mean

def Trishul_seperate(model,id_loaders,ood_loaders,adv_loaders,id_norm_score_mean,metrics_list = {},norm_val =np.inf):
    train_loader,_ = id_loaders.values()
    device = "cuda:0"
    
    start_time = time.perf_counter()
    metrics_ViyogT1 = OODMetrics()
    metrics_ViyogT1000 = OODMetrics()
    metrics_Viyog2 = OODMetrics()
    metrics_Viyog3 = OODMetrics()
    metrics_relu = OODMetrics()
    metrics_sigmoid =OODMetrics()

    with torch.no_grad():
        for item in ood_loaders:
            norm_scores_list = []
            loader = ood_loaders[item]

            for inputs,_ in loader:
                inputs = inputs.to(device)
                f  = model.Trishul_forward(inputs)[0] 
                f=cp.asarray(f.reshape(f.shape[0], -1))
                 
                norm_scores_list  = cp.linalg.norm(f,ord=norm_val, axis=1).tolist() 

                centered_norm_scores = (np.array(norm_scores_list) - id_norm_score_mean )
                ViyogT1_scores  = ViyogT1(torch.tensor(centered_norm_scores))
                ViyogT1000_scores = ViyogT1000(torch.tensor(centered_norm_scores)) 
                # Viyog2_scores = Viyog2(torch.tensor(centered_norm_scores)) 
                # Viyog3_scores = Viyog3(torch.tensor(centered_norm_scores))
                Relu_Scores = torch.relu(torch.tensor(centered_norm_scores))
                Sigmoid_scores = torch.sigmoid(torch.tensor(centered_norm_scores))

                labels = torch.full((len(inputs),),-1,dtype=torch.long)
                
                metrics_ViyogT1.update( ViyogT1_scores,labels)
                metrics_ViyogT1000.update(ViyogT1000_scores,labels)
                metrics_Viyog2.update( Viyog2_scores,labels)
                metrics_Viyog3.update(Viyog3_scores,labels)
                metrics_relu.update( Relu_Scores,labels)
                metrics_sigmoid.update( Sigmoid_scores,labels)
        
        for item in adv_loaders:

            norm_scores_list = []
            loader = adv_loaders[item]
            count = 0
            for inputs,_ in loader:
                inputs = inputs.to(device)
                f  = model.Trishul_forward(inputs)[0]
 
                f=cp.asarray(f.reshape(f.shape[0], -1))
                
                norm_scores_list  = cp.linalg.norm(f,ord=norm_val, axis=1).tolist() 

                centered_norm_scores = (np.array(norm_scores_list) - id_norm_score_mean )
                ViyogT1_scores  = ViyogT1(torch.tensor(centered_norm_scores))  
                ViyogT1000_scores = ViyogT1000(torch.tensor(centered_norm_scores))
                Viyog2_scores = Viyog2(torch.tensor(centered_norm_scores)) 
                Viyog3_scores = Viyog3(torch.tensor(centered_norm_scores))
                Relu_Scores = torch.relu(torch.tensor(centered_norm_scores))
                Sigmoid_scores = torch.sigmoid(torch.tensor(centered_norm_scores))

                labels = torch.full((len(inputs),),1,dtype=torch.long)
                metrics_ViyogT1.update( ViyogT1_scores,labels)
                metrics_ViyogT1000.update( ViyogT1000_scores, labels )
                metrics_Viyog2.update( Viyog2_scores,labels)
                metrics_Viyog3.update(Viyog3_scores,labels)
                metrics_relu.update( Relu_Scores,labels)
                metrics_sigmoid.update( Sigmoid_scores,labels)

        metrics_list["ViyogT1"] = (metrics_ViyogT1.compute())
        metrics_list["ViyogT1000"] = (metrics_ViyogT1000.compute())
        metrics_list["Viyog2"] = (metrics_Viyog2.compute())
        metrics_list["Viyog3"] = (metrics_Viyog3.compute())
        metrics_list["Relu"] = (metrics_relu.compute())
        metrics_list["sigmoid"] = (metrics_sigmoid.compute())
    total_time = time.perf_counter() - start_time



def Trishul(model_name,id_name,seed= 99):
    device = "cuda:0"
    batch_size = 512
    model = get_model(model_name,id_name).to(device)
    id_loaders = get_id_dataloaders(id_name,batch_size)
    adv_loaders= get_ADV_DataLoaders(model_name,id_name,seed,batch_size)
    ood_loaders, _ = get_OOD_Dataloaders(model_name,id_name,seed,batch_size)
    non_id_loaders = adv_loaders | ood_loaders
    size_batch_norm = 512
    if "resnet" in model_name:
        if "resnet50" in model_name:
            size_batch_norm = 2048
         
        layers = [
                    model.conv1,
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    model.layer4,
                    nn.BatchNorm2d(size_batch_norm),
                    nn.ReLU(inplace=True)
                ]
    elif "densenet" in model_name:
        layers = [  
                    model.conv1,
                    model.block1,
                    model.trans1,
                    model.block2,
                    model.trans2,
                    model.block3,
                    nn.Sequential(model.bn1, model.relu),
                  ]
    layers = [layer.to(device) for layer in layers]
    Trishul_S1_detectors = {
        "MultiMahalanobis": MultiMahalanobis(layers),
        "ODIN": ODIN(model),
        "Energy_Score": EnergyBased(model)
    }

    if "res" in model_name:
        model_head = model.linear
    elif "dens" in model_name:
        model_head = model.fc
    
    # Detectors_Baselines = {
    #     "Mahalanobis" : Mahalanobis(model.features),
    #      "MSP" : MaxSoftmax(model),
    #      "KNN" : KNN(model),
    #      "OpenMax" : OpenMax(model, tailsize=25, alpha=5, euclid_weight=0.5),
    #      "MCD" : MCD(model, mode="mean"),
    #      "ODIN" : ODIN(model),
    #      "EnergyBased" : EnergyBased(model),
    #      "Entropy" : Entropy(model),
    #      "MaxLogit" : MaxLogit(model),
    #      "KLMatching" : KLMatching(model),
    #      "React" : ReAct( backbone = model.features, head = model_head, detector = EnergyBased.score )
    #     }
    
    train_loader,_ = id_loaders.values()

    for detector in Trishul_S1_detectors:
        Trishul_S1_detectors[detector] = Trishul_S1_detectors[detector].fit(train_loader,device=device)
    # for detector in Detectors_Baselines:
    #     Detectors_Baselines[detector] = Detectors_Baselines[detector].fit(train_loader,device=device)

    metrics_detect = {}
    metrics_seperate = {}
 
    profiling_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    odin_stats, maha_stats, energy_stats = Trishul_Profiling(Trishul_S1_detectors,id_loaders)
    profiling_peak_mem = torch.cuda.max_memory_allocated(device)
    # # print(f' odin stats = {odin_stats} , maha stats = {maha_stats} , energy_stats = {energy_stats} ')
    profiling_time = time.perf_counter() - profiling_start

    # #Stage 1 
    detection_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    Trishul_Detect(Trishul_S1_detectors,odin_stats, maha_stats, energy_stats ,id_loaders,non_id_loaders,metrics_detect )
    detection_peak_mem = torch.cuda.max_memory_allocated(device)
    detection_time = time.perf_counter() - detection_start
    
    #Stage 2
    # profiling
    stage2_profiling_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    id_norm_score_mean = Viyog_Seperate_profiling(model,id_loaders)
    stage2_profiling_time = time.perf_counter() - stage2_profiling_start
    stage2_profiling_peak_mem = torch.cuda.max_memory_allocated(device)


    seperation_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    Trishul_seperate(model,id_loaders,ood_loaders,adv_loaders,id_norm_score_mean,metrics_seperate)
    seperation_peak_mem = torch.cuda.max_memory_allocated(device)
    seperation_time = time.perf_counter() - seperation_start

    # print( f" Profiling time Stage 1 = {profiling_time} , Detection_time = {detection_time} ")
    # print( f" Profiling time Stage 2 ={stage2_profiling_time}, Seperation_time = {seperation_time}") 

    print(f" Model Name | ID Name | Seed | Metric |  'AUROC'  | 'AUTC' | 'AUPR-IN' | 'AUPR-OUT | 'FPR95TPR' | Profiling_time | Detection Time | Profiling Peak Memory | Detection Peak Memory  ")
    print("------Detection metrics-----------")
    
    for metric in metrics_detect :
        print(f" {model_name} | {id_name} | {seed} | {metric} | {metrics_detect[metric]['AUROC']} | {metrics_detect[metric]["AUTC"]} | {metrics_detect[metric]['AUPR-IN']} | {metrics_detect[metric]['AUPR-OUT']} |{metrics_detect[metric]["FPR95TPR"]} | {profiling_time} | {detection_time} |  {profiling_peak_mem} | {detection_peak_mem} ")
        # print(metrics_seperate[metric])
    print("------Seperation metrics-----------")
    print(f" Model Name | ID Name | Seed | Metric |  'AUROC'  | 'AUTC' | 'AUPR-IN' | 'AUPR-OUT | 'FPR95TPR' | Profiling_time | Seperation_time | Profiling Peak Memory | Seperation Peak Memory")
    
    for metric in metrics_seperate :
        print(f" {model_name} | {id_name} | {seed} | {metric} | {metrics_seperate[metric]['AUROC']} | {metrics_seperate[metric]["AUTC"]} | {metrics_seperate[metric]['AUPR-IN']} | {metrics_seperate[metric]['AUPR-OUT']} |{metrics_seperate[metric]["FPR95TPR"]} | {stage2_profiling_time} |  {seperation_time} | {stage2_profiling_peak_mem} | {seperation_peak_mem}")
        # print(metrics_seperate[metric])
    return metrics_detect, metrics_seperate

def run_baseline_detect(detector_name,detector,id_loaders,non_id_loaders,metrics_list ={}):
    device = "cuda:1"
    train_loader, test_loader = id_loaders.values()
    detector.fit(train_loader,device = device)
    metrics = OODMetrics()
    for loader in non_id_loaders.keys():
        non_id_loader = non_id_loaders[loader] 
        for x,_ in non_id_loader:
            x.requires_grad = True
            labels = torch.full((x.size(0),),-1,dtype=torch.long) 
            metrics.update(detector(x.to(device)).squeeze(),labels) 
    for x,label in test_loader:
        x.requires_grad = True
        metrics.update(detector(x.to(device)).squeeze(),label)
    metrics_list[detector_name] = (metrics.compute())
    return metrics_list