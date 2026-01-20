import gc
import np 

def various_configs():

    models = ["resnet18","densenet3","resnet34","resnet50",] 
    ids = ["cifar10","cifar100","mnist","gtsrb","svhn"]
    def average_metric(metrics_dict,model_id , key,metric):
        avg_metric = []
        for seed in metrics_dict:
            if model_id in metrics_dict[seed] and key in metrics_dict[seed].keys():
                avg_metric.append(metrics_dict[seed][model_id][key][metric])
        return np.mean(avg_metric) if average_metric else None


    metrics_detect = {}
    metrics_seperate = {}
    seeds = [99]
    keys_detect = ["Trishul_S1_maha_energy","Trishul_S1_maha_odin_enegy"]
    keys_seperate = ["ViyogT1","ViyogT1000","Viyog2","Viyog3","Relu","sigmoid"]
    metrics = ["AUROC","AUPR-IN","AUPR-OUT","AUTC"]
    

    # plot_data = { }
    for seed in seeds:
        metrics_detect[seed] = {}
        metrics_seperate[seed] = {}
        for model_name in models:
            for id_name in ids:
                model_id = f"{model_name}_{id_name}"
            
                metrics_detect[seed][model_id] ,metrics_seperate[seed][model_id] = Trishul(model_name,id_name,seed) 
                gc.collect()
                torch.cuda.empty_cache()
                # for key in keys_detect:
                #     for metric in metrics:
                #         plot_data[model_id][metric] = average_metric(metrics_detect,model_id,key,metric)
                