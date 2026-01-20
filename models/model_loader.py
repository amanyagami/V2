 
from models.densenet import DenseNet3
from models.mobilenetv2 import MobileNetV2
from models.resnet_cifar import ResNet34
from models.resnet_imagenet import ResNet50, ResNet101
from models.vgg16 import vgg16
from models.convnext import convnext_base
from models.densenet_imagenet import densenet121
from models.vit import vit_b_16
from models.wideresnet import WideResNet

def get_models(m_name,id_name, num_classes,device):

    if id_name in ['cifar10','svhn','mnist']:
        num_classes =10 
    elif id_name == 'cifar100':
        num_classes = 100
    elif id_name == "gtsrb":
        num_classes = 43

    if m_name == 'resnet34':
        model = ResNet34(num_classes)
    if m_name == 'resnet50':
        model = ResNet50(num_classes=num_classes)       
    if m_name == 'resnet101':
        model = ResNet101(num_classes=num_classes)      
    if m_name == 'mobilenetv2':
        model = MobileNetV2(num_classes=num_classes)        
    if m_name == 'densenet-bc':
        model = DenseNet3(100, int(num_classes))        
    if m_name == 'wideresnet40':
        model = WideResNet(depth=40, num_classes=int(num_classes), widen_factor=2)       
    if m_name == 'vgg16':
        model = vgg16(num_classes=num_classes )        
    if m_name == 'convnext':
        model = convnext_base(num_classes=num_classes )        
    if m_name == 'densenet_imagenet':
        model = densenet121(num_classes=num_classes)    
    if m_name == 'vit':
        model = vit_b_16(num_classes=num_classes)      
    else:
        exit('{} model is not supported'.format(m_name))
    model_path = f'./pretrained/{m_name}_{id_name}_SGD.pth'
    model.load_state_dict(torch.load(model_path,map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model  
