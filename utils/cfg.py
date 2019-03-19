cfg = {
    # dataset
    "dataset": "voc",
    "img_size": 512,
    "num_workers": 5,
    "num_classes": 21,
    
    # model
    # "pretrained": "/home/we/.torch/models/resnet50-19c8e357.pth",
    "pretrained": "/home/lxg/.torch/models/resnet18-5c106cde.pth",
    "backbone": "resnet18",  # resnet18, resnet50
    "model": "PSPNet",  # FCN, UNet, PSPNet
    "num_loss": 2,
    "checkpoint": None,  # './checkpoint/FCN_test.pth',
    "resume": False,
    
    # train
    "gpus": '1,',
    "step_lr": [20, 25],
    "epoches": 30,
    "batch_size": 5,
    "learning_rate": 1e-9,
    "weight_decay": 5e-4,
}