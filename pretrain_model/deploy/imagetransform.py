from lib import *



class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
        'train' : transforms.Compose([                              # crop : cat anh
            transforms.RandomResizedCrop(resize, scale = (0.5,1.0)), # resize ve size minh muon, scale: thu nho =1/2 hoac ko
            transforms.RandomHorizontalFlip(),# xoay anh theo chieu ngang
            transforms.ToTensor(), # chuyen ve dang tensor
            transforms.Normalize(mean,std) 
        ]),
        'val' : transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
            'test' : transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])     
        }
        
    def __call__(self, img, phase = 'train'):
        return self.data_transform[phase](img)
    
