from lib import *
from imagetransform import ImageTransform
from config import *
from utils import *
from dataset import *

def main():

    train_list = make_datapath_list('train')
    val_list = make_datapath_list('val')

    train_dataset = Mydataset(train_list, transform  = ImageTransform(resize, mean, std), phase = 'train')
    val_dataset = Mydataset(val_list, transform  = ImageTransform(resize, mean, std), phase = 'val')

    batch_size = 2
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = False)

    dataloader_dict = {'train' :train_dataloader , 'val' : val_dataloader}

    net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    net.classifier[6] = nn.Linear(in_features = 4096, out_features= 2)
    print(net)

    criterior = nn.CrossEntropyLoss()

    params1, params2, params3 = params_to_update(net)
    optimizer = optim.Adam([
        {'params':params1, 'lr' : 1e-4 },
        {'params':params2, 'lr' : 5e-4 },
        {'params':params3, 'lr' : 1e-3 },
        ])


    train_model(net, dataloader_dict, criterior, optimizer, num_epochs)

if __name__ == "__main__":
    #main()
    net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    net.classifier[6] = nn.Linear(in_features = 4096, out_features= 2)
    load_model(save_path)