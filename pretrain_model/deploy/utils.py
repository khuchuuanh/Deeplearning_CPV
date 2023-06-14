from lib import *
from config import *

def make_datapath_list(phase = 'train'):
    rootpath = './data/dog_cat/'
    target_path = osp.join(rootpath +phase+'/**/*.jpg')
    path_list = []
    for path in glob.glob(target_path): # glob.glob : goi tat ca cac duong dan co cung dinh dang
        path_list.append(path)
        
    return path_list  


def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            elif phase == 'val':
                net.eval()
            epoch_loss = 0.0
            epoch_correct = 0
            
            if(epoch == 0) and(phase == 'train'):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]): # thu vien tqdm cho phep minh xem tien do chay cua model
                optimizer.zero_grad()# gán parameters = 0, nếu dính thông tin đạo hàm từ các epoch trước thì việc học sẽ ko chuẩn
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1) # return maxtix(batch, class) and find max sac xuat theo hang
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()# step(): co tac dung update cho parameter cho optimizer
                    
                    epoch_loss +=  loss.item()*inputs.size(0)
                    epoch_correct += torch.sum(preds == labels.data)
                    
            epoch_loss = epoch_loss/ len(dataloader_dict[phase].dataset)
            epoch_accuracy =epoch_correct.double() / len(dataloader_dict[phase].dataset)
            print('{} Loss : {:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_accuracy))
    torch.save(net.state_dict(), save_path)
      
    



def params_to_update(net):
  params_to_update_1 = []
  params_to_update_2 = []
  params_to_update_3 = []
  update_param_name_1 = ['features.10.weight','features.10.bias']
  update_param_name_2 = ['classifier.1.weight','classifier.1.bias','classifier.4.weight','classifier.4.bias']
  update_param_name_3 = ['classifier.6.weight','classifier.6.bias']

  for name, param in net.named_parameters():
    if name in update_param_name_1:
      param.requires_grad = True
      params_to_update_1.append(param)
    elif name in update_param_name_2:
      param.requires_grad = True
      params_to_update_2.append(param)    
    elif name in update_param_name_3:
      param.requires_grad = True
      params_to_update_3.append(param) 
    else:
      param.requires_grad = False     
  
  return params_to_update_1,params_to_update_2,params_to_update_3

def load_model(net,model_path):
   load_weight =  torch.load(model_path, map_location={'cuda : 0': 'cpu'})
   net.load_state_dict(load_weight)
   return net
