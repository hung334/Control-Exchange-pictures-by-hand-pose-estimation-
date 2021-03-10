import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
import numpy as np
import matplotlib.pyplot as plt
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size = 3, padding = 1),
                nn.ReLU(),#激活涵式
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(6, 16, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 24, kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # nn.Conv2d(24, 32, kernel_size = 3, padding = 1),
                # nn.ReLU(),
                # nn.MaxPool2d(2),
                )
        self.fc = nn.Sequential(
                nn.Linear(24*16*16,10),#torch.nn.Dropout(0.5),
                #nn.ReLU(),
                #nn.Linear(120,40),
                #nn.ReLU(),
                #nn.Linear(40,10),
                #nn.ReLU(),
                )
        

    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)#拉直步驟
        output = self.fc(x)
        return output



def acc_plt_show(num_epochs,training_accuracy,validation_accuracy,LR,save_file):
    plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
    plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
    plt.title('Train & Val accuracy,epoch:{}'.format(num_epochs))
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("{}/{}_acc.jpg".format(save_file,LR))
    plt.show()

def loss_plt_show(num_epochs,training_loss,validation_loss,LR,save_file):
    plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
    plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
    plt.title('Train & Val loss,epoch:{}'.format(num_epochs))
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    #plt.ylim(0, 1)
    plt.legend()
    plt.savefig("{}/{}_loss.jpg".format(save_file,LR))
    plt.show()
    
def plt_show(image):
    
    if(len(image.shape)==3):
        image = image[:,:,::-1]
    elif(len(image.shape)==2):
        image = image
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()
    
if __name__ == '__main__':
    
    
    print(torch.cuda.get_device_properties(0))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 20
    LR =0.0125#(0.62 / 1024 * BATCH_SIZE)#0.01
    EPOCHS = 40
    #num_class = len(class_name)
    
    best_acc=0
    
    #******************************************************************************************
    train_path="./save_image"
    train_data = datasets.ImageFolder(train_path,transform=transforms.Compose([
        transforms.Resize(size=(128,128)),#(h,w)
        #transforms.RandomCrop(size=(256,256), padding=5),
        #transforms.ColorJitter(brightness=0.2, contrast=0.5,saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        #transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        transforms.RandomRotation((-5,5)),#隨機角度旋轉
        #transforms.RandomGrayscale(p=0.4),
        transforms.ToTensor()
    ]))
    test_path="./test_save_image"
    test_data = datasets.ImageFolder(train_path,transform=transforms.Compose([
         transforms.Resize(size=(128,128)),#(h,w)
         transforms.ToTensor()
     ]))
    
    
    print(train_data.classes)#获取标签
    train_Loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    test_Loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0)
    
    net = Net().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=6e-4)
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=LR, momentum=0.9, weight_decay=6e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    
    
    net.train()
    
    training_loss,validation_loss=[],[]
    training_accuracy,validation_accuracy =[],[]
    best_acc,best_epoch=0,0
    train_best_acc=0
    
    save_file="new_save_train"
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    
    for epoch in range(EPOCHS):
        net.train()
        train_loss_reg,total_train,step_count,correct_train =0.0, 0,0,0
        #if(epoch in [50,70]):
        #    learn_rate=learn_rate/10
        #    print("****************learn_rate=",learn_rate,"*****************")
        #    optimizer = torch.optim.Adam(net.classifier.parameters(),lr=learn_rate)
        
        for step, (batch_x,label_y) in enumerate(train_Loader):
            #batch_x = torch.FloatTensor(batch_x.type(torch.FloatTensor)/255)
            #label_y = torch.LongTensor(label_y)
            #h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
            #input_shape = (-1,c,h,w)
            #train = Variable(batch_x.view(input_shape)).to(device)
            train = Variable(batch_x).to(device)
            labels = Variable(label_y).to(device)
            outputs = net(train)
            train_loss = loss_func(outputs,labels)
            
            optimizer.zero_grad()               # clear gradients for this training step
            train_loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            
            train_loss_reg +=train_loss.cpu().data
            #step_count += 1
            
            ans=torch.max(outputs,1)[1].squeeze()
            #total_train += len(labels)
            correct_train += (ans.cpu() == labels.cpu()).float().sum()
            
            print("Epoch:{}/{} Step:{}/{} Train_loss:{:1.3f} ".format(epoch+1,EPOCHS,step+1,len(train_Loader),train_loss))
            
        train_accuracy = 100 * correct_train / float(len(train_data))
        training_accuracy.append(train_accuracy)
        
        #print(step,step_count)
        avg_train_loss = train_loss_reg/len(train_Loader)
        training_loss.append(avg_train_loss)
        print("{}[Epoch:{}/{}  Avg_train_loss:{:1.3f} Acc_train:{:3.2f}%]".format(("*"*30),epoch+1,EPOCHS,avg_train_loss,train_accuracy))#loss.item()
        
        #if(train_accuracy>best_acc):
        #        best_acc = train_accuracy
        #        torch.save(net.state_dict(), '{}/best_train_save_net_{}_{}.pkl'.format(save_file,epoch,LR))
        
        with torch.no_grad():
            net.eval()
            val_loss_reg = 0.0
            total_val = 0
            step_count = 0
            correct_val = 0
            
            for step, (batch_x,label_y) in enumerate(test_Loader):
                #input_shape = (-1,1,24,24)
                #val = Variable(batch_x.view(input_shape)).to(device)
                val = Variable(batch_x).to(device)
                labels = Variable(label_y).to(device)
                outputs = net(val)
                val_loss = loss_func(outputs,labels)
    
                val_loss_reg +=val_loss
                step_count += 1
                
                ans=torch.max(outputs,1)[1].squeeze().cuda()
                total_val += len(labels)
                correct_val += (ans.cpu() == labels.cpu()).float().sum()
            
            
            val_accuracy = 100 * correct_val / float(total_val)
            validation_accuracy.append(val_accuracy)
            
            avg_val_loss = val_loss_reg/step_count
            validation_loss.append(avg_val_loss)
            print("{}[Epoch:{}/{}  Avg_val_loss:{:1.3f} Acc_val:{:3.2f}]".format(("*"*30),epoch+1,EPOCHS,avg_val_loss,val_accuracy))
        
            if(val_accuracy>=best_acc and train_accuracy>=train_best_acc):
                train_best_acc = train_accuracy
                best_acc = val_accuracy
                torch.save(net.state_dict(), '{}/best_save_net_{}_{}.pkl'.format(save_file,epoch,LR))
                
        torch.cuda.empty_cache()
        
    torch.save(net.state_dict(), '{}/save_net_{}_{}.pkl'.format(save_file,EPOCHS,LR))
    












    