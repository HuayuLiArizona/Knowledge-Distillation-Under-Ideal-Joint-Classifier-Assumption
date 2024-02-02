import os 
import time
from tqdm.auto import tqdm
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import imagenet_model_dict
from dataset.imagenet import get_imagenet_dataloaders

import utils

num_classes = 1000
print_freq = 100

exp_name = 'r50_mv1'

path = "config/imagenet/"+exp_name+".yaml"

with open(path, "r") as f:
    config = yaml.safe_load(f) 

epochs = config["SOLVER"]["EPOCHS"]
batch_size =  config["SOLVER"]["BATCH_SIZE"]

teacher_model_name = config["DISTILLER"]["TEACHER"]
student_model_name = config["DISTILLER"]["STUDENT"]

dim_student = config["DISTILLER"]["DIM_S"]
dim_teacher = config["DISTILLER"]["DIM_T"]

ckpt = './weights/student/'+student_model_name+'.pth'

class transfer_conv_plus_pro(nn.Module):
    def __init__(self, in_feature, out_feature, factor=2):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Connectors = nn.Sequential(
            nn.Conv2d(in_feature, out_feature//factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feature//factor), nn.ReLU(),
            nn.Conv2d(out_feature//factor, out_feature//factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_feature//factor), nn.ReLU(),
            nn.Conv2d(out_feature//factor, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feature), nn.ReLU())
        
    def forward(self, student):
        student = self.Connectors(student)
        return student


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
    
def train(train_loader, net_s, net_t, conector, optimizer, epoch):
    batch_time = utils.AverageMeter()
    data_time  = utils.AverageMeter()
    losses     = utils.AverageMeter()
    top1       = utils.AverageMeter()
    top5       = utils.AverageMeter()
    
    net_s.train()
    conector.train()
    end = time.time()
    
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        img, target, _ = data
        img = img.cuda()
        target = target.cuda()
        
        with torch.no_grad():
            logits_teacher, feature_teacher = net_t(img)
            feature_teacher = feature_teacher['feats'][-1]
            
        _, feature_student = net_s(img)
        
        feature_student = conector(feature_student['feats'][-1])
        
        loss_fm = F.mse_loss(feature_student, feature_teacher)
        
        feature_student = F.adaptive_avg_pool2d(feature_student, (1, 1)).view(img.size(0),-1)
        
        logits_student = net_t.module.fc(feature_student)
        
        loss_sr = F.mse_loss(logits_student, logits_teacher)

        loss =  1.0*F.cross_entropy(logits_student, target) + 1.0*loss_sr + 1.0*loss_fm
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        acc1, acc5 = utils.accuracy(logits_student, target, topk=(1, 5))
        
        losses.update(loss.detach().item(), img.size(0))
        top1.update(acc1[0].item(), img.size(0))
        top5.update(acc5[0].item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.sum:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Top-5 {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    
    return top1.avg, top5.avg

def validate(val_loader, net_s, net_t, conector, criterion):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    # switch to evaluate mode
    net_s.eval()
    conector.eval()

    end = time.time()
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            img, target = data
            img = img.cuda()
            target = target.cuda()

            # compute output
            _, feature_student = net_s(img)
            feature_student = conector(feature_student['feats'][-1])
            feature_student = F.adaptive_avg_pool2d(feature_student, (1, 1)).view(img.size(0),-1)
            
            logits_student = net_t.module.fc(feature_student)
            
            loss = criterion(logits_student, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(logits_student, target, topk=(1, 5))
            
            losses.update(loss.detach().item(), img.size(0))
            top1.update(acc1[0].item(), img.size(0))
            top5.update(acc5[0].item(), img.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('Testing: Top-1 {top1.avg:.3f}%  || Top-5 {top5.avg:.3f}%  || Loss ({loss.avg:.4f})'.format(loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(1234)
    train_loader, test_loader, num_data = get_imagenet_dataloaders(batch_size, batch_size, 8)
    
    net_t = imagenet_model_dict[teacher_model_name](pretrained=True)
    net_s = imagenet_model_dict[student_model_name](pretrained=False)
    
    net_t = nn.DataParallel(net_t).cuda()
    net_t.eval()
    
    trainable_list = nn.ModuleList([])
    
    net_s = nn.DataParallel(net_s).cuda()
    
    trainable_list.append(net_s)
    
    conector = nn.DataParallel(transfer_conv_plus_pro(dim_student, dim_teacher)).cuda()
    
    trainable_list.append(conector)
    
    optimizer = torch.optim.SGD(trainable_list.parameters(),
                                lr = config["SOLVER"]["LR"], momentum = config["SOLVER"]["MOMENTUM"],
                                weight_decay = config["SOLVER"]["WEIGHT_DECAY"])
    
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config["SOLVER"]["LR_DECAY_STAGES"], gamma=0.1)
    
    best_top1 = 0
    
    train_top_1 = []
    train_top_5 = []
    val_top_1 = []
    val_top_5 = []
    
    start_time = time.time()
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    for epoch in range(epochs):
       train_top1,train_top5 = train(train_loader, net_s, net_t, conector, optimizer, epoch)
       
       val_top1, val_top5 = validate(test_loader, net_s, net_t, conector, criterion)
       
       if val_top1 > best_top1:
           best_top1 = val_top1
       
       lr_scheduler.step()
       
       train_top_1.append(train_top1)
       train_top_5.append(train_top5)
       val_top_1.append(val_top1)
    
    end_time = time.time()
    
    print('%d epochs training and val time : %.2f'%(epochs, end_time-start_time))
    
    val_top1, val_top5 = validate(test_loader, net_s, net_t, conector, criterion)
    print("last epoch val accuracy : %.2f"%val_top1) #report this
    print("best accuracy : %.2f"%best_top1)
    
    logs = {"train_t1":train_top1,
            "train_t5":train_top5,
            "val_t1":val_top1,
            "val_t5":val_top5
            }
    
    np.save('./logs/imagenet/'+exp_name+'.npy', logs)