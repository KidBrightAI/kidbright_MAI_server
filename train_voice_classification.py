import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from models.voice_cnn import VoiceCnn
# import resnet as models
import numpy as np
import cv2
import os
import random
from torchsummary import summary
import torch.optim as optim
import time
import torch.backends.cudnn as cudnn



def train_voice_classification(project, path_to_save, project_dir,q,
        cuda=True, 
        learning_rate=1e-4, 
        batch_size=32, 
        start_epoch=0, 
        epoch=100,
        train_split=80, 
        model_type='mobilenet-75', 
        model_weight=None,
        validate_matrix='val_acc',
        save_method='best',
        step_lr=(150, 200),
        labels=None,
        weight_decay=5e-4,
        warm_up_epoch=6,
        input_shape = (3, 147, 13)
    ):
    
    os.makedirs(path_to_save, exist_ok=True)
    
    # cuda
    if cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("use cpu")

    # dataset and evaluator
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "Loading the dataset..."})

    data_dir = os.path.join(project_dir, "dataset", "mfcc")
    num_classes = len(labels)

    print('Project label:', labels)
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "Project label: " + str(labels)})
    print('The number of classes:', num_classes)
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "The number of classes: " + str(num_classes)})

    # split the dataset to train and val
    # create train and valid folder
    if not os.path.exists(os.path.join(data_dir, 'train')):
        os.makedirs(os.path.join(data_dir, 'train'))
    if not os.path.exists(os.path.join(data_dir, 'valid')):
        os.makedirs(os.path.join(data_dir, 'valid'))

    for label in labels:
        if not os.path.exists(os.path.join(data_dir, 'train', label)):
            os.makedirs(os.path.join(data_dir, 'train', label))
        if not os.path.exists(os.path.join(data_dir, 'valid', label)):
            os.makedirs(os.path.join(data_dir, 'valid', label))

    for label in labels:
        images = os.listdir(os.path.join(data_dir, label))
        random.shuffle(images)
        train_images = images[:int(len(images) * train_split / 100)]
        val_images = images[int(len(images) * train_split / 100):]

        for image in train_images:
            os.rename(os.path.join(data_dir, label, image), os.path.join(data_dir, 'train', label, image))
        for image in val_images:
            os.rename(os.path.join(data_dir, label, image), os.path.join(data_dir, 'valid', label, image))

    #remove empty folder
    for label in labels:
        if len(os.listdir(os.path.join(data_dir, label))) == 0:
            os.rmdir(os.path.join(data_dir, label))
            
    train_transforms = transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),
        #transforms.Resize(input_shape[1:]),
        #transforms.Resize(input_shape),
        #transforms.Resize(255),
        #transforms.RandomRotation(30),
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),
        #transforms.Resize(input_shape[1:]),
        #transforms.Resize(input_shape),
        #transforms.Resize(255),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    valset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    
    # model
    print("----------------------------------------------------------")
    print('Building the model...')
    q.announce({"time":time.time(), "event": "model_building", "msg" : "Building the model..."})
    print('model type:', model_type)
    if model_type == 'resnet18':
        net = models.resnet18(pretrained=True)
    elif model_type == 'resnet34':
        net = models.resnet34(pretrained=True)
    elif model_type == 'resnet50':
        net = models.resnet50(pretrained=True)
    elif model_type == 'resnet101':
        net = models.resnet101(pretrained=True)
    elif model_type == 'resnet152':
        net = models.resnet152(pretrained=True)        
    elif model_type == 'code':
        net = VoiceCnn(device, project["trainConfig"]["code"], input_size=input_shape, num_classes=num_classes, trainable=True)
    else:
        print('model type error')
        return False

    # add fc layer
    if model_type.startswith('mobilenet'):
        net.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)
    elif model_type.startswith('resnet'):
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=num_classes, bias=True)
    elif model_type == 'code':
        pass
    else:
        print('model type error')
        return False

    if model_weight:
        net.load_state_dict(torch.load(model_weight))

    net.to(device)
    summary(net, input_size=input_shape)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_lr, gamma=0.1)

    # train
    print("----------------------------------------------------------")
    print('Start training...')
    q.announce({"time":time.time(), "event": "train_start", "msg" : "Start training ..."})

    best_acc = 0.0
    best_epoch = 0
    max_epoch = epoch
    for epoch in range(start_epoch, epoch):
        print('Training at epoch %d/%d' % (epoch + 1, max_epoch))
        q.announce({
            "time":time.time(), 
            "event": "epoch_start", 
            "msg" : "Start epoch " + str(epoch + 1) + "/" + str(max_epoch) + " ... training", 
            "epoch": epoch + 1, 
            "max_epoch": max_epoch
        })
        if epoch < warm_up_epoch:
            lr = learning_rate * (epoch + 1) / warm_up_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('epoch:', epoch, 'lr:', lr)
        else:
            #scheduler.step()
            print('epoch:', epoch, 'lr:', optimizer.param_groups[0]['lr'])
            
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            print('inputs.shape:', inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        print('[%d] loss: %.5f' % (epoch, running_loss / len(trainloader)))

        
        # eval
        correct = 0
        total = 0
        net.eval()

        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print('Accuracy of the network on the val images: %d %%' % acc)
        if save_method == 'best':
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                print('save model params to', os.path.join(path_to_save, 'best_acc.pth'))
                torch.save(net.state_dict(), os.path.join(path_to_save, 'best_acc.pth'))
        elif save_method == 'last':
            print('save model params to', os.path.join(path_to_save, 'best_acc.pth'))
            torch.save(net.state_dict(), os.path.join(path_to_save, 'best_acc.pth'))
        elif save_method == 'best_one_of_third':
            if (epoch + 1) > max_epoch / 3:
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    print('save model params to', os.path.join(path_to_save, 'best_acc.pth'))
                    torch.save(net.state_dict(), os.path.join(path_to_save, 'best_acc.pth'))
        elif save_method == 'best_one_of_half':
            if (epoch + 1) > max_epoch / 2:
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    print('save model params to', os.path.join(path_to_save, 'best_acc.pth'))
                    torch.save(net.state_dict(), os.path.join(path_to_save, 'best_acc.pth'))

        # report
        # publish the result
        q.announce({
            "time":time.time(), 
            "event": "epoch_end", 
            "msg" : "End epoch " + str(epoch + 1) + "/" + str(max_epoch) + " ... training",
            "epoch": epoch + 1,
            "max_epoch": max_epoch,
            "matric": {                
                "train_loss": running_loss / len(trainloader),
                "val_acc": acc,      
            }
        })
    print('Finished Training')
    q.announce({"time":time.time(), "event": "train_end", "msg" : "Training is done"})
    print('Training is done')
    return True

if __name__ == "__main__":
    from utils.message_announcer import MessageAnnouncer
    reporter = MessageAnnouncer()
    train_voice_classification(
        "test",
        "out",
        "projects/eee",
        reporter,
        cuda=True,
        learning_rate=1e-4,
        batch_size=32,
        start_epoch=0,
        epoch=20,
        train_split=80,
        model_type='resnet18',
        model_weight=None,
        validate_matrix='val_acc',
        save_method='best',
        step_lr=(150, 200),
        labels=['aaa', 'vvv'],
        weight_decay=5e-4,
        warm_up_epoch=6
    )