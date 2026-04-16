from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import *
import tools

from utils.augmentations import SSDAugmentation
from utils.customapi_evaluator import CustomAPIEvaluator
from utils.map_evaluator import MAPEvaluator

# cards_id = [0, 1, 2, 3]

cards_id = [0]

os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(f"{id}" for id in cards_id)


class MyDataParallel(torch.nn.DataParallel):
    pass

def train_object_detection(project, path_to_save, project_dir,q,
        high_resolution=True, 
        multi_scale=True, 
        cuda=True, 
        learning_rate=1e-4, 
        batch_size=32, 
        start_epoch=0, 
        epoch=100,
        train_split=80, 
        model_type='slim_yolo_v2', 
        model_weight=None,
        validate_matrix='val_acc',
        save_method='best',
        step_lr=(150, 200),
        labels=None,
        momentum=0.9,
        weight_decay=5e-4,
        warm_up_epoch=6
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

    # multi-scale (V831 AWNN supports max 224x224)
    if multi_scale:
        print('use the multi-scale trick ...')
        train_size = [320, 320]
        val_size = [224, 224]
    else:
        train_size = [224, 224]
        val_size = [224, 224]

    cfg = train_cfg
    # dataset and evaluator
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "Loading the dataset..."})

    data_dir = os.path.join(project_dir, "dataset")
    num_classes = len(labels)

    print('Project label:', labels)
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "Project label: " + str(labels)})
    print('The number of classes:', num_classes)
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "The number of classes: " + str(num_classes)})
    

    dataset = CustomDetection(root=data_dir, 
                            labels=labels,
                            transform=SSDAugmentation(train_size, mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0))
    )

    evaluator = MAPEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=labels
                                )                                

    print('The dataset size:', len(dataset))
    q.announce({"time":time.time(), "event": "dataset_loading", "msg" : "The dataset size: " + str(len(dataset))})
    print("----------------------------------------------------------")

    # dataloader

    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=8,
                    pin_memory=True
                )

    if model_type == 'slim_yolo_v2':
        from models.slim_yolo_v2 import SlimYOLOv2
        # TODO: generate anchor size <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        anchor_size = ANCHOR_SIZE
    
        yolo_net = SlimYOLOv2(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=high_resolution)
        print('Let us train slim_yolo_v2 on the')        
    else:
        print('Unknown model name...')
        return False

    model = yolo_net
    model.to(device).train()
    
    # keep training
    # if args.resume is not None:
    #     print('keep training model: %s' % (args.resume))
    #     model.load_state_dict(torch.load(args.resume, map_location=device))

    # optimizer setup
    base_lr = learning_rate
    tmp_lr = base_lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr,  momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    max_epoch = epoch
    epoch_size = len(dataset) // batch_size
    best_map = 0.0
    # start training loop
    t0 = time.time()
    

    q.announce({"time":time.time(), "event": "train_start", "msg" : "Start training ..."})

    for epoch in range(start_epoch, max_epoch):
        epoch_train_loss = 0.0
        print('Training at epoch %d/%d' % (epoch + 1, max_epoch))
        # use step lr
        q.announce({
            "time":time.time(), 
            "event": "epoch_start", 
            "msg" : "Start epoch " + str(epoch + 1) + "/" + str(max_epoch) + " ... training", 
            "epoch": epoch + 1, 
            "max_epoch": max_epoch
        })
        if epoch in step_lr:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    

        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            q.announce({
                "time":time.time(), 
                "event": "batch_start", 
                "msg" : "Start batch " + str(iter_i) + "/" + str(epoch_size) + " ... training",
                "batch": iter_i,
                "max_batch": epoch_size
            })
            if epoch < warm_up_epoch:
                tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (warm_up_epoch*epoch_size), 4)
                # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                set_lr(optimizer, tmp_lr)

            elif epoch == warm_up_epoch and iter_i == 0:
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)
        
            # to device
            images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)
            if multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            
            # make labels
            targets = [label.tolist() for label in targets] 
            
            if model_type == 'slim_yolo_v2':
                targets = tools.gt_creator(input_size=train_size, 
                                                stride=yolo_net.stride, 
                                                label_lists=targets, 
                                                anchor_size=anchor_size
                                           )

            targets = torch.tensor(targets).float().to(device)
            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)
            epoch_train_loss += total_loss.item()
            # backprop
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # display
            if iter_i % 10 == 0:                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0], t1-t0),
                        flush=True)

                t0 = time.time()

            q.announce({
                "time":time.time(), 
                "event": "batch_end", 
                "msg" : "End batch " + str(iter_i) + "/" + str(epoch_size) + " ... training",
                "batch": iter_i,
                "max_batch": epoch_size,
                "matric": {
                    "train_loss": total_loss.item()
                }
            })
        # evaluation
        eval_epoch = 1
        # if save_method == 'best':
        #     eval_epoch = 1
        # elif save_method == 'best_one_of_third':
        #     eval_epoch = 10 if epoch < max_epoch / 3 else 1
        # elif save_method == 'best_one_of_half':
        #     eval_epoch = 10 if epoch < max_epoch / 2 else 1

        if (epoch + 1) % eval_epoch == 0:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # save model strategy
            if save_method == 'best':
                if evaluator.map > best_map:
                    best_map = evaluator.map
                    print('Saving state, epoch:', epoch + 1, ' mAP:', best_map)
                    torch.save(model.state_dict(), os.path.join(path_to_save,'best_map.pth'))                            
            elif save_method == 'best_one_of_third':
                if (epoch + 1) > max_epoch / 3:
                    if evaluator.map > best_map:
                        best_map = evaluator.map
                        print('Saving state, epoch:', epoch + 1, ' mAP:', best_map)
                        torch.save(model.state_dict(), os.path.join(path_to_save,'best_map.pth'))
            elif save_method == 'best_one_of_half':
                if (epoch + 1) > max_epoch / 2:
                    if evaluator.map > best_map:
                        best_map = evaluator.map
                        print('Saving state, epoch:', epoch + 1, ' mAP:', best_map)
                        torch.save(model.state_dict(), os.path.join(path_to_save,'best_map.pth'))            

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train()

        if save_method == 'last' and (epoch + 1) == max_epoch:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 'best_map.pth'))        

        # publish the result
        q.announce({
            "time":time.time(), 
            "event": "epoch_end", 
            "msg" : "End epoch " + str(epoch + 1) + "/" + str(max_epoch) + " ... training",
            "epoch": epoch + 1,
            "max_epoch": max_epoch,
            "matric": {                
                "train_loss": epoch_train_loss / epoch_size,
                "val_acc": evaluator.map or 0.0,                
                "val_precision": evaluator.precision or 0.0,
                "val_recall": evaluator.recall or 0.0,            
            }
        })
    
    q.announce({"time":time.time(), "event": "train_end", "msg" : "Training is done"})
    print('Training is done')
    return True

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    args = parse_args()
    train(args)
