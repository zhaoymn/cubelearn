import numpy as np
import torch
import torch.nn as nn
from dataset import *
import torch.optim as optim
from network import *
import os

if __name__ == '__main__':  

    learning_rate = 0.0003
    LPP_lr = 0#.001

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    train_set = Gesture_Dataset(0)
    validation_set = Gesture_Dataset(1)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle = True, num_workers=4, pin_memory = True)
    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle = False, num_workers=4, pin_memory = True)


    net = RT_2DCNN()
    net = net.to(device)

    range_nn_params = list(map(id, net.range_net.range_nn.parameters()))
    #doppler_nn_params = list(map(id, net.doppler_net.doppler_nn.parameters()))
    #base_params = filter(lambda p:id(p) not in range_nn_params + doppler_nn_params, net.parameters())
    base_params = filter(lambda p:id(p) not in range_nn_params, net.parameters())
    

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': net.range_net.parameters(), 'lr': LPP_lr},
        #{'params': net.doppler_net.parameters(), 'lr': LPP_lr},
    ], lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    previous_acc = 0
    previous_loss = 1000000

    model_name = 'test.pth'

    for epoch in range(start_epoch, 30):  # loop over the dataset multiple times
        #train
        net.train()
        running_loss = 0.0
        all_sample = 0.0
        correct_sample = 0.0
        for i, (data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*len(label)

            #calculation of accuracy
            all_sample = all_sample + len(label)
            prediction = torch.argmax(output, 1)
            correct_sample += (prediction == label).sum().float().item()
            if i % 5 == 4:    # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f, accuracy: %.5f' % (epoch + 1, i + 1, running_loss / all_sample , correct_sample/all_sample))
        print('[%d, %5d] loss: %.5f, accuracy: %.5f' % (epoch + 1, i + 1, running_loss/all_sample, correct_sample/all_sample))

        #validation and save model
        net.eval()
        validation_loss = 0
        val_all_sample = 0.0
        val_correct_sample = 0.0
        with torch.no_grad():
            for i, (data, label) in enumerate(validationloader):
                data = data.to(device)
                label = label.to(device)
                output = net(data)
                loss = criterion(output, label)
                validation_loss += loss.item()*len(label)
                val_all_sample = val_all_sample + len(label)
                prediction = torch.argmax(output, 1)
                val_correct_sample += (prediction == label).sum().float().item()
        val_acc = val_correct_sample / val_all_sample
        val_loss = validation_loss / val_all_sample
        if val_acc > previous_acc or (val_acc == previous_acc and val_loss < previous_loss):
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, model_name)
            previous_acc = val_acc
            previous_loss = val_loss
            print('saved')
        print('all validation: %.5f, correct validation: %.5f' % (val_all_sample, val_correct_sample))
        print('[%d, %5d] val loss: %.5f, accuracy: %.5f' % (epoch + 1, i + 1, val_loss, val_acc))