from lenet5 import Model
import numpy as np
import torch
from torchvision.datasets import FashionMNIST
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import visdom

import os

viz = visdom.Visdom()
train_dataset = FashionMNIST(root='./train', train=True, transform=ToTensor(), download=True)
test_dataset = FashionMNIST(root='./test', train=False, transform=ToTensor(), download=True)
data_path = "../Data/faces"


def Load_data(path):



def train(epoch,batch_size,learning_rate):
    cur_batch_win = None
    cur_batch_win_opts = {
        'title': 'Data Trace',
        'xlabel': 'Epoch Number',
        'ylabel': 'Data',
        'width': 1200,
        'height': 600,
        'legend': ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'],  # 右上角的画线
    }
    model = Model()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    for _epoch in range(epoch):
        model.train()
        print('第{}次训练开始:'.format(_epoch+1))
        for idx, (train_x, train_label) in enumerate(tqdm(train_loader,desc = '训练,当前epoch:{}'.format(_epoch))):
            label_np = np.zeros((train_label.shape[0], 10))
            optimizer.zero_grad()
            predict_y = model(train_x.float())
            loss = criterion(predict_y, train_label.long())
            loss.backward()
            optimizer.step()
        train_loss_list.append(loss.detach().cpu().item())
        print('第{}次训练结束，结果如下:'.format(_epoch + 1))
        print('epoch: {}, train_loss: {}'.format(_epoch+1, loss.sum().item()))

        for idx, (test_x, test_label) in enumerate(tqdm(test_loader,desc = '测试loss,当前epoch:{}'.format(_epoch))):
            label_np = np.zeros((test_label.shape[0], 10))
            predict_y = model(test_x.float())
            loss = criterion(predict_y, test_label.long())
        test_loss_list.append(loss.detach().cpu().item())
        print('epoch: {}, test_loss: {}'.format(_epoch+1, loss.sum().item()))
        epoch_list.append(_epoch + 1)

        train_correct = 0
        train_sum = 0
        test_correct = 0
        test_sum = 0
        model.eval()
        for idx, (train_x, train_label) in enumerate(tqdm(train_loader,desc='训练集正确率判断,当前epoch:{}'.format(_epoch))):
            predict_y = model(train_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = train_label.numpy()
            _ = predict_ys == train_label
            train_correct += np.sum(_.numpy(), axis=-1)
            train_sum += _.shape[0]
        train_accuracy_list.append(train_correct / train_sum)
        print('epoch: {}, train_accuracy: {}'.format(_epoch + 1, train_correct / train_sum))

        for idx, (test_x, test_label) in enumerate(tqdm(test_loader,desc = '测试集正确率判断,当前epoch:{}'.format(_epoch))):
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            test_correct += np.sum(_.numpy(), axis=-1)
            test_sum += _.shape[0]
        test_accuracy_list.append(test_correct / test_sum)
        print('epoch: {}, test_accuracy: {}'.format(_epoch+1, test_correct / test_sum))
        if viz.check_connection():
            cur_batch_win = viz.line(Y=np.column_stack((np.array(train_loss_list), np.array(test_loss_list),
                                                        np.array(train_accuracy_list), np.array(test_accuracy_list))),
                                     X=torch.Tensor(epoch_list),
                                     win=cur_batch_win,
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        torch.save(model, 'ex2ModelWithEpoch{}Batch_size{}learningRate{}.pth'.format(epoch, batch_size,learning_rate))



def main():
    #print('epoch = {}, batch_size = {},learning_rate = {}'.format(1000,20,0.1))
    #train(epoch = 100, batch_size=20,learning_rate=0.1)
    #print('epoch = {}, batch_size = {},learning_rate = {}'.format(2000, 20, 0.1))
    #train(epoch = 200, batch_size = 20, learning_rate = 0.1)
    #print('epoch = {}, batch_size = {},learning_rate = {}'.format(5000, 20, 0.1))
    #train(epoch = 200, batch_size = 20, learning_rate = 0.01)
    #print('epoch = {}, batch_size = {},learning_rate = {}'.format(5000, 100, 0.1))
    #train(epoch = 200, batch_size= 100,learning_rate=0.01)
    #train(epoch=300,batch_size=200,learning_rate=0.01)
    train(epoch=300, batch_size=200, learning_rate=0.001)


main()