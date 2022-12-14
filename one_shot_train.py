import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
#import seaborn as sns
from model import Siamese
from mydataset import RoofTrain, RoofTest

if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    train_path = 'training'
    test_path = 'evaluation'
    train_dataset = dset.ImageFolder(root=train_path)
    test_dataset = dset.ImageFolder(root=test_path)

    way = 3
    times = 200

    dataSet = RoofTrain(train_dataset, transform=data_transforms)
    testSet = RoofTest(test_dataset, transform=transforms.ToTensor(), times=times, way=way)
    testLoader = DataLoader(testSet, batch_size=way, shuffle=False, num_workers=12)
    dataLoader = DataLoader(dataSet, batch_size=32, shuffle=False, num_workers=1)  # Her 32 resimde 1


    def show_data_batch(sample_batched):
        grid = torchvision.utils.make_grid(sample_batched)
        plt.figure()
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')
        plt.axis('off')
        plt.show()


    #  def loss_fn(label, output):
    #  return -torch.mean(label * torch.log(output) + (1.0-label) * torch.log(1.0-output))
    # loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    learning_rate = 0.007

    net = Siamese()

    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    if cuda:
        net.cuda()
    # net.load_state_dict()
    show_every = 10
    save_every = 100
    test_every = 100
    train_loss = []
    loss_val = 0
    max_iter = 5000
    losses = []
    batch_ids = []
    totalRight = 0
    totalError = 0
    for batch_id, (img1, img2, path1, path2, label) in enumerate(dataLoader, 1):
        # file1 = open("Labels.txt", "a")
        # file1.writelines("\n---------------------------------------------\n")
        # file1.writelines("\n---------OnTrainClass-----------\n")
        # file1.writelines("\n---------------------------------------------\n")
        # file1.close()
        if batch_id > max_iter:
            print('-' * 100)
            with open('/content/drive/My Drive/batch_ids.txt', 'w') as f:
              np.savetxt(f, batch_ids)
            # file3 = open("/content/drive/MyDrive/batch_ids.txt", "a")
            # file3.writelines(str(batch_ids))
            # file3.close()
            with open('/content/drive/My Drive/losses.txt', 'w') as f:
              np.savetxt(f, losses)
            # file2 = open("/content/drive/MyDrive/losses.txt", "a")
            # file2.writelines(str(losses))
            # file2.close()
            break
        batch_start = time.time()
        if cuda:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        optimizer.zero_grad()
        # 32 resim,32 Path, 32 Label geliyor
        output = net.forward(img1, img2)
        # 32 resim,32 Path, 32 Label için output hesaplanıyor
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % show_every == 0:
            # file1 = open("Labels.txt", "a")
            # file1.writelines("\n---------------------------------------------\n")
            # file1.writelines("\n---------show_every-----------\n")
            # file1.writelines("\n---------------------------------------------\n")
            # file1.close()
            print('[%d]\tloss:\t%.5f\tTook\t%.2f s' % (
                batch_id, loss_val / show_every, (time.time() - batch_start) * show_every))
            batch_ids.append(float(batch_id))
            losses.append(loss_val / show_every)
            loss_val = 0
        if batch_id % save_every == 0:
            # torch.save(net.state_dict(), 'model/model-batch-%d.pth' % (batch_id + 1,))
            torch.save(net.state_dict(), '/content/drive/MyDrive/model/model-batch-%d.pth' % (batch_id + 1,))
        if batch_id % test_every == 0:
            # file1 = open("Labels.txt", "a")
            # file1.writelines("\n---------------------------------------------\n")
            # file1.writelines("\n---------test_every-----------\n")
            # file1.writelines("\n---------------------------------------------\n")
            # file1.close()
            right, error = 0, 0
            for _, (test1, test2, path3, path4) in enumerate(testLoader, 1):
                # file1 = open("/content/drive/MyDrive/model.txt", "a")
                # file1.writelines("\n---------TEST-----------\n")
                # file1.close()
                if cuda:
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)
                # file1 = open("/content/drive/MyDrive/Output1.txt", "a")
                # file1.writelines("\n---------------------------------------------\n")
                # file1.writelines("\n---------------------------------------------\n")
                # file1.writelines('Image1:[%s]\tImage2:[%s]' % (test1, test2))
                # file1.writelines("\n---------------------------------------------\n")
                # file1.writelines("\n---------------------------------------------\n")
                # file1.close()
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else:
                    error += 1
                file1 = open("/content/drive/MyDrive/model.txt", "a")
                file1.writelines('%d;%s;%s;%d\n' % (batch_id,path3, path4, pred))
                file1.close()
                file2 = open("/content/drive/MyDrive/model1.txt", "a")
                file2.writelines('output:[%s]\tImage1:[%s]\tImage2:[%s]\tResult:%d' % (output, path3, path4, pred))
                file2.writelines("\n---------------------------------------------\n")
                file2.close()
            print('*' * 70)
            print('[%d]\tright:\t%d\terror:\t%d\tprecision:\t%f' % (
                batch_id, right, error, right * 1.0 / (right + error)))
            print('*' * 70)
            totalRight += right
            totalError += error
            print('[%d]\tTotalRight:\t%d\tTotalError:\t%d\tTotalPrecision:\t%f' % (
                batch_id, totalRight, totalError, totalRight * 1.0 / (totalRight + totalError)))
            print('*' * 70)
        train_loss.append(loss_val)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)
