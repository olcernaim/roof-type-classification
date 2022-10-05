import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        # model = nn.Sequential(
        #     nn.Conv2d(1, 20, 5),
        #     nn.ReLU(),
        #     nn.Conv2d(20, 64, 5),
        #     nn.ReLU()
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),  # 128@42*42
            nn.MaxPool2d(2),  # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),  # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        # file1 = open("/content/drive/MyDrive/model.txt", "a")
        # file1.writelines("\n----------------forward_one-----------------------\n")
        x = self.conv(x)  # CNN işlemleri yapılıyor
        # The view function is meant to reshape the tensor. If there is any situation that you don't know how many
        # rows you want but are sure of the number of columns, then you can specify this with a -1. view() reshapes a
        # tensor by 'stretching' or 'squeezing' its elements into the shape you specify: You can read -1 as dynamic
        # number of parameters or "anything". Because of that there can be only one parameter -1 in view(). First of
        # all, the view () function in pytorch is used to change the shape of tensor, for example, changing tensor
        # with 2 rows and 3 columns into 1 row and 6 columns, where-1 means that the remaining dimensions will be
        # adaptively adjusted
        # file1.writelines('self.conv(x):[%s]' % x)
        # file1.writelines("\n---------------------------------------------\n")
        x = x.view(x.size()[0], -1)  #
        # torch.nn.Linear(2,1); is used as to create a single layer with 2 inputs and 1 output.
        # torch.nn.Linear(in_features, out_features, bias=True)
        # in_features – size of each input sample(i.e.size of x)
        # out_features – size of each output sample(i.e.size of y)
        # bias – If set to False, the layer will not learn an additive bias.Default: True
        # file1.writelines('x.view(x.size()[0], -1):[%s]' % x)
        # file1.writelines("\n---------------------------------------------\n")
        x = self.liner(x)
        # file1.writelines('self.liner(x):[%s]' % x)
        # file1.writelines("\n---------------------------------------------\n")
        # file1.close()
        return x

    def forward(self, x1, x2):
        # file1 = open("/content/drive/MyDrive/model.txt", "a")
        # file1.writelines("\n----------------forward-----------------------\n")
        # file1.writelines('x1:[%s]\tx2:[%s]' % (x1, x2))
        # file1.writelines("\n---------------------------------------------\n")
        # file1.close()
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        # file1 = open("/content/drive/MyDrive/model.txt", "a")
        # file1.writelines("\n----------------forward-----------------------\n")
        # file1.writelines('out1:[%s]\tout2:[%s]\tdis:[%s]\tout:[%s]' % (out1, out2, dis, out))
        # file1.writelines("\n---------------------------------------------\n")
        # file1.close()
        #  return self.sigmoid(out)
        return out


# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
