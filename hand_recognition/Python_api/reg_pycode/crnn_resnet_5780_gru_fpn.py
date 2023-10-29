import torch.nn as nn
import torch.nn.functional as F
import torch
'''
class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

'''
class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        #self.rnn_l = nn.LSTM(nIn, nHidden, bidirectional=False)
        #self.rnn_r = nn.LSTM(nIn, nHidden, bidirectional=False)
        self.rnn_l = nn.GRU(nIn, nHidden, bidirectional=False)
        self.rnn_r = nn.GRU(nIn, nHidden, bidirectional=False)

    def forward(self, input):
        input_revers = torch.flip(input,[0])
        recurrent, _ = self.rnn_l(input)
        recurrent_revers,_ = self.rnn_r(input_revers)
        recurrent_revers = torch.flip(recurrent_revers,[0])
        output = recurrent + recurrent_revers

        return output

class BuildBlock(nn.Module):
    def __init__(self, planes=512):
        super(BuildBlock, self).__init__()

        self.planes = planes
        # Top-down layers, use nn.ConvTranspose2d to replace
        # nn.Conv2d+F.upsample?
        self.toplayer1 = nn.Conv2d(
            256,#in
            256,#out
            kernel_size=1,
            stride=1,
            padding=0)  # Reduce channels
        self.toplayer2 = nn.Conv2d(
            256, 512, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1)

        self.downlayer1 = nn.Conv2d(
            256,
            256,
            kernel_size=1,
            stride=1,
            padding=0) #
        self.downlayer2 = nn.Conv2d(
            256, 512, kernel_size=3, stride=1, padding=1)
        self.downlayer3 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(
            256, 512, kernel_size=1, stride=1, padding=0)

        #down-top dateral layers
        self.datlayer1 = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding=0)
        self.datlayer2 = nn.Conv2d(
            256, 512, kernel_size=1, stride=1, padding=0)


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(
            x,
            size=(
                H,
                W),
            mode='bilinear',
            align_corners=True) + y
    #+加法与concat有区别，concat一般是在通道上累加，而+是两个相同尺寸大小的tensor直接在元素上值相加同add()函数作用相同
    def _downsample_add(self,x,y):
        _,_,H,W = y.size()
        return F.interpolate(
            x,size=(
                    H,
                    W
                    ),
            mode='bilinear',
            align_corners=True)+y

    def forward(self, c3, c4, c5):
        # Top-down
        p5 = self.toplayer1(c5)#out 256
        p4 = self._upsample_add(p5, self.latlayer1(c4))#每次将上采样与经过1*1通道同尺寸降维后的map相加融合后，需在下一步进行3*3卷积处理，以减少上采样带来的混淆现象
        p4 = self.toplayer2(p4)#out 512
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.toplayer3(p3)#out 256

        #p3，p4,p5再多增加一次down-top的操作
        n3 = self.downlayer1(p3)# out 256
        n4 = self._downsample_add(n3, self.datlayer1(p4))  # 每次将上采样与经过1*1通道同尺寸降维后的map相加融合后，需在下一步进行3*3卷积处理，以减少上采样带来的混淆现象
        n4 = self.downlayer2(n4)# out 512
        n5 = self._downsample_add(n4, self.datlayer2(p5))
        n5 = self.downlayer3(n5)
        #return p3, p4, p5
        return n5

class REG():
    def __init__(self,
                 imgH,
                 nc,
                 nclass,
                 nh,
                 zidian_file,
                 max_batchsize,
                 n_rnn=2,
                 leakyRelu=True):
        super(REG, self).__init__()
        self.CRNN = CRNN(imgH, nc, nclass, nh)
        self.f = open(zidian_file)
        self.zidian = [""]
        self.max_batchsize = max_batchsize
        for char in self.f:
            self.zidian.append(char.replace("\n", ""))

    def load_weights(self, model_file):
        self.CRNN.load_state_dict(torch.load(model_file))

class CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=True):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [5, 3, 3, 3, 3, 3, 3, 3, 3, (3,1), (3,1)]
        ps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        nm = [8, 32, 128, 256, 256, 256, 256, 256, 512, 512, 256]

        cnn = nn.Sequential()
        cnn1 = nn.Sequential()
        cnn2 = nn.Sequential()
        cnn3 = nn.Sequential()
        #self.cnn4 = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            if i<=2:
                cnn.add_module('conv{0}'.format(i),
                               nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            elif i<=7:
                cnn1.add_module('conv{0}'.format(i),
                               nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                cnn1.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn1.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            elif i==8:
                cnn2.add_module('conv{0}'.format(i),
                                nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                cnn2.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn2.add_module('relu{0}'.format(i),
                                nn.LeakyReLU(0.2, inplace=True))
            elif i==9:
                cnn2.add_module('Conv{0}'.format(i),
                               nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                cnn2.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn2.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            elif i==10:
                cnn3.add_module('Conv{0}'.format(i),
                                nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                cnn3.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn3.add_module('relu{0}'.format(i),
                                nn.LeakyReLU(0.2, inplace=True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1, True)
        convRelu(2, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(3, True)
        convRelu(4, True)
        convRelu(5, True)
        convRelu(6, True)
        convRelu(7, True)
        cnn1.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(8, True)
        convRelu(9, True)
        cnn2.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(10, True)  # 512x1x16

        self.cnn = cnn
        self.cnn1 = cnn1
        self.cnn2 = cnn2
        self.cnn3 = cnn3
        self.fpn = BuildBlock()
        self.rnn_1 = BidirectionalLSTM(nh,nh,nh)
        self.rnn_2 = BidirectionalLSTM(nh,nh,nh)
        self.fc = nn.Linear(nh,nclass)
        '''
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        '''

    def forward(self, input):
        # conv features
        #print('---forward propagation---')
        #conv = self.cnn(input)
        c2 = self.cnn(input)
        c3 = self.cnn1(c2)
        c4 = self.cnn2(c3)
        c5 = self.cnn3(c4)
        #print("c3 shape",c3.shape)
        #print("c4 shape",c4.shape)
        #print("c5 shape",c5.shape)
        conv = self.fpn(c3,c4,c5)
        #
        b, c, h, w = conv.size()
        #print("b, c, h, w",b, c, h, w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #output = F.log_softmax(self.rnn(conv), dim=2)
        rnn_1 = self.rnn_1(conv) + conv
        rnn_2 = self.rnn_2(rnn_1) + conv
        T,n,h = rnn_2.size()
        t_rec = rnn_2.view(T*b,h)
        output = self.fc(t_rec)
        output = output.view(T,b,-1)
        #print("output",output.shape)
        output = F.log_softmax(output, dim=2)
        # output = F.softmax(output, dim=2)
        output = output.permute(1 , 0 , 2)
        #print("inside",output.shape)
        return output
