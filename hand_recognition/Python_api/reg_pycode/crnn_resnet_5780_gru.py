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

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            if i != 9 and i != 10:
                cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            else:
                cnn.add_module('Conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

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
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(8, True)
        convRelu(9, True)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(10, True)  # 512x1x16

        self.cnn = cnn
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
        conv = self.cnn(input)
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
        output = output.permute(1 , 0 , 2)
        #print("inside",output.shape)
        return output
