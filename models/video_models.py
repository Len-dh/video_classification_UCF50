import torch.nn as nn
import torchvision.models as models
import torch 
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101


class resnet3D(nn.Module):
    def __init__(self, params_model):
        super(resnet3D, self).__init__()
        pretrained = params_model["pretrained"]
        progress = params_model["progress"]

        baseModel = models.video.r3d_18(pretrained=pretrained, progress=progress)
        baseModel.classifier = nn.Sequential(*(list(baseModel.children())[:-1]))
        self.baseModel = baseModel
        self.fc = nn.Linear(200, 4)

    def forward(self,inputs):
        batch_size,sequence_lengh, Channels, H, W = inputs.shape

        r_in = inputs.reshape(Channels, sequence_lengh, H, W, batch_size)
        self.baseModel(r_in)
        r_out = self.baseModel(r_in)
        # print('r_out : ', r_out, r_out.shape)
        # print('r_out[-1, :]', r_out[-1, :])
        r_out = r_out.reshape(80, 2, 200)
        # print('r_out rehape : ', r_out.shape)
        out = self.fc(r_out[-1])
        # print('out : ', out, out.shape)
        return out


class Cnn_GRU(nn.Module):
    def __init__(self, params_model):
        super(Cnn_GRU, self).__init__()
        num_classes = params_model["num_classes"]
        device = params_model["device"]
        num_layers = params_model["num_layers"]
        hidden_size = params_model["hidden_size"]
        model_cnn = params_model["model_cnn"]
        VIDEO_CUT_LENGTH = params_model["VIDEO_CUT_LENGTH"]
        bidirectional = params_model["bidirectional"]

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.VIDEO_CUT_LENGTH = VIDEO_CUT_LENGTH
        self.bidirectional = bidirectional

        if model_cnn == 'alexnet': 
            alexnet = models.alexnet(pretrained=True)

            # Removing last layer of vgg 16
            embed = nn.Sequential(*list(alexnet.classifier.children())[:-1])
            alexnet.classifier = embed

            self.cnn = alexnet 

        if model_cnn == 'resnet18':
            resnet18 = models.resnet18(pretrained=pretrained)

            # Removing last layer of vgg 16
            embed = nn.Sequential(*list(resnet18.classifier.children())[:-1])
            resnet18.classifier = embed

            self.cnn = resnet18 

        if model_cnn == 'vgg16':
            # Loading a VGG16 
            vgg = models.vgg16(pretrained=True, progress=False)
        
            # Removing last layer of vgg 16
            embed = nn.Sequential(*list(vgg.classifier.children())[:-1])
            vgg.classifier = embed

            # # Freezing the feature layers          
            for parameter in vgg.features[:-5].parameters():
                parameter.requires_grad = False
            for param in vgg.classifier.parameters():
                param.requires_grad = False
    
            self.cnn = vgg

        if model_cnn == 'vgg19':
            # Loading a VGG16 
            vgg = models.vgg19(pretrained=True, progress=False)
        
            # Removing last layer of vgg 16
            embed = nn.Sequential(*list(vgg.classifier.children())[:-1])
            vgg.classifier = embed     

            # # Freezing the feature layers          
            for parameter in vgg.features[:-5].parameters():
                parameter.requires_grad = False
            for param in vgg.classifier.parameters():
                param.requires_grad = False

            self.cnn = vgg

        if bidirectional == False:
            self.gru = nn.GRU(input_size = 4096, hidden_size = hidden_size, num_layers = num_layers, bidirectional = False)
            self.fc = nn.Linear(hidden_size, num_classes)
        else: 
            self.gru = nn.GRU(input_size = 4096, hidden_size = hidden_size, num_layers = num_layers, bidirectional = True)
            self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, inputs):
        sequence_lengh = self.VIDEO_CUT_LENGTH
        batch_size = inputs.size(0)
        # print('batch_size : ', batch_size)
        # num_layers = 6
        Channels = 3
        # hidden_size = 128
        H = 224
        W = 224

        c_in = inputs.view(batch_size*sequence_lengh, Channels, H, W)
        # print('c_in : ', c_in.shape)
        c_out = self.cnn(c_in)
        # print('c_out : ',c_out.shape)

        if self.bidirectional == False:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        else:
            h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)

        gru_in = c_out.view(sequence_lengh, batch_size, -1)
    
        gru_out, hidden = self.gru(gru_in, h0)
        out = self.fc(gru_out[-1, :, :])     
        return out


class GRU(nn.Module):
    def __init__(self, params_model):
        super(GRU, self).__init__()
        # self.bidirectional = bidirectional
        device = params_model["device"]
        input_size = params_model["input_size"]
        num_layers = params_model["num_layers"]
        hidden_size = params_model["hidden_size"]
        num_classes = params_model["num_classes"]
        dropout = params_model["dropout"]
        VIDEO_CUT_LENGTH = params_model["VIDEO_CUT_LENGTH"]
        bidirectional = params_model["bidirectional"]

        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.VIDEO_CUT_LENGTH = VIDEO_CUT_LENGTH
        self.bidirectional = bidirectional


        # if self.bidirectional:
        #     self.bidirect = 2
        # else:
        #     self.bidirect = 1
        if self.bidirectional == False:    
            self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=False, dropout=dropout, bidirectional=False)
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=False, dropout=dropout, bidirectional=True)
            self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, inputs):
        # batch_size, timesteps, C,H, W = inputs.size()
        sequence_lengh = self.VIDEO_CUT_LENGTH
        batch_size = inputs.size(0)
        # num_layers = 2 
        #hidden = torch.zeros(timesteps, inputs.size(0), 128).to(self.device)
        if self.bidirectional == False:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        else:
            h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        # print('size of h0 : ', len(h0))
		# Pack them up nicely
        # gru_in = inputs.view(sequence_lengh*3, batch_size , 224*224)    
        gru_in = inputs.view(sequence_lengh, batch_size, -1)    
        # print('size of gru_in : ',len(gru_in))
        # print('gru_in[-1] : ', gru_in[-1])
        # print('size of gru_in[-1] : ', len(gru_in[-1]))      
        out, hidden = self.gru(gru_in, h0)
        # print('size of out : ', len(out))
        out = self.fc(out[-1, :, :])
        return out


class Cnn_lstm(nn.Module):
    def __init__(self,params_model):
        super(Cnn_lstm, self).__init__()
        num_classes = params_model["num_classes"]
        device = params_model["device"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        model_cnn = params_model["model_cnn"]
        VIDEO_CUT_LENGTH = params_model["VIDEO_CUT_LENGTH"]
        bidirectional = params_model["bidirectional"]

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.device = device
        self.VIDEO_CUT_LENGTH = VIDEO_CUT_LENGTH
        self.bidirectional = bidirectional

        if model_cnn == 'alexnet': 
            alexnet = models.alexnet(pretrained=True)

            # Removing last layer of vgg 16
            embed = nn.Sequential(*list(alexnet.classifier.children())[:-1])
            alexnet.classifier = embed

            self.cnn = alexnet 

        if model_cnn == 'resnet18':
            resnet18 = models.resnet18(pretrained=True)

            # Removing last layer of vgg 16
            embed = nn.Sequential(*list(resnet18.classifier.children())[:-1])
            resnet18.classifier = embed

            self.cnn = resnet18 

        if model_cnn == 'vgg16':
            # Loading a VGG16 
            vgg = models.vgg16(pretrained=True, progress=False)
        
            # Removing last layer of vgg 16
            embed = nn.Sequential(*list(vgg.classifier.children())[:-1])
            vgg.classifier = embed

            # # Freezing the feature layers          
            for parameter in vgg.features[:-5].parameters():
                parameter.requires_grad = False
            for param in vgg.classifier.parameters():
                param.requires_grad = False
    
            self.cnn = vgg

        if model_cnn == 'vgg19':
            # Loading a VGG16 
            vgg = models.vgg19(pretrained=True, progress=False)
        
            # Removing last layer of vgg 16
            embed = nn.Sequential(*list(vgg.classifier.children())[:-1])
            vgg.classifier = embed     

            # # Freezing the feature layers          
            for parameter in vgg.features[:-5].parameters():
                parameter.requires_grad = False
            for param in vgg.classifier.parameters():
                param.requires_grad = False
    
            self.cnn = vgg

        num_features = 4096

        # self.dropout= nn.Dropout(dr_rate)

        if bidirectional == False:
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, dropout = dr_rate, bidirectional=False)
            self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
        else: 
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, dropout = dr_rate, bidirectional=True)
            self.fc1 = nn.Linear(rnn_hidden_size*2, num_classes)

            
    def forward(self, inputs):
        sequence_lengh = self.VIDEO_CUT_LENGTH
        Channels = 3
        batch_size = inputs.size(0)
        H = 224
        W = 224

        c_in = inputs.reshape(batch_size*sequence_lengh, Channels, H, W)
        # print('c_in : ', c_in.shape)
        c_out = self.cnn(c_in)
        # print('c_out : ',c_out.shape)
        r_in =  c_out.view(sequence_lengh, batch_size, -1)
        if self.bidirectional == False:
            h_0 = torch.randn(self.rnn_num_layers,batch_size,self.rnn_hidden_size).to(self.device)
            c_0 = torch.randn(self.rnn_num_layers,batch_size,self.rnn_hidden_size).to(self.device)
        else:
            h_0 = torch.randn(self.rnn_num_layers*2,batch_size,self.rnn_hidden_size).to(self.device)
            c_0 = torch.randn(self.rnn_num_layers*2,batch_size,self.rnn_hidden_size).to(self.device)
        # print('r_in :  ', r_in.shape, r_in)
        # print('c_out.shape[-1] : ', c_out.shape[-1])
        # print('c_out.view(-1,batch_size,c_out.shape[-1]) : ', c_out.view(-1,batch_size,c_out.shape[-1]))
        # r_out, (h_n, h_c) = self.rnn(c_out.view(-1,batch_size,c_out.shape[-1]))
        r_out, (h_n, h_c) = self.rnn(r_in, (h_0, c_0))
        # r_out, _ = self.rnn(r_in)
        # print('r_out', r_out.shape)
        # print('h_n and h_c : ', h_n.shape, h_c.shape)
        # print('r_out[-1, :, :] : ', r_out[-1, :, :]) #, r_out[-1, :, :].shape)
        fc1 = self.fc1(r_out[-1, :, :])     
        # print('size of fc1 : ', len(fc1), fc1)
        return fc1