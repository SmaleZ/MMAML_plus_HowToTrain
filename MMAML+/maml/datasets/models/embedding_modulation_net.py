from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np


class ConvEmbeddingModel(torch.nn.Module):
    """Implements similar taskenet as
        https://github.com/shaohua0116/MMAML-Classification/blob/master/maml/models/conv_embedding_model.py
    """

    def __init__(self, input_size, output_size, embedding_dims,hidden_size=128, num_conv=4, num_channels=32, img_size=(1, 28, 28)):
        super(ConvEmbeddingModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_dims = embedding_dims
        self.device = 'cuda'
        self.num_conv = num_conv
        self.num_channels = num_channels
        self.img_size = img_size
        self.embeddings_array = []

        conv_list = OrderedDict([])
        num_ch = [self.img_size[0]] + [self.num_channels*2**i for i in range(self.num_conv)]
        
        modelcov = []
        for i in range(self.num_conv):
            modelcov += [   torch.nn.Conv2d(num_ch[i], num_ch[i+1], (3, 3), stride=2, padding=1), 
                        torch.nn.BatchNorm2d(num_ch[i+1], momentum=0.001),
                        torch.nn.ReLU(inplace=True) ]

        self.num_layer_per_conv = len(modelcov) // self.num_conv
        self.modelcov = torch.nn.Sequential(*modelcov)

        self.rnn_input_size = modelcov[self.num_layer_per_conv*(self.num_conv-1)].out_channels
        self.modelrnn = torch.nn.GRU(self.rnn_input_size, self.hidden_size, 1, bidirectional=True)

        self.embedding_input_size = self.hidden_size*2
        self.embeddings = torch.nn.ModuleList()
        for dim in embedding_dims:
            self.embeddings.append(torch.nn.Linear(self.embedding_input_size, dim))

    def forward(self, task, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = task.x
        x = self.modelcov(x)              
        
        x = F.relu(x) #relu
        x = x.view(x.size(0), x.size(1), -1) #avgpool k*n,256,1
        x = torch.mean(x, dim=2) # 256 feature vector
        h0 = torch.zeros(2, 1, self.hidden_size, device=self.device)
        inputs = x.view(x.size(0), 1, -1)
        output, hn = self.modelrnn(inputs, h0)

        N, B, H = output.shape
        output = output.view(N, B, 2, H // 2)
        embedding_input = torch.cat([output[-1, :, 0], output[0, :, 1]], dim=1)

        out_embeddings = []
        for i, embedding in enumerate(self.embeddings):
            embedding_vec = embedding(embedding_input)
            out_embeddings.append(embedding_vec)
            
        return out_embeddings

    def to(self, device, **kwargs):
        self._device = device
        super(ConvEmbeddingModel, self).to(device, **kwargs)