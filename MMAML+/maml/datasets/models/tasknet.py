from collections import OrderedDict
import torch
import torch.nn.functional as F

class TaskNet(torch.nn.Module):
    """Implements similar taskenet as
    https://github.com/shaohua0116/MMAML-Classification/blob/master/maml/models/gated_conv_net.py

    """
    def __init__(self,input_size, output_size ,num_channels=64, img_side_len=28):

        super(TaskNet, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._num_channels = num_channels
        self._kernel_size = 3
        self._stride = 2
        self._padding = 1
        self._features_size = (img_side_len // 14)**2
        
        self.modulated_conv = torch.nn.Sequential()
        #the 1st block of conv + modulated_conv module
        self.modulated_conv.add_module('conv1', torch.nn.Conv2d(self._input_size, self._num_channels, self._kernel_size, 
                                                                self._stride, self._padding))
        self.modulated_conv.add_module('bn1', torch.nn.BatchNorm2d(self._num_channels, affine=False, momentum=0.001))
        self.modulated_conv.add_module('mdconv1', torch.nn.ReLU(inplace=True))
        self.modulated_conv.add_module('relu1', torch.nn.ReLU(inplace=True))
        #the 2rd block of conv + modulated_conv module
        self.modulated_conv.add_module('conv2', torch.nn.Conv2d(self._num_channels, self._num_channels*2, self._kernel_size, 
                                                                self._stride, self._padding))
        self.modulated_conv.add_module('bn2', torch.nn.BatchNorm2d(self._num_channels*2, affine=False, momentum=0.001))
        self.modulated_conv.add_module('mdconv2', torch.nn.ReLU(inplace=True))
        self.modulated_conv.add_module('relu2', torch.nn.ReLU(inplace=True))
        #the 3th block of conv + modulated_conv module
        self.modulated_conv.add_module('conv3', torch.nn.Conv2d(self._num_channels*2, self._num_channels*4, self._kernel_size, 
                                                                self._stride, self._padding))
        self.modulated_conv.add_module('bn3', torch.nn.BatchNorm2d(self._num_channels*4, affine=False, momentum=0.001))
        self.modulated_conv.add_module('mdconv3', torch.nn.ReLU(inplace=True))
        self.modulated_conv.add_module('relu3', torch.nn.ReLU(inplace=True))
        #the 4th block of conv + modulated_conv module
        self.modulated_conv.add_module('conv4', torch.nn.Conv2d(self._num_channels*4, self._num_channels*8, self._kernel_size, 
                                                                self._stride, self._padding))
        self.modulated_conv.add_module('bn4', torch.nn.BatchNorm2d(self._num_channels*8, affine=False, momentum=0.001))
        self.modulated_conv.add_module('mdconv4', torch.nn.ReLU(inplace=True))
        self.modulated_conv.add_module('relu4', torch.nn.ReLU(inplace=True))

        self.dense = torch.nn.Sequential()
        self.dense.add_module('fn', torch.nn.Linear(self._num_channels*8, self._output_size))

        self.apply(init_weights)
    
    @property
    def param_dict(self):
        return OrderedDict(self.named_parameters())
    
    def forward(self, task, params=None, embeddings=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        #print(params)
        if embeddings is not None:
            embeddings = {'mdconv1': embeddings[0],
                          'mdconv2': embeddings[1],
                          'mdconv3': embeddings[2],
                          'mdconv4': embeddings[3]}
        x = task.x
        #the 1st block of conv + modulated_conv module
        x = F.conv2d(x, weight=params['modulated_conv.conv1.weight'],
                        bias=params['modulated_conv.conv1.bias'],
                        stride=self._stride, padding=self._padding) #0
        x = F.batch_norm(x, weight=params.get('modulated_conv.bn1.weight',None),
                        bias=params.get('modulated_conv.bn1.bias',None),
                        running_mean=self.modulated_conv[1].running_mean,
                        running_var=self.modulated_conv[1].running_var, training=True) #1 
        #Atention based  modulation
        if embeddings is not None:
            x= x * torch.sigmoid(embeddings['mdconv1']).view(1, -1, 1, 1).expand_as(x) #2
        else:
            x = x #2
        x = F.relu(x) #3

        #the 2rd block of conv + modulated_conv module
        x = F.conv2d(x, weight=params['modulated_conv.conv2.weight'],
                        bias=params['modulated_conv.conv2.bias'],
                        stride=self._stride, padding=self._padding) #4
        x = F.batch_norm(x, weight=params.get('modulated_conv.bn2.weight',None),
                        bias=params.get('modulated_conv.bn2.bias',None),
                        running_mean=self.modulated_conv[5].running_mean,
                        running_var=self.modulated_conv[5].running_var, training=True) #5
        #Atention based  modulation    
        if embeddings is not None:
            x= x * torch.sigmoid(embeddings['mdconv2']).view(1, -1, 1, 1).expand_as(x) #6
        else:
            x = x #6
        x = F.relu(x) #7

        #the 3th block of conv + modulated_conv module
        x = F.conv2d(x, weight=params['modulated_conv.conv3.weight'],
                        bias=params['modulated_conv.conv3.bias'],
                        stride=self._stride, padding=self._padding) #8
        x = F.batch_norm(x, weight=params.get('modulated_conv.bn3.weight',None),
                        bias=params.get('modulated_conv.bn3.bias',None),
                        running_mean=self.modulated_conv[9].running_mean,
                        running_var=self.modulated_conv[9].running_var, training=True) #9    
        #Atention based  modulation
        if embeddings is not None:
            x= x * torch.sigmoid(embeddings['mdconv3']).view(1, -1, 1, 1).expand_as(x) #10
        else:
            x = x #10
        x = F.relu(x) #11

        #the 4th block of conv + modulated_conv module
        x = F.conv2d(x, weight=params['modulated_conv.conv4.weight'],
                        bias=params['modulated_conv.conv4.bias'],
                        stride=self._stride, padding=self._padding) #12
        x = F.batch_norm(x, weight=params.get('modulated_conv.bn4.weight',None),
                        bias=params.get('modulated_conv.bn4.bias',None),
                        running_mean=self.modulated_conv[13].running_mean,
                        running_var=self.modulated_conv[13].running_var, training=True) #13    
        #Atention based  modulation
        if embeddings is not None:
            x= x * torch.sigmoid(embeddings['mdconv4']).view(1, -1, 1, 1).expand_as(x) #14
        else:
            x = x #14
        x = F.relu(x) #15
    
        
        x = x.view(x.size(0), self._num_channels*8, self._features_size)
        x = torch.mean(x, dim=2)
        out = F.linear(x, weight=params['dense.fn.weight'], bias=params['dense.fn.bias'])

        return out


        
        
def init_weights(m):
    if(isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()





    
