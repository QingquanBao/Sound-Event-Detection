import math
import torch
import torch.nn as nn


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)
    
def convblock(in_filters, out_filters, kernel_size=9, bn=True):
    ''' Scale feature map x2 in terms of width and height
    '''
    block = [nn.Conv2d(in_filters, out_filters, (kernel_size,kernel_size), padding=kernel_size // 2)] 
    if bn:
        block.append(nn.BatchNorm2d(out_filters)) #, 0.8))
    #block.append(nn.LeakyReLU(0.1, inplace=True))
    block.append(nn.GELU())
    block.append(nn.MaxPool2d((1,2)))

    return block
    
def convNextblock(in_filters, out_filters, kernel_size=9, bn=True):
    ''' Scale feature map x2 in terms of width and height
    '''
    block = [nn.Conv2d(in_filters, out_filters, (kernel_size,kernel_size), padding=kernel_size // 2, stride=(1,2))] 
    if bn:
        block.append(nn.BatchNorm2d(out_filters)) #, 0.8))
    block.append(nn.GELU())

    return block

class Crnn(nn.Module):
    def __init__(self, num_freq, class_num):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        super().__init__()
        num_layers = math.floor(math.log2(num_freq // 8))
        last_dim = 64
        convs = [ nn.BatchNorm2d(1),
                     *convblock(1, 16),
                     *convblock(16, 32),
                     *convblock(32, last_dim)]

        for i in range(num_layers - 3):
            convs += convblock(last_dim, last_dim * 2)
            last_dim *= 2
        #convs = [ 
        #             *convNextblock(1, 16),
        #             *convNextblock(16, 32),
        #             *convNextblock(32, 64)]

        hidden_size = 512
        self.conv = nn.Sequential(*convs)
        self.gru  = nn.GRU(input_size=8*last_dim, 
                            hidden_size=hidden_size, 
                            num_layers=3, 
                            batch_first=True, 
                            bidirectional=True,
                           )

        #self.classifier = nn.Sequential(
        #                    nn.Linear(hidden_size * 2, hidden_size // 4),
        #                    nn.LeakyReLU(0.1),
        #                    nn.Linear(hidden_size // 4, class_num),
        #                    nn.Sigmoid(),
        #                     )

        self.classifier = nn.Sequential(
                            nn.Linear(hidden_size * 2,  class_num),
                            nn.Sigmoid(),
                             )
    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        features = self.conv(x.unsqueeze(1))
        latents  = self.gru(features.permute((0,2,1,3)).flatten(2))
        logits   = self.classifier(latents[0])
        return logits

        
    def forward(self, x): 
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }
