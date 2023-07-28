from typing import Callable
import torch
import torch.nn as nn

class SEBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample = None, reduction=8):
        """
        in_channel - size of input channel
        out_channel - size of output channel
        stride - stride size of this block. this is applied to first convolution.
        """
        super(SEBasicBlock, self).__init__()

        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)
        # convolution path
        self.conv_path = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            self.relu,
            nn.BatchNorm2d(out_channel),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        # squeeze and excitation path
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitate= nn.Sequential(
            nn.Linear(out_channel, out_channel // reduction),
            self.relu,
            nn.Linear(out_channel // reduction, out_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        residual - same as x for short-cut connection
        out - feature map which is enhanced by SE
        se_vec - vector that contains SE info
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = self.conv_path(x)
        b, c, _, _ = out.size()
        
        se_vec = self.squeeze(out).view(b, c)
        se_vec = self.excitate(se_vec).view(b, c, 1, 1)

        out = out * se_vec

        out += residual
        out = self.relu(out)
        return out





class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_blocks, stride=1):
        super(EncoderBlock, self).__init__()

        self.downsample = None
        if stride > 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        self.layers = nn.Sequential()
        self.layers.add_module("se_block0", SEBasicBlock(in_channel=in_channel, out_channel=out_channel, stride=stride, downsample=self.downsample))
        for i in range(1, num_blocks):
            self.layers.add_module(f"se_block{i}", SEBasicBlock(in_channel=out_channel, out_channel=out_channel))
        

    def forward(self, x):
        """"""
        return self.layers(x)

        
class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_blocks, stride = 1):
        """
        in_channel - size of input channel.
        out_channel - size of output channel
        num_blocks - number of SEBasic block to use
        stride - stride size(expansion rate) for conv layer after SEBlocks. only 1 or 2
        if stride == 1: conv2d is used
        if stride == 2: convT2d is used
        """
        assert stride == 1 or stride == 2
        super(DecoderBlock, self).__init__()

        # -------------- SEBlocks -------------- #
        self.se_blocks = nn.Sequential()
        self.se_blocks.add_module("se_block0", SEBasicBlock(in_channel=in_channel, out_channel=in_channel, stride=1, downsample=None))
        for i in range(1, num_blocks):
            self.se_blocks.add_module(f"se_block{i}", SEBasicBlock(in_channel=in_channel, out_channel=in_channel))

        #self.se_path = nn.Sequential(se_blocks)

        # -------------- Conv layer -------------- #
        self.conv = None
        # first and last Decoder block use Tconv : stride 1, kernel_size 1, padding 1
        # others use Conv : stride 2, kernel_size 2 and padding 1
        if stride == 2:
            self.conv = nn.ConvTranspose2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=2, padding=0, stride = 2)
        else:
            self.conv = nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=1, stride = 1)



    def forward(self, x:torch.Tensor):
        """
        x[0] = x_d - calculated feature map from previous decoder block on path.
        x[1] = x_e - encoded data from corresponding encoder.
        both inputs must have same size
        """
        #x_d = self.se_path(x_d)
        #x = torch.cat((x_d, x_e), dim = 1)
        x_d = x[0]
        x_e = x[1]
        x_d = self.se_blocks(x_d)
        x = torch.cat((x_d, x_e), dim = 1)

        return self.conv(x)

class Hooker(nn.Module):
    """
    This is for hooking feature maps for skip connection.
    You can use this to process data in middle of layers
    """
    def __init__(self, storage:dict = None, save_id:str = None, forward_function:Callable = None):
        """_summary_

        Args:
            storage (dict): storage to save input of this module. if this is None, does not save
            save_id (str): id for saving
            forward_function (type.FunctionType): if pass a function, it will be executed when forward function is called 
        """
        super(Hooker, self).__init__()
        self.storage = storage
        self.hooking_id = save_id
        self.forward_function = forward_function
    
    def forward(self, x):
        """_summary_
        hooks the input, and execute forward_function to create result
        """
        if self.storage is not None:
            self.storage[self.hooking_id] = x
        if self.forward_function is not None:
            print(self.forward_function)
            return self.forward_function(x)
        return x

class UNet(nn.Module):
    """
    mel spec --> encoder --> code
                    |
            -->  decoder
                    |
            -->  extractor

    need to implement those skip connections(|) 
    """

    def __init__(self):
        """
        conv1 -> encoders -> decoders -> tconv -> conv2 -> extractors -> fc & pooling
        """
        super(UNet, self).__init__()


        #       storage to save output of each blocks   #
        self.storage = {
            'CV1' : None,
            'EB1' : None, 'EB2' : None, 'EB3' : None, 'EB4' : None,
            'DB1' : None, 'DB2' : None, 'DB3' : None, 'DB4' : None
        }
        
        #       conv1       #
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16 , kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Hooker(storage=self.storage, save_id='CV1', forward_function=None)
        )
        
        
        #       setting encoder blocks      #
        self.encoders = nn.Sequential(
            EncoderBlock(in_channel=16, out_channel=16, num_blocks=3, stride=1),
            Hooker(storage=self.storage, save_id='EB1', forward_function=None),
            EncoderBlock(in_channel=16, out_channel=32, num_blocks=4, stride=2),
            Hooker(storage=self.storage, save_id='EB2', forward_function=None),
            EncoderBlock(in_channel=32, out_channel=64, num_blocks=6, stride=2),
            Hooker(storage=self.storage, save_id='EB3', forward_function=None),
            EncoderBlock(in_channel=64, out_channel=128, num_blocks=3, stride=1),
            Hooker(storage=self.storage, save_id='EB4', forward_function=None),
        )
        
        #       setting decoder blocks      #
        self.decoders = nn.Sequential(
            Hooker(forward_function=(lambda x: torch.stack((x, self.storage['EB4'])))),
            DecoderBlock(in_channel=128, out_channel=64, num_blocks=3, stride=1),
            Hooker(storage=self.storage, save_id='DB1', forward_function=(lambda x: torch.stack((x, self.storage['EB3'])))),
            DecoderBlock(in_channel=64, out_channel=32, num_blocks=6, stride=2),
            Hooker(storage=self.storage, save_id='DB2', forward_function=(lambda x: torch.stack((x, self.storage['EB2'])))),
            DecoderBlock(in_channel=32, out_channel=16, num_blocks=4, stride=2),
            Hooker(storage=self.storage, save_id='DB3', forward_function=(lambda x: torch.stack((x, self.storage['EB1'])))),
            DecoderBlock(in_channel=16, out_channel=16, num_blocks=3, stride=1),
            Hooker(storage=self.storage, save_id='DB4')
        )
            
        #       setting TConv               #
        self.tconv = nn.Sequential(
            Hooker(forward_function=(lambda x: torch.cat((x, self.storage['CV1']), dim=1))),
            nn.ConvTranspose2d(in_channels=16*2, out_channels=1, kernel_size=(1,1), padding=0, stride = (1,1), bias=False)
        )


    def forward(self, s, is_test=False):
        """
        s - original speech(mel spectrogram)
        es - enhanced speech
        """

        # conv
        s = self.conv1(s)

        #Encoder Path
        s = self.encoders(s)

        #Decoder Path
        s = self.decoders(s)

        # transpose conv
        es = self.tconv(s) # enhanced speech
        
        return es

if __name__ == "__main__":
    from torchsummary import summary

    model = UNet().cuda()
    summary(model, (3, 224, 178))