import torch
from torch import nn
import torch.nn.functional as F


# Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, enc_dim, num_filters_enc, num_filters_dec):
        super(Generator, self).__init__()
        # encoder part
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=num_filters_enc[0],
                              kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_filters_enc[0], out_channels=num_filters_enc[1],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_features=num_filters_enc[1])
        self.conv3 = nn.Conv2d(in_channels=num_filters_enc[1], out_channels=num_filters_enc[2],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_features=num_filters_enc[2])
        self.conv4 = nn.Conv2d(in_channels=num_filters_enc[2], out_channels=num_filters_enc[3],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_features=num_filters_enc[3])
        self.conv5 = nn.Conv2d(in_channels=num_filters_enc[3], out_channels=enc_dim,
                               kernel_size=4, stride=1, padding=0, bias=False)
        self.conv5_bn = nn.BatchNorm2d(num_features=enc_dim)

        # decoder part
        self.deconv1 = nn.ConvTranspose2d(in_channels=enc_dim, out_channels=num_filters_dec[0],
                                          kernel_size=4, stride=1, padding=0)
        self.deconv1_bn = nn.BatchNorm2d(num_filters_dec[0])
        self.deconv2 = nn.ConvTranspose2d(in_channels=num_filters_dec[0], out_channels=num_filters_dec[1],
                                          kernel_size=4, stride=2, padding=1)
        self.deconv2_bn = nn.BatchNorm2d(num_filters_dec[1])
        self.deconv3 = nn.ConvTranspose2d(in_channels=num_filters_dec[1], out_channels=num_filters_dec[2],
                                          kernel_size=4, stride=2, padding=1)
        self.deconv3_bn = nn.BatchNorm2d(num_filters_dec[2])
        self.deconv4 = nn.ConvTranspose2d(in_channels=num_filters_dec[2], out_channels=num_filters_dec[3],
                                          kernel_size=4, stride=2, padding=1)
        self.deconv4_bn = nn.BatchNorm2d(num_filters_dec[3])
        self.deconv5 = nn.ConvTranspose2d(in_channels=num_filters_dec[3], out_channels=output_dim,
                                          kernel_size=4, stride=2, padding=1)

    # weights initialization
    def weights_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input_):
        # encoder part
        x = F.leaky_relu(self.conv1(input_), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), negative_slope=0.2, inplace=True)
        # decoder part
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, num_filters):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=num_filters[0],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_filters[1])
        self.conv3 = nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(num_filters[2])
        self.conv4 = nn.Conv2d(in_channels=num_filters[2],  out_channels=num_filters[3],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(num_filters[3])
        self.conv5 = nn.Conv2d(in_channels=num_filters[3], out_channels=output_dim,
                               kernel_size=4, stride=1, padding=0, bias=False)

    # weights initialization
    def weights_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input_):
        x = F.leaky_relu(self.conv1(input_), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), negative_slope=0.2, inplace=True)
        x = F.sigmoid(self.conv5(x))
        # x = self.conv5(x)
        return x


# weights initialization
def normal_init(mfun, mean, std):
    if isinstance(mfun, nn.ConvTranspose2d) or isinstance(mfun, nn.Conv2d):
        mfun.weight.data.normal_(mean=mean, std=std)
        mfun.bias.data.zero_()

