import torch
import torchvision
from torch.autograd import Variable
from denormalization import denormalization
from discogan_network import Generator, Discriminator
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.utils import save_image
import numpy as np
import os
import time
from itertools import chain
from write_out_result import *

# parameters
image_size = 64
batch_size = 128
num_epochs = 200

generator_enc_num_filters = [64, 128, 256, 512]
generator_dec_num_filters = [512, 256, 128, 64]
discriminator_num_filters = [64, 128, 256, 512]

generator_enc_dim = 100

generator_input_dim = 3
generator_output_dim = 3
discriminator_input_dim = 3
discriminator_output_dim = 1
learning_rate = 2e-4
generator_beta_1, generator_beta_2 = 0.5, 0.999
discriminator_beta_1, discriminator_beta_2 = 0.5, 0.999

num_iterations_decay_gan_loss = 10000
rate_start = 0.01
rate_changed = 0.5
n_gen = 5
use_adam = False
cuda_ = True if torch.cuda.is_available() else False
data_dir = '//'
output_dir = '/'
sample_interval = 100

def get_gan_loss(input_, target, gan_criterion, cuda_):
    if target is True:
        tmp_tensor = torch.FloatTensor(input_.size()).fill_(1.0)
        labels = Variable(tmp_tensor, requires_grad=False)
    else:
        tmp_tensor = torch.FloatTensor(input_.size()).fill_(0.0)
        labels = Variable(tmp_tensor, requires_grad=False)
    if cuda_:
        labels = labels.cuda()
    gan_loss = gan_criterion(input_, labels)
    return gan_loss


def get_feature_match_loss(real_features, fake_features, feature_match_criterion, cuda_):
    feature_match_losses = 0
    for real_feature, fake_feature in zip(real_features, fake_features):
        l2 = (real_feature.mean(0) - fake_feature.mean(0)) * (real_feature.mean(0) - fake_feature.mean(0))
        ones_ = Variable(torch.ones(l2.size()))
        if cuda_:
            ones_ = ones_.cuda()
        feature_match_loss = feature_match_criterion(l2, ones_)
        feature_match_losses += feature_match_loss
    return feature_match_losses



# dataset)
transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

celebA_data = dsets.ImageFolder(root=data_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=celebA_data,
                                          batch_size=batch_size,
                                          shuffle=True)

# Models
generator_AtoB = Generator(input_dim=generator_input_dim, output_dim=generator_output_dim,
                           enc_dim=generator_enc_dim, num_filters_enc=generator_enc_num_filters,
                           num_filters_dec=generator_dec_num_filters)
generator_BtoA = Generator(input_dim=generator_input_dim, output_dim=generator_output_dim,
                           enc_dim=generator_enc_dim, num_filters_enc=generator_enc_num_filters,
                           num_filters_dec=generator_dec_num_filters)
discriminator_A = Discriminator(input_dim=discriminator_input_dim, output_dim=discriminator_output_dim,
                                num_filters=discriminator_num_filters)
discriminator_B = Discriminator(input_dim=discriminator_input_dim, output_dim=discriminator_output_dim,
                                num_filters=discriminator_num_filters)
generator_AtoB.weights_init(mean=0.0, std=0.02)
generator_BtoA.weights_init(mean=0.0, std=0.02)
discriminator_A.weights_init(mean=0.0, std=0.02)
discriminator_B.weights_init(mean=0.0, std=0.02)
if cuda_:
    generator_AtoB = generator_AtoB.cuda()
    generator_BtoA = generator_BtoA.cuda()
    discriminator_A = discriminator_A.cuda()
    discriminator_B = discriminator_B.cuda()
Tensor = torch.cuda.FloatTensor if cuda_ else torch.FloatTensor

# optimizers
generator_params = chain(generator_AtoB.parameters(), generator_BtoA.parameters())
discriminator_params = chain(discriminator_A.parameters(), discriminator_B.parameters())
if use_adam:
    generator_optimizer = torch.optim.Adam(params=generator_params, lr=learning_rate,
                                           betas=(generator_beta_1, generator_beta_2),
                                           weight_decay=1e-5)
    discriminator_optimizer = torch.optim.Adam(params=discriminator_params, lr=learning_rate,
                                               betas=(discriminator_beta_1, discriminator_beta_2),
                                               weight_decay=1e-5)
else:
    generator_optimizer = torch.optim.RMSprop(params=generator_params, lr=learning_rate)
    discriminator_optimizer = torch.optim.RMSprop(params=discriminator_params, lr=learning_rate)

# losses
reconstruction_loss_criterion = nn.BCELoss()
gan_loss_criterion = nn.MSELoss()
feature_match_loss_criterion = nn.HingeEmbeddingLoss()

# training process
total_step = len(data_loader)

for epoch in range(num_epochs):
    D_losses, G_losses = [], []
    num_iter = 0
    for batch_ndx, sample in enumerate(data_loader):
        num_iter += 1
        input_A = sample['A']
        input_B = sample['B']
        # random shuffle
        idx_A = np.arange(input_A.size(0))
        idx_B = np.arange(input_B.size(0))
        np.random.shuffle(idx_A)
        np.random.shuffle(idx_B)

        input_A = input_A.numpy()
        input_B = input_B.numpy()

        input_A = torch.from_numpy(input_A[idx_A])
        input_B = torch.from_numpy(input_B[idx_B])
        if cuda_:
            input_A = input_A.cuda()
            input_B = input_B.cuda()
        A = Variable(input_A)
        B = Variable(input_B)

        # forward pass
        generator_AtoB.zero_grad()
        generator_BtoA.zero_grad()
        discriminator_A.zero_grad()
        discriminator_B.zero_grad()

        AtoB = generator_AtoB(A)
        BtoA = generator_BtoA(B)

        AtoBtoA = generator_BtoA(AtoB)
        BtoAtoB = generator_AtoB(BtoA)

        A_real, A_real_features = discriminator_A(A)
        A_fake, A_fake_features = discriminator_A(BtoA)

        B_real, B_real_features = discriminator_B(B)
        B_fake, B_fake_features = discriminator_B(AtoB)

        # train discriminator
        D_A_loss = (get_gan_loss(input_=A_real, target=True, gan_criterion=gan_loss_criterion, cuda_=cuda_)
                    + get_gan_loss(input_=A_fake, target=False, gan_criterion=gan_loss_criterion, cuda_=cuda_))*0.5
        D_B_loss = (get_gan_loss(input_=B_real, target=True, gan_criterion=gan_loss_criterion, cuda_=cuda_)
                    + get_gan_loss(input_=B_fake, target=False, gan_criterion=gan_loss_criterion, cuda_=cuda_))*0.5
        D_loss = D_A_loss + D_B_loss

        # train generator
        G_A_reconstruction_loss = reconstruction_loss_criterion(AtoBtoA, A)
        G_B_reconstruction_loss = reconstruction_loss_criterion(BtoAtoB, B)

        G_A_loss = get_gan_loss(input_=A_fake, target=False, gan_criterion=gan_loss_criterion, cuda_=cuda_)
        G_B_loss = get_gan_loss(input_=B_fake, target=False, gan_criterion=gan_loss_criterion, cuda_=cuda_)

        G_A_feature_match_loss = get_feature_match_loss(real_features=A_real_features,
                                                        fake_features=A_fake_features,
                                                        feature_match_criterion=feature_match_loss_criterion,
                                                        cuda_=cuda_)
        G_B_feature_match_loss = get_feature_match_loss(real_features=B_real_features,
                                                        fake_features=B_fake_features,
                                                        feature_match_criterion=feature_match_loss_criterion,
                                                        cuda_=cuda_)

        if num_iter < num_iterations_decay_gan_loss:
            rate = rate_start
        else:
            rate = rate_changed
        G_A_loss_total = (G_A_loss*0.1 + G_A_feature_match_loss*0.9) * (1.0 - rate) \
                         + G_A_reconstruction_loss * rate
        G_B_loss_total = (G_B_loss*0.1 + G_B_feature_match_loss*0.9) * (1.0 - rate) \
                         + G_B_reconstruction_loss * rate

        G_loss = G_A_loss_total + G_B_loss

        # optimization
        if num_iter % n_gen == 0:
            D_loss.backward()
            discriminator_optimizer.step()
        G_loss.backeard()
        generator_optimizer.step()

        if num_iter % 100 == 0:
            print("epoch: %d/%d, iteration: %d/%d, batch step: %d/%d, D_loss: %f, G_loss: %f"
                  % (epoch+1, num_epochs, num_iter, len(data_loader), (batch_ndx+1) % int(len(data_loader)/batch_size),
                     int(len(data_loader)/batch_size), D_loss.item(), G_loss.item()))

        if num_iter % sample_interval == 0:
            if not os.path.exists("images/"):
                os.makedirs("images/")
            result_image_1 = torch.cat((torch.cat((A, AtoB), dim=3), AtoBtoA), dim=3)
            result_image_2 = torch.cat((torch.cat((B, BtoA), dim=3), BtoAtoB), dim=3)
            result_image = torch.cat((result_image_1, result_image_2), dim=2)
            save_image(denormalization(result_image.data),
                       os.path.join("images/", "%d_%d.png" % (epoch+1, num_iter)),
                       normalize=True)


