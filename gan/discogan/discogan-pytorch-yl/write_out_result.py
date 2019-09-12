import matplotlib.pyplot as plt
from torch.autograd import Variable
from denormalization import denormalization
import pdb
import os
import numpy as np


# write results
def write_out_result(generator, noise, epoch_nums, iteration_nums,
                     save=False, save_dir='celebA_wgan_results/', show=False, fig_size=(5, 5)):
    generator.eval()
    noise = Variable(noise.cuda(), volatile=True)
    # pdb.set_trace()
    gen_img = denormalization(generator(noise))
    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = (noise.size()[0]//n_rows).astype(np.int32)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_img):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        # intensity rescale
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    title = 'Epoch {0}'.format(epoch_nums + 1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = save_dir + 'celebA_wgan_epoch_{:d}_iteration_{:d}'.format((epoch_nums+1),
                                                                                  (iteration_nums+1)) + '.png'
        plt.savefig(save_filename)

    if show:
        plt.show()
    else:
        plt.close()
