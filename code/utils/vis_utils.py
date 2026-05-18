import re
import os
import joblib
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from utils import create_dir, xy_inverse_transform

def plot_reconstruction(hparam, x_gt, x_guess, x_rec, y_gt, y_guess, y_rec, epoch=None, plot_y=False):
    margin = 1.5
    if epoch is not None:
        img_dir = hparam['LOG_DIR'] + '/tmp/'
        create_dir(hparam['LOG_DIR'], 'tmp')
    else:
        img_dir = hparam['LOG_DIR'] + '/figures/'
        create_dir(hparam['LOG_DIR'], 'figures')

    if x_gt != None: x_gt, y_gt = xy_inverse_transform(hparam, x_gt.cpu().detach().numpy().flatten(), y_gt.cpu().detach().numpy().flatten())
    if x_guess != None: x_guess, y_guess = xy_inverse_transform(hparam, x_guess.cpu().detach().numpy().flatten(), y_guess.cpu().detach().numpy().flatten())
    if x_rec != None: x_rec, y_rec = xy_inverse_transform(hparam, x_rec.cpu().detach().numpy().flatten(), y_rec.cpu().detach().numpy().flatten())

    if plot_y:
        num_channels_x = x_guess.shape[0]
        num_channels_y = y_guess.shape[0]
        num_channels = num_channels_x + num_channels_y
        fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 2, 4))
    else:
        num_channels_x = x_guess.shape[0]
        fig, axes = plt.subplots(1, num_channels_x, figsize=(num_channels_x*2, 4))

    for i in range(num_channels_x):
        min_ax_x = min(x_gt[i], x_rec[i]) * (-margin)
        max_ax_x = max(x_gt[i], x_rec[i]) * margin

        axes[i].scatter(0, x_rec[i], c='green', label=f'x{i} rec')
        if x_guess.all() != None:
            axes[i].scatter(0, x_guess[i], c='cyan', label=f'x{i} guess')
        if x_gt.all() != None:
            axes[i].scatter(0, x_gt[i], c='black', label=f'x{i} gt')

        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(min_ax_x, max_ax_x)

        axes[i].grid(True)
        axes[i].set_xticklabels([])
        axes[i].legend()

    if plot_y:
        for j in range(num_channels_y):
            min_ax_y= min(y_guess[j], y_gt[j]) * (-margin)
            max_ax_y = max(y_guess[j], y_gt[j]) * margin

            axes[i + j + 1].scatter(0, y_rec[j], c='green', label=f'y{j} rec')
            if y_guess.all() != None:
                axes[i + j + 1].scatter(0, y_guess[j], c='cyan', label=f'y{j} guess')
            if y_gt.all() != None:
                axes[i + j + 1].scatter(0, y_gt[j], c='black', label=f'y{j} gt')

            axes[i + j + 1].set_xlim(-1, 1)
            axes[i + j + 1].set_ylim(min_ax_y, max_ax_y)

            axes[i + j + 1].grid(True)
            axes[i + j + 1].set_xticklabels([])
            axes[i + j + 1].legend()

    if epoch:
        fig.suptitle(f'Reconstruction of Parameters Epoch {epoch}')
    else:
        fig.suptitle('Reconstruction of Parameters')

    plt.tight_layout()
    plt.savefig(img_dir + f'rec_{epoch}.png')
    plt.close()


def make_gif(hparam):
    img_dir = hparam['LOG_DIR'] + '/tmp/'
    # sorting all images in directory for creating the gif
    def to_int(str):
        return int(str) if str.isdigit() else str

    def natural_keys(str):
        return [to_int(c) for c in re.split(r'(\d+)', str)]

    img_list = os.listdir(img_dir)
    img_list.sort(key=natural_keys)

    frames = []
    for img in img_list:
        new_frame = Image.open(os.path.join(img_dir, img))
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(hparam['LOG_DIR'] + '/reconstruction.gif', format='GIF', append_images=frames[1:], save_all=True,
                   duration=50, loop=0)

    for img in img_list:
        os.remove(os.path.join(img_dir, img))
    os.rmdir(img_dir)
    print(f'Gif was created and saved under {hparam["LOG_DIR"]}')


import torch
from scipy.ndimage import gaussian_filter

def plot_optimization_surface(hparam, x_rec, model):
    device = 'cpu'

    # Define the original grid
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)

    # Generate a grid of parameter values
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    # Calculate the loss for each combination of parameters
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_rec[0] = torch.tensor(X[i, j], dtype=torch.float32).to(device)
            x_rec[1] = torch.tensor(Y[i, j], dtype=torch.float32).to(device)
            Z[i, j] = model(x_rec).item()

    # Apply Gaussian filter to smooth the surface
    Z_smooth = gaussian_filter(Z, sigma=2)  # Adjust sigma for more or less smoothing

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, -Z_smooth, cmap='viridis')

    # Label the axes
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Loss')

    ax.set_title('Optimization Surface')

    img_dir = hparam['LOG_DIR'] + '/figures/'
    create_dir(hparam['LOG_DIR'], 'figures')
    plt.savefig(img_dir + 'optimization_surface.png')

    plt.close()
