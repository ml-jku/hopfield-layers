import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from math import factorial

def load_images(image_paths, n_pixel):
    """loading images from given paths

    Args:
        image_paths ([str]): list of strings indicating paths to images
        n_pixel (int): number of pixels per side for each image

    Returns:
        [PIL.Image.Image]: list of PIL.Image
    """
    orig_shape = n_pixel,n_pixel
    images = []
    for path in image_paths:
        im = Image.open(path)
        im = im.resize(orig_shape)
        images.append(im)
    return images

def convert_images(images, color_option):
    """converts images to either binary black and white or gray-scale with pixel values
    in set {-1,+1} (for bw) or in intervall [-1,+1] (for gray-scale)

    Args:
        images ([PIL.Image]]): list of images loaded from PIL 
        color_option (str): either "black-white" or "gray-scale"

    Returns:
        [np.ndarray]: list of numpy arrays
    """
    valid_options = ['black-white', 'gray-scale']
    assert color_option in valid_options, 'unkown color option %s, Please choose from %s' % (color_option, valid_options)

    images_np = []
    for im in images:
        if color_option == 'black-white':
            im_grey = np.mean(im, axis=2)
            im_np = np.asarray(im_grey)
            im_np = np.where(im_np>128, -1, +1)
        elif color_option == 'gray-scale':
            im_grey = im.convert('L')
            im_np = np.asarray(im_grey)/255 *2 - 1
        images_np.append(im_np)
    return images_np

def mask_lower_half_image(images_np, value=0):
    """masks lower half of the image with given value
    note that for binary images, the default value has to be set manually to +/-1
    for gray-scale images, 0 is ok

    Args:
        images_np ([np.ndarray]): list of converted images, each as numpy array
        value (int, optional): The value used for masked pixels.

    Returns:
        [np.ndarray]: list of masked images as numpy arrays
    """
    images_np_masked = []
    n_pixel = images_np[0].shape[0]
    for im_np in images_np:
        im_masked = im_np.copy()
        for i in range(n_pixel):
            for j in range(n_pixel):
                if i > n_pixel/2:
                    im_masked[i][j] = value
        images_np_masked.append(im_masked)
    return images_np_masked

def mask_image_random(images_np):
    """masks every pixel with 50% chance. Masking value is randomly +/-1 

    Args:
        images_np ([np.ndarray]): list of images, each as numpy array

    Returns:
        [np.ndarray]: list of masked images as numpy arrays
    """
    images_np_masked = []
    n_pixel = images_np[0].shape[0]
    for im_np in images_np:
        im_masked = im_np.copy()
        for i in range(n_pixel):
            for j in range(n_pixel):
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        im_masked[i][j] = -1
                    else:
                        im_masked[i][j] = +1
        images_np_masked.append(im_masked)
    return images_np_masked

def plot_images_loaded_converted_masked(
        images, images_np, images_masked_np, 
        n_pixel=64, color_option='black-white'):
    """create a plot for each triple of color image, converted image, masked image in 1x3 layout 

    Args:
        images ([PIL.Image.Image]): list of image loaded via PIL
        images_np ([np.ndarray]): list of converted images, each as numpy array
        images_masked_np ([np.ndarray]): list of masked images, each as numpy array
        n_pixel (int, optional): number of pixels per side of each image. Defaults to 64.
        color_option (str, optional): how images were converted. Defaults to 'black-white'.
    """
    assert color_option in ['black-white', 'gray-scale'], 'unknown color option %s' %color_option
    assert len(images) == len(images_np) == len(images_masked_np)

    if color_option == 'black-white':
        cmap='binary'
    elif color_option == 'gray-scale':
        cmap = 'gray'
    
    orig_shape = n_pixel,n_pixel
    N = np.prod(orig_shape)

    for im, im_np, im_masked in zip(images, images_np, images_masked_np):
    # for path in image_paths:
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(wspace=0.5)
        plt.subplot(1,3,1)
        plt.imshow(im, cmap=cmap)
        plt.title('Original image',fontsize=20)

        plt.subplot(1,3,2)
        plt.imshow(im_np, cmap=cmap, vmin=-1, vmax=1)

        if color_option == 'black-white':
            plt.title('Black-white image',fontsize=20)
        elif color_option == 'gray-scale':
            plt.title('Gray-scale image',fontsize=20)

        plt.subplot(1,3,3)
        plt.imshow(im_masked, cmap=cmap, vmin=-1, vmax=1)
        plt.title('Masked image',fontsize=20)
        
    return

def plot_experiment_blog_style_24(train_patterns, test, reconstructed, orig_shape, cmap='binary'):
    """create a plot with 24 input patterns in a 4x6 grid and show masked and retrieved image
    layout: 4x6 grid of input images, |, masked, ➔, retrieved

    Args:
        train_patterns ([np.ndarray]): list of converted images
        test (np.ndarray): masked image which shall be retrieved
        reconstructed (np.ndarray): retrieved image
        orig_shape ((int,int)): shape of resized input images
        cmap (str, optional): colormap, "binary" for black-white and "gray" for gray-scale. Defaults to 'binary'.
    """
    plots_per_row = 6
    plot_vec_as_img = lambda x:  plt.imshow(x.reshape(orig_shape), cmap=cmap, vmin=-1, vmax=1)

    fig = plt.figure(figsize=(2.5*10,3*4))
    gs = gs = GridSpec(4, 12, figure=fig)

    # plots for training images
    train_axes = []
    for idx, img in enumerate(train_patterns):
        rowidx, colidx = idx //plots_per_row, idx % plots_per_row
        train_axes.append(fig.add_subplot(gs[rowidx, colidx]))
        plot_vec_as_img(img)

        ax = train_axes[-1]
        plt.margins(0,0)
        ax.set_frame_on(False) # removes box around axes
        ax.set_xticks([]) # removes ticks 
        ax.set_yticks([])
        plt.xlabel('train input %d'%(idx+1))

    # special plot |
    ax_bar = fig.add_subplot(gs[:, 6])
    ax_bar.set_xlim(-1, 1)
    ax_bar.set_ylim(-1, 1)
    xmid = 0
    ax_bar.plot([xmid,xmid], [-1,1], 'k--', linewidth=5)
    plt.margins(0,0)
    ax_bar.set_frame_on(False) # removes box around axes
    ax_bar.set_axis_off()


    # plot masked
    ax_masked = fig.add_subplot(gs[1:3, 7:9])
    plot_vec_as_img(test)
    plt.margins(0,0)
    ax_masked.set_frame_on(False) # removes box around axes
    ax_masked.set_xticks([]) # removes ticks 
    ax_masked.set_yticks([])
    plt.xlabel('masked test image')

    # special plot for ➔  
    ax_arrow = fig.add_subplot(gs[:, 9])
    ax_arrow.set_xlim(-1,1)
    ax_arrow.set_ylim(-1,1)
    ax_arrow.text(-0.7,-0.1,'➔', dict(size=100))
    plt.margins(0,0)
    ax_arrow.set_frame_on(False)
    ax_arrow.set_axis_off()

    # plot retrieved
    ax_retrieved = fig.add_subplot(gs[1:3, 10:12])
    plot_vec_as_img(reconstructed)
    plt.margins(0,0)
    ax_retrieved.set_frame_on(False) # removes box around axes
    ax_retrieved.set_xticks([]) # removes ticks 
    ax_retrieved.set_yticks([])
    plt.xlabel('retrieved')
    return

def plot_experiment(train, retrieve, experiment_idx, orig_shape, cmap='binary'):
    """plot experiments with an ID in title
    layout is: retrieved, masked, input images

    Args:
        train ([np.ndarray]): input images 
        retrieve (np.ndarray): retrieved image
        experiment_idx (int): number of the experiemnt for later selection
        orig_shape ((int,int)): shape of the resized input images
        cmap (str, optional): colormap, "binary" for black-white and "gray" for gray-scale. Defaults to 'binary'.
    """
    input_len = len(train)
    n_plots = input_len + 2
    fig = plt.figure(figsize=(2.5*n_plots,5))
    fig.suptitle('Experiment %d; %d train patterns' %(experiment_idx, input_len), fontsize=40)
    plt.subplots_adjust(top=1.2)
    
    images = retrieve[::-1] + train
    plot_vec_as_img = lambda x:  plt.imshow(x.reshape(orig_shape), cmap=cmap, vmin=-1, vmax=1)
    for subplot_idx, data in zip(range(n_plots),images):
        plt.subplot(1, n_plots, subplot_idx+1)
        plot_vec_as_img(data)
        if subplot_idx >= 2:
            plt.xlabel('train input %d'%(subplot_idx-1))
        elif subplot_idx == 1:
            plt.xlabel('masked test image')
        elif subplot_idx == 0:
            plt.xlabel('retrieved')
    
    return

def plot_experiment_blog_style(train, retrieve, orig_shape, cmap='binary'):
    """plot small experiemnt for blog
    layout: train_1, ..., train_n, |, masked, ➔, retrieved


    Args:
        train ([np.ndarray]): input images 
        retrieve (np.ndarray): retrieved image
        orig_shape ((int,int)): shape of the resized input images
        cmap (str, optional): colormap, "binary" for black-white and "gray" for gray-scale. Defaults to 'binary'.
    """
    input_len = len(train)
    n_plots = input_len + 4
    fig = plt.figure(figsize=(2.5*n_plots,5))
    
    images = train + [None] + [retrieve[0], None, retrieve[1]] # none are markers 
    plot_vec_as_img = lambda x:  plt.imshow(x.reshape(orig_shape), cmap=cmap, vmin=-1, vmax=1)
    for subplot_idx, data in zip(range(n_plots),images):
        plt.subplot(1, n_plots, subplot_idx+1, aspect='equal')
        ax = plt.gca()
        # ax.set_axis_off() # removes label to, which we want to keep!
        # the following 3 lines are needed to remove white margin when exporting
        plt.margins(0,0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        ax.set_frame_on(False) # removes box around axes
        ax.set_xticks([]) # removes ticks 
        ax.set_yticks([])
        
        if data is not None:
            plot_vec_as_img(data)
            
        if subplot_idx == 0:
            bbox= ax.dataLim
            
        if subplot_idx < input_len:
            plt.xlabel('train input %d'%(subplot_idx+1))
        elif subplot_idx == input_len:
            # special plot |
            ax.set_xlim(bbox.xmin, bbox.xmax)
            ax.set_ylim(bbox.ymin, bbox.ymax)
            xmid = (bbox.xmax-bbox.xmin)/2
            ax.plot([xmid,xmid], [bbox.ymin,bbox.ymax], 'k--', linewidth=5)
        elif subplot_idx == input_len + 1:
            plt.xlabel('masked test image')
        elif subplot_idx == input_len + 2:
            # special plot for ➔  
            ax.set_xlim(bbox.xmin, bbox.xmax)
            ax.set_ylim(bbox.ymin, bbox.ymax)
            ax.text(20,25,'➔', dict(size=50))
        elif subplot_idx == input_len + 3 :
            plt.xlabel('retrieved')
            
    return

def calc_combinations(n, r):
    """ calculates number of possible combinations when taking r elements out of n elements without putting them back 

    Args:
        n (int): total number of elements to choose from
        r (int): number of elements to be taken

    Returns:
        int: number of combinations
    """
    return factorial(n) // factorial(r) // factorial(n-r) # not most efficient way, but enough for small numbers as in our case

def calc_combinations_with_replacement(n, r):
    """calculates number of possible combinations when taking r elements out of n elements with putting them back 

    Args:
        n (int): total number of elements to choose from
        r (int): number of elements to be taken

    Returns:
        int: number of combinations with replacement
    """
    return factorial(n+r-1) // (factorial(r) *factorial(n-1)) # not most efficient way, but enough for small numbers as in our case

def binomial(x, y):
    """Calculate binomial coefficient xCy = x! / (y! (x-y)!)

    Args:
        x (int): -
        y (int): -

    Returns:
        int: -
    """
    try:
        binom = factorial(x) // factorial(y) // factorial(x - y)
    except ValueError:
        binom = 0
    return binom