import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import data
from skimage.color import rgb2hed

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.filters import sobel 
from skimage.feature._canny import canny
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.transform import resize
from sklearn.cluster import KMeans
import pdb
# flake8: noqa

# Create an artificial color close to the original one
cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                            'saddlebrown'])
cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
                                            'white'])


def plot_images(imgs, save_title='imgs.png', show = False, labels=None, Title='Sample Images'):
    '''
    Arguments: 
        imgs: Dictionary of images, plotting options. 
        babels (optional) is a list of n titles to label the subplots
    Output: Plot of 2, 4, or 8 images.  
    '''
    if len(imgs) == 2:
        fig, axs = plt.subplots(1,2, figsize=(8, 8))
    elif len(imgs) == 4:
        fig, axs = plt.subplots(2,2, figsize=(8, 8))
    elif len(imgs) == 8:
        fig, axs = plt.subplots(4,2, figsize=(10, 10))
    else:
        raise ValueError("Plot function requires 2, 4, or 8 items. Passed in {} images".format(len(imgs)))

    if labels == None:
        labels = sorted(imgs.keys())
    else:
        labels = sorted(labels)	

    # stuff for sorting; probably there's a better way
    lst_k = []
    lst_v = []
    for i, key in enumerate (sorted(imgs)):
        lst_k.append(labels[i])
        lst_v.append(imgs[key])

    #for ax, k, v in zip(axs.flatten(), labels, sorted(imgs).values()): 
    for ax, k, v in zip(axs.flatten(), lst_k, lst_v):
        ax.grid(False)
        ax.imshow(v, cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Image ' + k)

    fig.suptitle(Title)
    fig.tight_layout()
    fig.savefig(save_title, dpi = 200)
    if show: 
        plt.show()	

def get_grayscale(imgs, show_bool = False): 
	'''
	Arguments: 
        imgs - Dictionary of images, plotting option. 
	Output: Dictionary of images, plot of 8 images. 

	Transform the colored images to greyscale, and plot them to make sure 
	it worked.
	'''
	gray_imgs = {}
	for k, v in imgs.items(): 
		gray_imgs[k] = rgb2gray(v)

	if show_bool: 
		plot_images(gray_imgs, save_title='gray_imgs.png', show=show_bool)

	return gray_imgs

def dye_color_separation (ihc_rgb, channel=None):
    '''
    Arguments:
        ihc_rgb is a color image in a numpy array, 3D
        channel (optiona) If set to "H" or "E" returns only the hetatoxylin or eosin channel as greyscale, else, channel separated 3D
    Returns:

    '''
    ihc_hed = rgb2hed(ihc_rgb)  
    return ihc_hed

def dye_color_separation_dict (img_dict):
    '''
    Arguments:
        img_dict is a dictionary of images to dye color separate 
    returns 
        img_dict of images separated into the primary dye colors for HE and DAG
    '''
    d = {}
    for k, v in img_dict.items():
        d[k] = rgb2hed(v)  
    return d    

def get_dye_separation(img_dict, color="H"):
    '''
    Arguments:
        ihc_rgb is a dictionary of color images
        color = "H" (hematoxylin) or "E" Eosin
    Returns:
        dictionary of greyscale 2D images, separated by main dye colors into H or E component
    rgb2hed is RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.    
    '''
    d = {}
    for k, v in img_dict.items():
        if color=="H":
            d[k] = rgb2hed(v)[:, :, 0]
        else:
            d[k] = rgb2hed(v)[:, :, 1]
    return d    

def get_he_separation(img_dict):
    '''
    Arguments:
        ihc_rgb is a dictionary of color images
        color = "H" (hematoxylin) or "E" Eosin
    Returns:
        dictionary of new 3D images, separated to only include H or E components
    '''
    d = {}
    for k, v in img_dict.items():

        h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
        e = rescale_intensity(ihc_hed[:, :, 1], out_range=(0, 1))
        zdh = np.dstack((np.zeros_like(h), e, h))         
        d[k] = zdh
    return d    


def plot_dye_separation(ihc_rgb, ihc_hed):
    '''
    Arguments:
        ihc_rgb is a dictionary of color images
        ihc_hed has been dye separated by the main dye colors
    Output:
        a plot of the H&E components    
    '''
    # accepts original image and the image that has been split to 3 dye types
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    #ax[0].imshow(ihc_rgb)
    #ax[0].set_title("Original image")

    ax[0].imshow(ihc_hed[:, :, 0], cmap=cmap_hema)
    ax[0].set_title("Hematoxylin")

    ax[1].imshow(ihc_hed[:, :, 1], cmap=cmap_eosin)
    ax[1].set_title("Eosin")

    #ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)
    #ax[3].set_title("DAB")

    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()
    return ax


def rescale_dye_signals (ihc_hed, ax):
    # Rescale hematoxylin and eosin signals and give them a fluorescence look
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    e = rescale_intensity(ihc_hed[:, :, 1], out_range=(0, 1))
    zdh = np.dstack((np.zeros_like(h), e, h))

    fig = plt.figure()
    axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
    axis.imshow(zdh)
    axis.set_title("Stain separated image (rescaled)")
    axis.axis('off')
    plt.show()
    return zdh 

def apply_filter(imgs, img_filter=sobel, save_title = 'a.png', show_bool=False): 
    '''
    Input: Dictionary of images, image filter, plotting options. 
    Output: Dictionary of images, plot of 8 images. 
    Apply the filter to the set of images and plot them to make sure it worked. 
    '''

    filtered_imgs = {}
    for k, v in imgs.items(): 
        filtered_imgs[k] = img_filter(v)
    if show_bool: 
        plot_images(filtered_imgs, save_title=save_title, show = show_bool)
    return filtered_imgs

def test_canny_sigma(imgs): 
    '''
    Input: Dictionary of gray scaled images. 
    Output: None

    Apply several different levels of sigma to the canny filter, plotting 
    each result. Figure out which canny works best. 
    '''
    for sig in range(5): 
        filtered_imgs = {}
        for i, (k, v) in enumerate(imgs.items()): 
            filtered_imgs[k] = canny(v, sigma = sig)
            save_tit = 'sig' + str(sig) + '.png'
            if i % 2 == 0:
                print("Sigma: {}".format(sig))
        plot_images(filtered_imgs, save_title=save_tit, show = True)

def test_denoise(imgs, denoise = denoise_bilateral, 
				 save_title = 'denoise.png', show_bool=False):
    ''' 
    Arguments:
        imgs - Dictionary of gray scaled images, denoise function, plotting options. 
    Output: Plot of images with denoise applied. 

    Apply the denoise function to the gray-scaled, canny-filtered images and examine the plots
    to see which denoise method we might want to use. 
     (sigma_range is deprecated)
    '''
    denoised_imgs = {}
    if denoise == denoise_bilateral: 
        print("bilateral denoising") 
        for x in np.arange(start=0.2, stop=1, step=0.2): 
            for k, v in imgs.items(): 
                filtered_img = canny(v, sigma=1)
                denoised_imgs[k] = denoise(filtered_img, sigma_spatial=x, multichannel=False)
            if show_bool:
                print("Sigma, spatial: {}".format(x)) 
                tit = 'Sig=' + str(np.round(x, 2)) + ':' + save_title
                plt.suptitle(tit)
                plot_images(denoised_imgs, save_title = tit, show = show_bool)
    elif denoise == denoise_tv_chambolle: 
        print("tv_chambolle denoising") 
        for x in np.arange(start=0.3, stop=3, step=0.54): 
            for k, v in imgs.items(): 
                filtered_img = canny(v, sigma=1)
                denoised_imgs[k] = denoise(filtered_img, weight=x)
            if show_bool: 
                print("Chambolle weight: {}".format(x)) 
                tit = 'Weight=' + str(x) + ':' + save_title
                plt.suptitle(tit)
                plot_images(denoised_imgs, save_title = tit, show = show_bool)
    return denoised_imgs 


def apply_KMeans(color_imgs):
	'''
	Arguments: 
        color_imgs - Dictionary of Color Images
	Returns: 
        Clusters of colors per image. 

	Fit KMeans to each class of images (here only 2) to get clusters of colors. 
	'''
	clusters = []
	for img in color_imgs.values(): 
		nrow, ncol, depth = img.shape 
		lst_of_pixels = [img[irow][icol] for irow in range(nrow) for icol in range(ncol)]
		X = np.array(lst_of_pixels)

		sklearn_km = KMeans(n_clusters=3)
		result = sklearn_km.fit_predict(X)
		clusters.append(sklearn_km.cluster_centers_)

	return clusters

if __name__ == '__main__':
    
    ihc_rgb = data.immunohistochemistry()  # sample from library
    ihc_hed = dye_color_separation(ihc_rgb)
    ax = plot_dye_separation(ihc_rgb, ihc_hed)

    rescaled_img = rescale_dye_signals(ihc_hed, ax)
    
