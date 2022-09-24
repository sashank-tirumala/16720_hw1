import os
import multiprocessing
from os.path import join, isfile
import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans
from opts import get_opts
import util
from pathlib import Path
import shutil


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    filter_scales = opts.filter_scales
    # ----- TODO -----
    #img = 1
    channel_count = img.shape[-1]
    filter_responses = []
    #breakpoint()
    if img.max() > 1:
        img = img/255.0
    if channel_count == 4:
        img = img[:,:,:3]
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    img = skimage.color.rgb2lab(img)
    for filter_cur in filter_scales:
        out = []
        for channel in range(3):
            filtered_channel = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma=filter_cur)
            out.append(filtered_channel)

        for channel in range(3):
            filtered_channel = scipy.ndimage.gaussian_laplace(img[:,:,channel], sigma=filter_cur)
            out.append(filtered_channel)
            
        for channel in range(3):
            filtered_channel = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma=filter_cur, order = (0,1))
            out.append(filtered_channel)

        for channel in range(3):
            filtered_channel = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma=filter_cur, order = (1,0))
            out.append(filtered_channel)
        out = np.stack(out, axis=-1)
        filter_responses.append(out)
    filter_responses = np.concatenate(filter_responses, axis=-1)
    return filter_responses


def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """
    img = skimage.io.imread(args['img_path'])
    pixel_vals = np.argwhere(np.ones(img.shape[:-1]))
    pixel_choice = np.random.randint(pixel_vals.shape[0], size = args['opts'].alpha)
    pixel_vals = pixel_vals[pixel_choice]
    filter_response = extract_filter_responses(args['opts'], img)
    outp = filter_response[pixel_vals[:,0], pixel_vals[:,1]]
    np.save("tmp/"+str(args['id'])+".npy", outp)
    
    return outp


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    inps=[]
    for i in range(len(train_files)):
        tmp={'opts':opts, "img_path":opts.data_dir+"/"+train_files[i], "id":i}
        inps.append(tmp)
    os.mkdir("tmp")
    pool = multiprocessing.Pool(processes=64)
    pool.map(compute_dictionary_one_image, inps)
    arrs = []
    for npfile in os.listdir("tmp/"):
        cur_file = "tmp/"+npfile
        arr = np.load(cur_file)
        arrs.append(arr)
    arrs = np.concatenate(arrs, axis=0)
    kmeans = KMeans(n_clusters=K).fit(arrs)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    shutil.rmtree("tmp")


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    filter_response = extract_filter_responses(opts, img)
    n = filter_response.shape[-1]
    dists = scipy.spatial.distance.cdist(filter_response.reshape(-1,n),dictionary)
    mins = np.argmin(dists, axis=1)
    wordmap = mins.reshape(img.shape[:-1])
    return wordmap

def aquarium_plot(opts):
    opts.filter_scales=[1,2,4,8,16]
    img_path = opts.data_dir+"/aquarium/sun_aztvjgubyrgvirup.jpg"
    img = skimage.io.imread(img_path)
    filters = extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filters)

def plot_wordmap():
    opts = get_opts()
    dictionary = np.load("dictionary.npy")
    train_files = open(join(opts.data_dir, "train_files_small.txt")).read().splitlines()
    train_files = train_files[:3]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,3)
    for i in range(len(train_files)):
        img_path = opts.data_dir+"/"+train_files[i]
        img = skimage.io.imread(img_path)
        wordmap = get_visual_words(opts, img, dictionary)
        ax[0,i].imshow(img)
        ax[1,i].imshow(wordmap)
    plt.savefig("wordmaps")

if  __name__ == "__main__":
    plot_wordmap()
    exit()
    opts = get_opts()
    img_path = opts.data_dir+"/aquarium/sun_aztvjgubyrgvirup.jpg"
    img = skimage.io.imread(img_path)
    dictionary = np.load("dictionary.npy")
    words = get_visual_words(opts, img, dictionary)
