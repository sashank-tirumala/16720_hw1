import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
from visual_words import get_visual_words
from opts import get_opts
import skimage

import shutil
import sklearn

def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    hist, _ = np.histogram(wordmap, K)
    hist  = hist/hist.sum()
    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    K = opts.K
    L = opts.L
    H,W = wordmap.shape
    hists = []
    for l in range(L+1):
        x_indices = np.array([ (H/2**l) * i for i in range(2**l+1)]).astype(int) 
        y_indices = np.array([ (W/2**l) * i for i in range(2**l+1)]).astype(int) 
        for i in range(len(x_indices)-1):
            for j in range(len(y_indices)-1):
                cur_wordmap = wordmap[int(x_indices[i]): int(x_indices[i+1]), int(y_indices[j]):int(y_indices[j+1])] 
                hist = get_feature_from_wordmap(opts, cur_wordmap)                
                if l ==0  or l ==1:
                    wt = 2**(-L)
                else:
                    wt = 2**(l-L-1)
                hist = hist *wt
                hists.append(hist)
    hists = np.concatenate(hists)
    hists = hists/hists.sum()
    return hists


def get_image_feature(args):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """
    opts, img_path, dictionary, idx = args
    img = skimage.io.imread(img_path)
    wordmap = get_visual_words(opts, img, dictionary)
    hist = get_feature_from_wordmap_SPM(opts, wordmap)
    np.save("tmp/"+str(idx)+".npy", hist)
    return hist


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))
    all_args=[]
    features = []
    idx = 0
    os.mkdir("tmp")
    for file_name in train_files:
        img_path = opts.data_dir+"/"+file_name
        all_args.append((opts, img_path, dictionary, idx))
        idx += 1
    pool = multiprocessing.Pool(processes = 64)
    pool.map(get_image_feature, all_args)
    N = len(os.listdir('tmp/'))
    for i in range(N):
        npfile = str(i)+".npy" 
        cur_file = "tmp/"+npfile
        arr = np.load(cur_file)
        features.append(arr)
    shutil.rmtree("tmp/")
    features = np.stack(features, axis=0)
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def similarity_to_set(word_hist, histograms):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """
    N = histograms.shape[0]
    large_word_hist = np.stack([word_hist]*N)
    comb = np.stack([large_word_hist, histograms], axis=-1)
    comb = comb.min(axis=-1)
    sim = comb.sum(axis=-1)
    return sim


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    histograms = trained_system["features"]
    train_labels = trained_system["labels"]
    all_args = []
    idx = 0
    for test_file in test_files:
        img_path = opts.data_dir + "/" + test_file
        all_args.append((opts, img_path, dictionary, histograms, idx))
        idx+=1
    os.mkdir("tmp")
    pool = multiprocessing.Pool(processes = 64)
    pool.map(compute_single_test_image_sim, all_args)
    pool.close()
    pred_labels = np.zeros(len(test_labels))
    sim_scores = np.zeros(len(test_labels))
    for file_name in os.listdir("tmp/"):
        idx = int(file_name.split(".")[0])
        sim_index = np.load("tmp/"+file_name)
        sim_index = sim_index.argmax()
        pred_label = train_labels[sim_index]
        pred_labels[idx] = pred_label
    shutil.rmtree("tmp")
    breakpoint()
    acc = (pred_labels == test_labels).sum()/len(pred_labels) 
    conf = sklearn.metrics.confusion_matrix(test_labels, pred_labels) 
    return conf, acc

def compute_single_test_image_sim(args):
    opts, img_path, dictionary, histograms, idx = args
    img = skimage.io.imread(img_path)
    wordmap = get_visual_words(opts, img, dictionary)
    hist = get_feature_from_wordmap_SPM(opts, wordmap)
    sim = similarity_to_set(hist,histograms)
    np.save("tmp/"+str(idx)+".npy", sim)
    return sim

def compute_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass

def evaluate_recognition_System_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass

def test_wordmap():
    opts = get_opts()
    img_path = opts.data_dir + "/aquarium/sun_aztvjgubyrgvirup.jpg"
    img = skimage.io.imread(img_path)
    dictionary = np.load("dictionary.npy")
    wordmap = get_visual_words(opts, img, dictionary)
    opts.L = 3
    hist = get_feature_from_wordmap_SPM(opts, wordmap) 

def test_sim():
    opts = get_opts()
    img_path = opts.data_dir + "/aquarium/sun_aztvjgubyrgvirup.jpg"
    img = skimage.io.imread(img_path)
    dictionary = np.load("dictionary.npy")
    wordmap = get_visual_words(opts, img, dictionary)
    opts.L = 3
    hist = get_feature_from_wordmap_SPM(opts, wordmap) 
    img_files = os.listdir(opts.data_dir+"/aquarium/")
    hist_list = []
    count = 0
    for fl in img_files:
        img = skimage.io.imread(opts.data_dir+"/aquarium/"+fl)
        cur_wordmap = get_visual_words(opts, img, dictionary)
        cur_hist = get_feature_from_wordmap_SPM(opts, cur_wordmap)
        hist_list.append(cur_hist)
        count +=1
        if(count >3):
            break
    hist_list.append(hist)
    hist_list = np.stack(hist_list, axis=0)
    sim = similarity_to_set(hist, hist_list)

if __name__ == "__main__":
    #test_sim() 
    opts = get_opts()
    #build_recognition_system(opts)
    evaluate_recognition_system(opts)
    pass
