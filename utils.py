import numpy as np
import easygui

np.set_printoptions(suppress=True)

def rgb2lin(srgb):

    ## Reference1: https://www.mathworks.com/help/images/ref/rgb2lin.html
    ## Reference2: https://github.com/PetterS/opencv_srgb_gamma/blob/master/srgb.py

    linear = np.float32(srgb) / 255.0
    less = linear <= 0.04045
    linear[less] = linear[less] / 12.92
    linear[~less] = np.power((linear[~less] + 0.055) / 1.055, 2.4)
    linear = linear * 255.0
    linear = np.round(linear).astype(np.uint8)

    return linear

def lin2rgb(linear):

    srgb = np.float32(linear.copy()) / 255.0
    
    less = srgb <= 0.0031308
    srgb[less] = srgb[less] * 12.92
    srgb[~less] = 1.055 * np.power(srgb[~less], 1.0 / 2.4) - 0.055

    return np.round(srgb * 255.0).astype(np.uint8)

def pca(x):

    # x = x - np.mean(x, axis=0)
    dof = len(x)
    cov = np.cov(x, rowvar = False)
    latent, coeff = np.linalg.eig(cov)
    idx = np.argsort(latent)[::-1]
    coeff = coeff[:, idx]
    latent = latent[idx]
    coeff = coeff[:, :dof]
    latent = latent[:dof]
    
    scores = np.dot(x, coeff) 

    # scores = (scores - np.mean(scores, axis=0)) / np.std(scores, axis=0)

    explained = 100 * latent / np.sum(latent)

    return coeff.astype(np.float32), scores.astype(np.float32), latent.astype(np.float32), explained.astype(np.float32)

def load(filename):
    
    res = list()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            res.append(line.replace("\n", "").split("\t"))

    return np.array(res)

    