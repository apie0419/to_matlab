import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import easygui

from matplotlib.widgets import RectangleSelector

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

def pca(x, centered=True, economy=True):

    if centered:
        x = x - np.mean(x, axis=0)
        dof = len(x) - 1
    else:
        dof = len(x)
    
    cov = np.dot(x.T, x)
    
    latent, coeff = np.linalg.eig(cov.astype(np.float32))
    coeff = coeff.astype(np.float32)
    latent = latent.astype(np.float32)
    idx = np.argsort(latent)[::-1]
    coeff = coeff[:, idx]
    latent = latent[idx]
    
    if economy:
        latent = latent[:dof] / dof
        coeff = coeff[:, :dof]

    scores = np.dot(x, coeff)
    explained = 100 * latent / np.sum(latent)

    return coeff.astype(np.float32), scores.astype(np.float32), latent.astype(np.float32), explained.astype(np.float32)

def imrect(filename):

    def onselect(eclick, erelease):
        pass

    def toggle_selector(event):
        if event.dblclick:
            toggle_selector.RS.set_active(False)
            plt.close()

    def fix():
        """
            position: (x1, x2, y1, y2)
        """

        position = toggle_selector.RS.extents

        return int(position[0]), int(position[1]), int(position[2]), int(position[3])

    img = mpimg.imread(filename)

    fig, ax = plt.subplots(num="Figure")
    ax.imshow(img)

    toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box', interactive=True)
    fig.canvas.mpl_connect('button_press_event', toggle_selector)
    plt.axis('off')
    plt.show()

    if toggle_selector.RS.extents == (0, 0, 0, 1):
        return (0, img.shape[0], 0, img.shape[1])

    position = fix()

    return position

if __name__ == "__main__":
    a = np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10], [9, 10, 11 ,12, 13, 14]])
    b = np.array([[6, 2, 3, 10], [2, 31, 51, 20], [10, 5, 21, 15]])
    
    coeff, scores, latent, explained = pca(a, centered=False)
    # print (scores @ coeff.T)
    print ("coeff:\n", coeff)
    print ("scores:\n", scores)
    print ("latent:", latent)
    print ("explained:", explained)

    # coeff, scores, latent, explained = pca(b, centered=False)

    # print ("coeff:\n", coeff)
    # print ("scores:\n", scores)
    # print ("latent:", latent)
    # print ("explained:", explained)