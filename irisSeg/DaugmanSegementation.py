import numpy as np
import cv2
from scipy.misc import imresize
from skimage.morphology import erosion
import matplotlib.pyplot as plt
from .utils import partialDerivative, search, drawcircle

# this function returns the indexes that satisfy certain function func
indices = lambda a, func: [i for (i, val) in enumerate(a) if func(val)]

#  This function replicates the matlab function rgb2gray
rgb2gray = lambda x: np.dot(x[..., :3], [0.2989, 0.5870, 0.1140])

#  This function replicates the matlab function im2double
im2double = lambda im: im.astype(np.float) / np.iinfo(im.dtype).max  # Divide all values by the largest possible value in the datatype


def irisSeg(filename, rmin, rmax, view_output=False):
    '''
        To search the center coordinates of the iris and pupil and their respective radii
         Camus&Wildes' method to select possible center coordinates
         detect Iris using Daugman's Daugman's integrodifferential operator

    :param filename:  image file path
    :param rmin:  min radius of the iris
    :param rmax:  max radius of the iris
    :param view_output: view the resulatant image if set to True
    :return: returns the x and y coordinate of Pupil and Iris and Resultant Segemented Image
    '''
    image = cv2.imread(filename, -1)

    scale = 1
    # Libor Masek's idea that reduces complexity
    # significantly by scaling down all images to a constant image size
    # to speed up the whole process
    rmin = rmin * scale
    rmax = rmax * scale

    image = im2double(image)

    pimage = image  # store the image for display

    image = imresize(image, 1 / scale)

    image = erosion(image)

    rows, cols = image.shape

    X, Y = np.where(image < 0.5)  # Generates a column vector of the image elements

    s = X.shape[0]
    nan = -99999
    for k in np.arange(0, s):
        if (X[k] > rmin) and (Y[k] > rmin) and (X[k] <= (rows - rmin)) and (Y[k] < (cols - rmin)):
            A = image[(X[k] - 1):(X[k] + 1), (Y[k] - 1):(Y[k] + 1)]
            M = A.min()
            # this process scans the neighbourhood of the selected pixel
            # to check if it is a local minimum
            if image[X[k], Y[k]] != M:
                X[k] = nan
                Y[k] = nan

    d_index = np.where(X == nan)
    X = np.delete(X, d_index)
    Y = np.delete(Y, d_index)
    # deletes all pixels that are NOT local minima(that have been set to NaN)

    d_index = np.where((X <= rmin) | (Y <= rmin) | (X > (rows - rmin)) | (Y > (cols - rmin)))
    X = np.delete(X, d_index)
    Y = np.delete(Y, d_index)
    # delete all the points near the border that can't possibly be the center coord

    N = X.shape[0]
    maxBlur = np.zeros((rows, cols))
    maxRadius = np.zeros((rows, cols))
    # defines two arrays maxb and maxrad to store the maximum value of blur
    # for each of the selected centre points and the corresponding radius
    for j in np.arange(0, N):
        max_blur, max_blur_radius, blur = partialDerivative(image, [X[j], Y[j]], rmin, rmax, 'inf', 600, 'iris')
        maxBlur[X[j], Y[j]] = max_blur
        maxRadius[X[j], Y[j]] = max_blur_radius

    x, y = np.where(maxBlur == maxBlur.max())

    coord_iris = search(image, rmin, rmax, x[0], y[0], 'iris')
    # finds the maximum value of blur by scanning all the centre coordinates
    coord_iris = coord_iris / scale
    # the function search searches for the centre of the pupil and its radius
    # by scanning a 10*10 window around the iris centre for establishing
    # the pupil's centre and hence its radius
    coord_pupil = search(image, np.round(0.1 * max_blur_radius), np.round(0.8 * max_blur_radius), coord_iris[0] * scale,
                         coord_iris[1] * scale, 'pupil')
    coord_pupil = coord_pupil / scale

    segemented_img = drawcircle(pimage, [coord_iris[0], coord_iris[1]], coord_iris[2], 600)
    segemented_img = drawcircle(segemented_img, [coord_pupil[0], coord_pupil[1]], coord_pupil[2], 600)
    if view_output:
        # displaying the segmented image
        plt.imshow(segemented_img)
        plt.show()
    return coord_iris, coord_pupil, segemented_img


if __name__ == '__main__':
    # UBIRIS_200_150_R/Sessao_1/3/Img_3_1_1.jpg

    # img = cv2.imread('UBIRIS_200_150_R/Sessao_1/3/Img_3_1_1.jpg',-1)
    # img = cv2.imread('pupilorig.jpg',-1)

    # print(img.shape)

    coord_iris, coord_pupil, output_image = irisSeg('Data/sample_img.jpg', 40, 70, view_output=True)
    # ci,co,output = main(img,40,90)

    print(coord_iris)
    print(coord_pupil)
