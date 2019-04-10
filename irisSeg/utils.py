import numpy as np
from scipy.signal import convolve

_round = lambda x: np.round(x).astype(np.uint8)


def search(image, rmin, rmax, x, y, feature):
    '''
        function to detect the pupil  boundary
        it searches a certain subset of the image
        with a given radius range(rmin,rmax)
        around a 10*10 neighbourhood of the point x,y given as input

    :param image: image to be processed
    :param rmin: min radius
    :param rmax: max radius
    :param x: x - coord of center point
    :param y: y - coord of center point
    :param feature: 'pupil' or 'iris'
    :return: Center coord followed by radius
    '''
    sigma = 0.5
    maxRadius = np.zeros(image.shape)
    maxBlur = np.zeros(image.shape)
    for i in np.arange(int(x) - 5, int(x) + 5):
        for j in np.arange(int(y) - 5, int(y) + 5):
            max_blur, max_blur_radius, blur = partialDerivative(image, [i, j], rmin, rmax, sigma, 600, feature)
            maxRadius[i, j] = max_blur_radius
            maxBlur[i, j] = max_blur
    X, Y = np.where(maxBlur == maxBlur.max())
    radius = maxRadius[X, Y]
    coordPupil = np.array([X, Y, radius])
    return coordPupil


def NormalLineIntegral(image, coord, r, n, feature):
    '''
      function to calculate the normalised line integral around a circular contour
      A polygon of large number of sides approximates a circle and hence is used
      here to calculate the line integral by summation
      if the search is for the pupil,the function uses the entire circle(polygon) for computing L
      for the iris only the lateral portions are used to mitigate the effect of occlusions
      that might occur at the top and/or at the bottom

    :param image: image to be processed
    :param coord: [x,y]  center coord of the circum center < Origin is Top Left Corner >
    :param r: radius of the circum circle
    :param n: number of sides
    :param feature: To indicate wheter search is for iris or pupil
    :return: the line integral divided by circumference
    '''
    theta = (2 * np.pi) / n  # angle subtended at the center by the sides
    # orient one of the radii to lie along the y axis
    rows, cols = image.shape
    angle = np.arange(theta, 2 * np.pi, theta)
    x = coord[0] - r * np.sin(angle)
    y = coord[1] + r * np.cos(angle)

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if np.any(x >= rows) or np.any(y >= cols) or np.any(x <= 1) or np.any(y <= 1):
        # This process returns L=0 for any circle that does not fit inside the image
        return 0
    if feature == 'pupil':
        s = 0
        for i in np.arange(0, n - 1):
            val = image[_round(x[i]), _round(y[i])]
            s += val
        line = s / n
        return line

    elif feature == 'iris':
        s = 0
        for i in np.arange(1, _round(n / 8)):
            val = image[_round(x[i]), _round(y[i])]
            s += val

        for i in np.arange(_round(3 * n / 8) + 1, _round((5 * n / 8))):
            val = image[_round(x[i]), _round(y[i])]
            s += val

        for i in np.arange(np.round((7 * n / 8)).astype(np.uint8) + 1, n - 1):
            val = image[_round(x[i]), _round(y[i])]
            s += val

        line = (2 * s) / n
        return line


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def partialDerivative(image, coord, rmin, rmax, sigma, n, feature):
    '''
        calculates the partial derivative of the normailzed line integral
        holding the centre coordinates constant
        and then smooths it by a gaussian of appropriate sigma
        rmin and rmax are the minimum and maximum values of radii expected
        function also returns the maximum value of blur and the corresponding radius
        with finite differnce vector blur

    :param image: preprocessed image
    :param coord: centre coordinates
    :param rmin: min radius
    :param rmax: max radius
    :param sigma: standard deviation of the gaussian
    :param n: number of sides of the polygon(for LineIntegral)
    :param feature: pupil or Iris
    :return: It gives finite differences vector, max value of blur
            and radius at max blur
    '''
    R = np.arange(rmin, rmax)
    count = R.shape[0]

    lineIntegral = []
    # lineIntegral = np.empty(7)

    for k in np.arange(0, count):
        # computing the normalized line integral for each radius
        temp = NormalLineIntegral(image, coord, R[k], n, feature)
        if temp == 0:
            # this case occurs iff the radius takes the circle out of the image
            # In this case,L is deleted as shown below and no more radii are taken for computation
            # (for that particular centre point).This is accomplished using the break statement
            break
        else:
            lineIntegral.append(temp)
            # np.append(lineIntegral,temp)
    if not isinstance(lineIntegral, np.ndarray):
        lineIntegral = np.array(lineIntegral)

    disc_diff = np.diff(lineIntegral)
    D = np.concatenate(([0], disc_diff))  # append one element at the beginning

    if sigma == 'inf':
        kernel = np.ones(7) / 7
    else:
        kernel = matlab_style_gauss2D([1, 5], sigma)  # generates a 5 member 1-D gaussian
        kernel = np.reshape(kernel, kernel.shape[1], order='F')

    blur = np.abs(convolve(D, kernel, 'same'))
    # Smooths the D vecor by 1-D convolution

    values, index = blur.max(0), blur.argmax(0)
    max_blur_radius = R[index]
    max_blur = blur[index]
    return max_blur, max_blur_radius, blur


def drawcircle(I, C, r, n=600):
    '''
        generate the pixels on the boundary of a regular polygon of n sides
        the polygon approximates a circle of radius r and is used to draw the circle
    :param I: image to be processed
    :param C: [x,y] Centre coordinates of the circumcircle
    :param r: radius of the circumcircle
    :param n: no of sides
    :return: Image with circle
    '''
    theta = (2 * np.pi) / n
    rows, cols = I.shape
    angle = np.arange(theta, 2 * np.pi, theta)

    x = C[0] - r * np.sin(angle)
    y = C[1] + r * np.cos(angle)

    if np.any(x >= rows) or np.any(y >= cols) or np.any(x <= 1) or np.any(y <= 1):
        return I
    for i in np.arange(1, n - 1):
        I[np.round(x[i]).astype(np.uint8), np.round(y[i]).astype(np.uint8)] = 1
    return I
