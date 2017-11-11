
# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt


def drawPoints(t, img_x, img_y, scale, phi, omega):
    """ Calculate position of a meteor on image with the given simulation parameters. 

    Arguments:
        t: [float or ndarray] Time in seconds.
        img_x: [int] Image X size.
        img_y: [int] Image Y size.
        scale: [float] Image scale in px/deg
        phi: [float] Meteor angle, counterclockwise from Y axis.
        omega: [float] Meteor angular velocity in deg/s.

    Return:
        (x, y): [tuple of floats] X and Y coordinate of the meteor.
    """

    # Image centre
    x0 = img_x/2
    y0 = img_y/2

    # Convert angle to radians
    phi = np.radians(phi)

    # Calculate distance from centre in pixels
    r = omega*t*scale

    # Calculate position of meteor on the image
    x = x0 - np.sin(phi)*r
    y = y0 + np.cos(phi)*r

    return (x, y)


def twoDimensionalGaussian(x, y, x0, y0, sigma_x, sigma_y, amplitude):
    delta_x = x - x0
    delta_y = y - y0

    return amplitude * np.exp(- ((delta_x / (np.sqrt(2) * sigma_x)) ** 2 + (delta_y / (np.sqrt(2) * sigma_y)) ** 2))


if __name__ == "__main__":

    # Array of points in time
    t_meteor = 0.5 / 2
    t_arr = np.arange(-t_meteor, t_meteor, 1/300)

    # Image size
    img_x = 720
    img_y = 576

    # Pixel scale in px/deg
    scale = 720/64

    # Meteor angle counterclockwise from the Y axis (deg)
    phi = 15

    # Meteor's angular velocity (deg/s)
    omega = 35

    x = np.arange(0, img_x)
    y = np.arange(0, img_y)

    xx, yy = np.meshgrid(x, y)

    # Define multiplication factor and other parameters for video
    multi_factor = 10
    rate = 1/300

    # Amplitude and standard deviation of two dimensional gaussian function
    amplitude = 255/multi_factor
    sigma_x = 2
    sigma_y = 2

    # Show twoDimensionalGaussian function for each point in time
    """
    for t in t_arr:
        x, y = drawPoints(t, img_x, img_y, scale, phi, omega)
        temp = twoDimensionalGaussian(xx, yy, x, y, sigma_x, sigma_y, amplitude)
        img_array += temp
 
    plt.imshow(img_array)
    plt.show()
    """

    # Make video representation
    for i in range(multi_factor):
        t_start = -t_meteor + i * multi_factor * rate
        t_finish = -t_meteor + (i + 1) * multi_factor * rate

        t_arr_iter = np.arange(t_start, t_finish, rate)

        img_array = np.zeros((img_y, img_x), np.float_)

        for t in t_arr_iter:
            x, y = drawPoints(t, img_x, img_y, scale, phi, omega)
            temp = twoDimensionalGaussian(xx, yy, x, y, sigma_x, sigma_y, amplitude)
            img_array += temp

        #Add Gaussian noise

        noise = np.random.normal(loc = 0, scale = 10, size = (img_y, img_x)) 
        plt.imshow(img_array + noise)
        plt.show()


    # Calculate positions of a meteor in different points in time
    x_arr, y_arr = drawPoints(t_arr, img_x, img_y, scale, phi, omega)

    # Make scatter plot
    plt.scatter(x_arr, y_arr, c=t_arr)

    # Set plot size
    plt.xlim([0, img_x])
    plt.ylim([img_y, 0])

    plt.colorbar(label='Time (s)')

    plt.show()
    