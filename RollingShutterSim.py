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
    """
    Calculates the probability given by the two dimensional Gaussian function for a calculated meteor position. 
    
    Arguments:
        x [float or numpy meshgrid]: X coordinate of a point for which the probability is calculated
        y [float or numpy meshgrid]: Y coordinate of the point
        x0 [float]: X coordinate of the center of the distribution
        y0 [float]: Y coordinate of the center of the distribution
        sigma_x [float]: standard deviation along the X axis
        sigma_y [float]: standard deviation along the Y axis
        amplitude [float]: amplitude of the Gaussian distribution

    Return:
        p: probability at a given point
    """

    delta_x = x - x0
    delta_y = y - y0

    p = amplitude * np.exp(- ((delta_x / (np.sqrt(2) * sigma_x)) ** 2 + (delta_y / (np.sqrt(2) * sigma_y)) ** 2))

    return p



def meteorCentroid(img, x0, y0, x_start, x_finish, y_start, y_finish):

    x_start = int(round(x_start))
    x_finish = int(round(x_finish))
    y_start = int(round(y_start))
    y_finish = int(round(y_finish))


    # Calculate length of meteor
    r = np.sqrt((x_finish - x_start)**2 + (x_finish - x_start)**2)

    # Crop image
    img_crop = img[y_start:y_finish, x_start:x_finish]

    plt.imshow(img_crop, cmap='gray')
    plt.show()


    # Sums
    sum_x = 0
    sum_y = 0
    sum_intens = 0

    # Background noise value
    nx, ny = img.shape
    #print(nx, ny)
    y, x = np.ogrid[:ny, :nx]
    img_mask = ((x - x0)**2 + (y - y0)**2 > r**2)
    #plt.imshow(img_mask)
    #plt.show()
    back_noise = np.ma.masked_array(img, mask = img_mask).mean()

    for x in range(img_crop.shape[1]):
        for y in range(img_crop.shape[0]):
        

            value = img_crop[y, x] - back_noise

            sum_x += x * value
            sum_y += y * value
            sum_intens += value

    print(sum_x/sum_intens, sum_y/sum_intens)
    
    # Calculate centroid coordinates
    x_centr = sum_x/sum_intens + x_start
    y_centr = sum_y/sum_intens + y_start


    return (x_centr, y_centr)


if __name__ == "__main__":

    ### Defining function parameters ###
    ####################################

    # Define multiplication factor and framerate for video
    multi_factor = 10
    framerate = (1/30)/multi_factor

    # Array of points in time
    t_meteor = 0.5 / 2
    t_arr = np.arange(-t_meteor, t_meteor, framerate)

    # Image size
    img_x = 720
    img_y = 576

    # Level of background offset
    offset = 20

    # Center of image
    x0 = img_x/2
    y0 = img_y/2

    # Pixel scale in px/deg
    scale = img_x/64

    # Meteor angle counterclockwise from the Y axis (deg)
    phi = 45

    # Meteor's angular velocity (deg/s)
    omega = 30

    # Construct frame grid
    x = np.arange(0, img_x)
    y = np.arange(0, img_y)

    xx, yy = np.meshgrid(x, y)

    # Amplitude and standard deviation of two dimensional gaussian function
    amplitude = 255/multi_factor
    sigma_x = 2
    sigma_y = 2

    ### Displaying the simulated meteor and the coordinates of the centroid ###
    ###########################################################################

    # Make video representation
    for i in range(multi_factor):

        # Defining time limits
        t_start = -t_meteor + i*multi_factor*framerate
        t_finish = -t_meteor + (i + 1)*multi_factor*framerate

        # Array of points in time defined by framerate
        t_arr_iter = np.arange(t_start, t_finish, framerate)

        # Calculate beginning and ending points of meteor
        x_start, y_start = drawPoints(t_start, img_x, img_y, scale, phi, omega)
        x_finish, y_finish = drawPoints(t_finish, img_x, img_y, scale, phi, omega)

        # Make sure beginnings are smaller then endings
        x_start, x_finish = sorted([x_start, x_finish])
        y_start, y_finish = sorted([y_start, y_finish])

        # Calculate beginning end ending points of crop
        if np.sin(np.radians(phi)) >= 0:
            x_start -= 3*sigma_x
            x_finish += 3*sigma_x

        else:
            x_start += 3*sigma_x
            x_finish -= 3*sigma_x


        if np.cos(np.radians(phi)) >= 0:
            y_start -= 3*sigma_y
            y_finish += 3*sigma_y

        else:
            y_start += 3*sigma_y
            y_finish -= 3*sigma_y


        print('Start end:')
        print(x_start, x_finish)
        print(y_start, y_finish)



        # Image array
        img_array = np.zeros((img_y, img_x), np.float_)


        for t in t_arr_iter:
            # Draw two dimensional Gaussian function for each point in time
            x, y = drawPoints(t, img_x, img_y, scale, phi, omega)
            temp = twoDimensionalGaussian(xx, yy, x, y, sigma_x, sigma_y, amplitude)
            img_array += temp

        # Add Gaussian noise and offset
        noise = np.random.normal(loc = 0, scale = 10, size = (img_y, img_x))
        img_array += abs(noise) + offset

        # Clip pixel levels
        np.clip(img_array, 0, 255)

        # Convert image to 8-bit unsigned integer
        img_array = img_array.astype(np.uint8)

        # Centroid coordinates
        t_mid = (t_start + t_finish)/2
        x_mid, y_mid = drawPoints(t_mid, img_x, img_y, scale, phi, omega)
        x_centr, y_centr = meteorCentroid(img_array, x_mid, y_mid, x_start, x_finish, y_start, y_finish)

        # Model coordinates
        x_model, y_model = drawPoints(t_mid, img_x, img_y, scale, phi, omega)
        
        # Show frame
        plt.imshow(img_array, cmap = "gray", vmin = 0, vmax = 255)
        plt.scatter(x_centr, y_centr, marker = 'o')
        plt.scatter(x_model, y_model, marker = '*')
        plt.show()


    """
    ### Displaying the meteor's movement ###
    ########################################

    x_arr, y_arr = drawPoints(t_arr, img_x, img_y, scale, phi, omega)

    # Make scatter plot
    plt.scatter(x_arr, y_arr, c = t_arr)

    # Set plot size
    plt.xlim([0, img_x])
    plt.ylim([img_y, 0])

    plt.colorbar(label = 'Time (s)')

    plt.show()
    """