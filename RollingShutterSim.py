# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt


def drawPoints(t, x_center, y_center, scale, phi, omega):
    """ Calculate position of a meteor on image with the given simulation parameters. 

    Arguments:
        t: [float or ndarray] Time in seconds.
        x_center: [int] Image X center.
        y_center: [int] Image Y center.
        scale: [float] Image scale in px/deg.
        phi: [int or float] Meteor angle, counterclockwise from Y axis.
        omega: [int or float] Meteor angular velocity in deg/s.

    Return:
        (x_meteor, y_meteor): [tuple of floats] X and Y coordinate of the meteor.
    """

    # Convert angle to radians
    phi = np.radians(phi)

    # Calculate distance from centre in pixels
    z = omega*t*scale

    # Calculate position of meteor on the image
    x_meteor = x_center - np.sin(phi)*z
    y_meteor = y_center + np.cos(phi)*z

    return (x_meteor, y_meteor)


def twoDimensionalGaussian(x, y, x0, y0, sigma_x, sigma_y, amplitude):
    """
    Calculates the probability given by the two dimensional Gaussian function for a calculated meteor position. 
    
    Arguments:
        x: [float or numpy meshgrid] X coordinate of a point for which the probability is calculated.
        y: [float or numpy meshgrid] Y coordinate of the point.
        x0: [int or float] X coordinate of the center of the distribution.
        y0: [int or float] Y coordinate of the center of the distribution.
        sigma_x: [float] Standard deviation along the X axis.
        sigma_y: [float] Standard deviation along the Y axis.
        amplitude: [float] Amplitude of the Gaussian distribution.

    Return:
        p [float]: Probability at a given point.
    """

    delta_x = x - x0
    delta_y = y - y0

    p = amplitude * np.exp(- ((delta_x / (np.sqrt(2) * sigma_x)) ** 2 + (delta_y / (np.sqrt(2) * sigma_y)) ** 2))


    return p



def meteorCentroid(img, x0, y0, x_start, x_finish, y_start, y_finish):
    """
    Calculates the X and Y coordinates of a meteor centroid.

    Arguments:
        img: [numpy ndarray] Image.
        x0: [float] X coordinate of centroid's center.
        y0: [float] Y coordinate of centroid's center.
        x_start: [float] X coordinate of beginning crop point.
        x_finish: [float] X coordinate of ending crop point.
        y_start: [float] Y coordinate of beginning crop point.
        y_finish: [float] Y coordinate of ending crop point.

    Return:
        (x_centr, y_centr): [tuple of floats] X and Y coordinates of the calculated centroid, 
    """

    # Calculate length of meteor
    r = np.sqrt((x_finish - x_start)**2 + (x_finish - x_start)**2)

    # Crop image
    img_crop = img[y_start:y_finish, x_start:x_finish]

    # Define intensity sums
    sum_x = 0
    sum_y = 0
    sum_intens = 0

    # Background noise value
    nx, ny = img.shape
    y, x = np.ogrid[:ny, :nx]
    img_mask = ((x - x0)**2 + (y - y0)**2 > r**2)
    back_noise = np.ma.masked_array(img, mask = img_mask).mean()

    # Evaluate intensity sums
    for x in range(img_crop.shape[1]):
        for y in range(img_crop.shape[0]):

            value = img_crop[y, x] - back_noise

            sum_x += x*value
            sum_y += y*value

            sum_intens += value

    #print(sum_x/sum_intens, sum_y/sum_intens)
    
    # Calculate centroid coordinates
    x_centr = sum_x/sum_intens + x_start
    y_centr = sum_y/sum_intens + y_start


    return (x_centr, y_centr)


def pointsCentroidAndModel(t_meteor, phi, omega, img_x, img_y, scale, multi_factor, sigma_x, sigma_y, noise_scale, offset, show_plots):
    """
    Returns coordinates of meteor center calculated by centroiding and from a meteeor movement model.

    Arguments:
        t_meteor: [int or float] Duration of meteor.         
        phi: [int or float] Meteor angle, counterclockwise from Y axis.
        omega: [int or float] Meteor angular velocity in deg.
        img_x: [int] Size of image X axis. 
        img_y: [int] Size of image Y axis.
        scale: [float] Image scale in px/deg.
        multi_factor: [int] Number of meteor points per frame. 
        sigma_x: [float] Standard deviation along the X axis.
        sigma_y: [float] Standard deviation along the Y axis.
        noise scale [float] The standard deviation of a probability density function. 
        offset: [int or float] Offset of pixel levels for the entire image.
        show_plots: [bool] Argument for showing individual frame plots.

    Return:
        centroid_coordinates: [list of tuples] X and Y coordinates of meteor center calculated by centroiding.
        model_coordinates: [list of tuples] X and Y coordinates of meteor center calculated by model.

    """
    
    # Define multiplication factor and amplitude of gaussian
    framerate = (1/30)/multi_factor
    amplitude = 255/multi_factor

    # Array of points in time
    t_arr = np.arange(-t_meteor/2, t_meteor/2, framerate)


    # X and Y coordinates of image center
    x_center = img_x/2
    y_center = img_y/2

    # Lists of model and centroid coordinates
    centroid_coordinates = []
    model_coordinates = []


    for i in range(multi_factor):

        # Defining time limits
        t_start = -t_meteor/2 + i*multi_factor*framerate
        t_finish = -t_meteor/2 + (i + 1)*multi_factor*framerate

        # Array of points in time defined by framerate
        t_arr_iter = np.arange(t_start, t_finish, framerate)


        # Calculate beginning and ending points of meteor
        x_start, y_start = drawPoints(t_start, x_center, y_center, scale, phi, omega)
        x_finish, y_finish = drawPoints(t_finish, x_center, y_center, scale, phi, omega)

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


        # Adjusting crop borders
        x_start = int(round(x_start))
        x_finish = int(round(x_finish))
        y_start = int(round(y_start))
        y_finish = int(round(y_finish))

        # Two dimensional gaussian function crop window
        x_window = np.arange(x_start, x_finish)
        y_window = np.arange(y_start, y_finish)
        xx, yy = np.meshgrid(x_window, y_window)

        # Image array
        img_array = np.zeros((img_y, img_x), np.float_)

        # Draw two dimensional Gaussian function for each point in time
        for t in t_arr_iter:
            
            x, y = drawPoints(t, x_center, y_center, scale, phi, omega)
            temp = twoDimensionalGaussian(xx, yy, x, y, sigma_x, sigma_y, amplitude)

            img_array[y_start:y_finish, x_start:x_finish] += temp


        # Add Gaussian noise and offset
        if noise_scale > 0:
            gauss_noise = np.random.normal(loc = 0, scale = noise_scale, size = (img_y, img_x))
            img_array += abs(gauss_noise) + offset

        # Clip pixel levels
        np.clip(img_array, 0, 255)

        # Convert image to 8-bit unsigned integer
        img_array = img_array.astype(np.uint8)


        # Centroid coordinates
        t_mid = (t_start + t_finish)/2
        x_mid, y_mid = drawPoints(t_mid, x_center, y_center, scale, phi, omega)
        x_centr, y_centr = meteorCentroid(img_array, x_mid, y_mid, x_start, x_finish, y_start, y_finish)

        # Add centroid coordinates to list
        centroid_coordinates.append((x_centr, y_centr))

        # Model coordinates
        x_model, y_model = drawPoints(t_mid, x_center, y_center, scale, phi, omega)        

        # Add model coordinates to list
        model_coordinates.append((x_model, y_model))


        # Show frame
        if show_plots:
            plt.imshow(img_array, cmap = "gray", vmin = 0, vmax = 255)
            plt.scatter(x_centr, y_centr, marker = 'o')
            plt.scatter(x_model, y_model, marker = '*')
            plt.show()

        
    return (centroid_coordinates, model_coordinates)

def averageDifference(centroid_coordinates, model_coordinates):
    """
    Calculates average distance between centroid coordinates and model coordinates for each frame.
    
    Arguments:
        centroid_coordinates: [list or of tuples] List of (X, Y) centroid coordinates for each frame. 
        model_coordinates: [list or of tuples] List of (X, Y) model coordinates for each frame.

    Return:
        diff_avg: [float] Average distance of centroid and model points for a given set of frames. 
    """
    
    n = len(centroid_coordinates)
    diff_list = []

    for c in range(n):
        
        x_centr = centroid_coordinates[c][0]
        y_centr = centroid_coordinates[c][1]
        x_model = model_coordinates[c][0]
        y_model = model_coordinates[c][1]
        
        diff = np.sqrt((x_centr - x_model)**2 + (y_centr - y_model)**2)
        diff_list.append(diff)

    diff_avg = np.average(diff_list)

    return diff_avg


if __name__ == "__main__":

    ### Defining function parameters ###

    # Meteor duration
    t_meteor = 0.5

    # Meteor angle counterclockwise from the Y axis (deg)
    phi = 45

    # Meteor's angular velocity (deg/s)
    omega = 30

    # Angular velocity array
    omega_arr = np.arange(1, 50.5, 0.5)

    # Image size
    img_x = 720
    img_y = 576

    # Pixel scale in px/deg
    scale = img_x/64

    # Number of meteor points per frame
    multi_factor = 10

    # Standard deviation along X and Y axis
    sigma_x = 2
    sigma_y = 2

    # Scale of background noise
    noise_scale = 10

    # Scale of background noise array
    noise_scale_arr = [0, 5, 10, 20]

    # Level of background offset
    offset = 20

    # Plot individual frames?
    show_plots = False


    ### Average difference as a function of angular velocity ###

    # Array of average of averages
    noise_diff_arr = []

    # Number of runs
    n = 10

    for noise in noise_scale_arr:

        # Average of averages difference array
        diff_avg_avg = []

        print("Noise level: {}".format(noise))
    
        for omega_i in omega_arr:

            # Average differences array
            diff_avg = []

            for i in range(n):

                # Compute centroid and model coordinates
                centroid_coordinates, model_coordinates = pointsCentroidAndModel(t_meteor, phi, omega_i, img_x, img_y, scale, multi_factor, sigma_x, sigma_y, noise, offset, show_plots)
                
                # Compute average distance
                diff = averageDifference(centroid_coordinates, model_coordinates)
                
                print('{} Average difference: {:.4f}'.format(i, diff))
                diff_avg.append(diff)

            print('Angular velocity[deg/s]: {:.2f} Average of difference averages: {:.4f}'.format(omega_i, np.average(diff_avg)))
            diff_avg_avg.append(np.average(diff_avg))

        noise_diff_arr.append(diff_avg_avg)


    # Saving data
    np.savez('data.npz', omega_arr, *noise_diff_arr)