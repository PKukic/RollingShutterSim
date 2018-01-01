""" A set of functions used to perform the meteor simulation.
"""


# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

    # Distance from starting point
    delta_x = x - x0
    delta_y = y - y0

    # Calculate possibility
    p = amplitude * np.exp(- ((delta_x / (np.sqrt(2) * sigma_x)) ** 2 + (delta_y / (np.sqrt(2) * sigma_y)) ** 2))


    return p



def meteorCentroid(img, x_start, x_finish, y_start, y_finish):
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
        (x_centr, y_centr): [tuple of floats] X and Y coordinates of the calculated centroid
    """

    x0 = (x_start + x_finish)/2
    y0 = (y_start + y_finish)/2

    # Calculate length of meteor
    r = np.sqrt((x_finish - x_start)**2 + (x_finish - x_start)**2)

    # Crop image to centroid
    img_crop = img[y_start:y_finish, x_start:x_finish]

    # Init intensity sums
    sum_x = 0
    sum_y = 0
    sum_intens = 1e-10

    # Background noise value
    ny, nx = img.shape
    y, x = np.ogrid[:ny, :nx]
    img_mask = ((x - x0)**2 + (y - y0)**2 <  r**2)
    back_noise = np.ma.masked_array(img, mask = img_mask).mean()

    # Evaluate intensity sums
    for x in range(img_crop.shape[1]):
        for y in range(img_crop.shape[0]):

            value = img_crop[y, x] - back_noise

            sum_x += x*value
            sum_y += y*value

            sum_intens += value
    
    # print("\t\tBack noise: {}".format(back_noise))
    # print("\t\tIntensity sum: {}".format(sum_intens))
    # print("\t\tX sum: {}".format(sum_x))
    # print("\t\tY sum: {}".format(sum_y))
    
    # Calculate centroid coordinates
    x_centr = sum_x/sum_intens + x_start
    y_centr = sum_y/sum_intens + y_start


    return (x_centr, y_centr)



def calcSigmaWindowLimits(x_start, x_finish, sigma_x, y_start, y_finish, sigma_y, phi):
    """ Given the beginning and ending coordinate of the simulated meteor, get the window size which will 
        encompass the meteor with 3 sigma borders around edge points.

    Arguments:
        x_start: [int or float] X coordinate of beginning crop point.
        x_finish: [int or float] X coordinate of ending crop point.
        sigma_x: [int or float] Standard deviation along the X axis.
        y_start: [int or float] Y coordinate of beginning crop point.
        y_finish: [int or float] Y coordinate of ending crop point.
        sigma_y: [int or float] Standard deviation along the Y axis.
        phi: [int or float] Meteor angle, counterclockwise from Y axis.

    Return:
        (x_start, x_finish, y_start, y_finish): [tuple of ints] Image coords encompassing the meteor in the
            given time interval.

    """


    # Calculate beginning end ending points of crop, making sure 3 sigma parts of the meteor will be 
    # included
    if np.sin(np.radians(phi)) < 0:
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



    # Make sure beginnings are smaller then the endings
    x_start, x_finish = sorted([x_start, x_finish])
    y_start, y_finish = sorted([y_start, y_finish])


    # Crop borders to integer size
    x_start = int(round(x_start))
    x_finish = int(round(x_finish))
    y_start = int(round(y_start))
    y_finish = int(round(y_finish))


    return x_start, x_finish, y_start, y_finish




def pointsCentroidAndModel(rolling_shutter, t_meteor, phi, omega, img_x, img_y, scale, fps, sigma_x, \
    sigma_y, noise_scale, offset, show_plots):
    """
    Returns coordinates of meteor center calculated by centroiding and from a meteor movement model.

    Arguments:
        rolling_shutter: [bool] True if rolling shutter is used, False otherwise. 
        t_meteor: [int or float] Duration of meteor.
        phi: [int or float] Meteor angle, counterclockwise from Y axis.
        omega: [int or float] Meteor angular velocity in deg.
        img_x: [int] Size of image X axis. 
        img_y: [int] Size of image Y axis.
        scale: [float] Image scale in px/deg.
        fps: [int] Number of frames per second.
        sigma_x: [float] Standard deviation along the X axis.
        sigma_y: [float] Standard deviation along the Y axis.
        noise scale [float] The standard deviation of a probability density function. 
        offset: [int or float] Offset of pixel levels for the entire image.
        show_plots: [bool] Argument for showing individual frame plots.

    Return:
        centroid_coordinates: [list of tuples] X and Y coordinates of meteor center calculated by centroiding.
        model_coordinates: [list of tuples] X and Y coordinates of meteor center calculated by model.

    """

    # Calculate the amplitude and compensate for the movement loss
    # omegapxs = omega * scale
    amplitude = 255/img_y*(2*omega*scale)

    # Total number of frames of the duration of the meteor
    frame_number = int(round(t_meteor*fps))

    # X and Y coordinates of image center
    x_center = img_x/2
    y_center = img_y/2

    # Arrays of model and centroid coordinates
    centroid_coordinates = []
    model_coordinates = []

    # Rolling shutter parameters
    first_run = True
    read_x_encounter_prev = 0
    read_y_encounter_prev = 0


    # Go thorugh all frames
    for i in range(frame_number):

        # Define time limits
        t_start = -t_meteor/2 + i*(1/fps)
        t_finish = -t_meteor/2 + (i + 1)*(1/fps)

        # Last frame case
        #if i == (frame_number - 1):
           #t_finish = t_meteor/2

        # print("time limits: {:.4f} {:.4f}".format(t_start, t_finish))

        # Array of points in time defined by step and time limits
        t_arr_iter = np.linspace(t_start, t_finish, img_y)

        # Calculate beginning and ending points of meteor
        x_start, y_start = drawPoints(t_start, x_center, y_center, scale, phi, omega)
        x_finish, y_finish = drawPoints(t_finish, x_center, y_center, scale, phi, omega)


        # Add 3 sigma border around them
        x_start, x_finish, y_start, y_finish = calcSigmaWindowLimits(x_start, x_finish, sigma_x, y_start, \
            y_finish, sigma_y, phi)

        [x_start, x_finish] = np.clip([x_start, x_finish], 0, img_x)
        [y_start, y_finish] = np.clip([y_start, y_finish], 0, img_y)

        # print(x_start, x_finish)
        # print(y_start, y_finish)

        # Make 2D Gaussian function crop window
        x_window = np.arange(x_start, x_finish)
        y_window = np.arange(y_start, y_finish)
        xx, yy = np.meshgrid(x_window, y_window)

        # Plot gauss window
        # plt.gca().add_patch(patches.Rectangle((x_start, y_start), x_finish - x_start, \
            # y_finish - y_start, fill=False, color='r'))



        # Init the sensor array only during the first run of the rolling shutter
        if (not rolling_shutter) or (rolling_shutter and first_run):
            sensor_array = np.zeros(shape=(img_y, img_x), dtype=np.float_)

        # Init line counter and encounter checker
        if rolling_shutter:
            line_counter = 0
            reader_first_encounter = True



        # Read image (output image) array
        read_image_array = np.zeros(shape=(img_y, img_x), dtype=np.float_)


        # Draw two dimensional Gaussian function for each point in time
        for t in t_arr_iter:

            # Evaluate meteor point and 2D Gaussian
            x, y = drawPoints(t, x_center, y_center, scale, phi, omega)
            temp = twoDimensionalGaussian(xx, yy, x, y, sigma_x, sigma_y, amplitude)

            # print("\t\t Y start: {:.2f}; Y finish {:.2f}; X start: {:.2f}; X finish: {:.2f}".format(y_start, y_finish, x_start, y_finish))
            # print("\t\t", abs(y_start - y_finish), abs(x_start - x_finish))
            
            sensor_array[y_start:y_finish, x_start:x_finish] += temp

            # Rolling shutter part 
            if rolling_shutter:
                
                # Read a line from the sensor array, add it to the read image array and
                # set the same line in the sensor array to 0
                read_image_array[line_counter] = sensor_array[line_counter]
                sensor_array[line_counter] = np.zeros(shape=img_x, dtype=np.float_)

                # Checking time
                # print("{:.5f} {}".format(t, line_counter))

                line_counter += 1

                # Check if the reader has encountered the position of the meteor
                if reader_first_encounter:
                    if line_counter > y:
                        read_x_encounter = int(round(x))
                        read_y_encounter = int(round(y))
                        
                        reader_first_encounter = False

                
                # plt.imshow(sensor_array)
                # plt.show()
                # plt.imshow(read_image_array)
                # plt.show()


        
        if rolling_shutter:

            # If this was the first run of the rolling shutter, repeat it
            if first_run and (i == 0):
                first_run = False
                read_y_encounter_prev = read_y_encounter
                read_x_encounter_prev = read_x_encounter
                continue
            
            # Otherwise determine the crop window
            else:
                x_start = read_x_encounter_prev
                x_finish = read_x_encounter

                y_start = read_y_encounter_prev
                y_finish = read_y_encounter

                # Add 3 sigma borders around crop points
                x_start, x_finish, y_start, y_finish = calcSigmaWindowLimits(x_start, x_finish, sigma_x, \
                y_start, y_finish, sigma_y, phi)

        else:
            read_image_array = sensor_array

        # Rescale image
        read_image_array /= np.amax(read_image_array)
        read_image_array *= 255

        # Add Gaussian noise and offset
        if noise_scale > 0:
            gauss_noise = np.random.normal(loc=0, scale=noise_scale, size=(img_y, img_x))
            read_image_array += abs(gauss_noise) + offset

        # Clip pixel levels
        read_image_array = np.clip(read_image_array, 0, 255)

        # Convert image to 8-bit unsigned integer
        read_image_array = read_image_array.astype(np.uint8)

        # Centroid coordinates
        x_centr, y_centr = meteorCentroid(read_image_array, x_start, x_finish, y_start, y_finish)
        
        # Model coordinates
        t_mid = (t_start + t_finish)/2
        x_model, y_model = drawPoints(t_mid, x_center, y_center, scale, phi, omega) 
        

        # Check if the meteor is located outside of the image
        # if x_model >= 0 and x_model <= img_x and y_model >= 0 and y_model <= img_y:
        
        centroid_coordinates.append((x_centr, y_centr))
        model_coordinates.append((x_model, y_model))


        # Keep track where the reader enountered the meteor at the previous frame
        if rolling_shutter:
            read_y_encounter_prev = read_y_encounter
            read_x_encounter_prev = read_x_encounter


        
        if show_plots:
            
            # Show frame
            plt.imshow(read_image_array, cmap='gray', vmin=0, vmax=255)
            
            # Plot crop window
            plt.gca().add_patch(patches.Rectangle((x_start, y_start), x_finish - x_start, \
                y_finish - y_start, fill=False, color='w'))

            # Plot centroid
            plt.scatter(x_centr, y_centr, c='red', marker='+')

            # Plot model centre
            plt.scatter(x_model, y_model, c='blue', marker='+')
            plt.show()

        
    return (centroid_coordinates, model_coordinates)


def centroidDifference(centroid_coordinates, model_coordinates):
    """Calculates difference between coordinates of centroid and model point.

    Arguments:
        centroid coordinates: [tuple of floats or ints] Tuple that consists of X and Y centroid coordinates.
        model coordinates: [tuple of floats or ints] Tuple that consists of X and Y model point coordinates.

    Return:
        diff: [float] Difference (length) between the centroid and the model point. 
    """

    # Model and centroid points coordinates
    x_centr = centroid_coordinates[0]
    y_centr = centroid_coordinates[1]

    x_model = model_coordinates[0]
    y_model = model_coordinates[1]

    # Calculate difference (length) between points
    return np.sqrt((x_centr - x_model)**2 + (y_centr - y_model)**2)


def centroidAverageDifference(centroid_coordinates, model_coordinates):
    """
    Calculates average distance between centroid coordinates and model coordinates for each frame.
    
    Arguments:
        centroid_coordinates: [list or of tuples] List of (X, Y) centroid coordinates for each frame. 
        model_coordinates: [list or of tuples] List of (X, Y) model coordinates for each frame.

    Return:
        diff_avg: [float] Average distance of centroid and model points for a given set of frames. 
    """

    # Difference array
    diff_arr = []
    
    # Get length of coordinate array (= frame number)
    coord_arr_len = len(centroid_coordinates)


    # Calculate difference for every couple of model and centroid coordinates
    for c_num in range(coord_arr_len):
        diff = centroidDifference(centroid_coordinates[c_num], model_coordinates[c_num])
        diff_arr.append(diff)

    # Calculate average of differences
    diff_avg = np.average(diff_arr)

    return diff_avg