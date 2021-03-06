""" A set of functions used to perform the meteor simulation.
"""


# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.special as sp

def ConvToSim(conv_phi):

    sim_phi = (conv_phi + 270) % 360

    return sim_phi

def SimToConv(sim_phi):

    conv_phi = (sim_phi - 270) % 360

    return conv_phi

def drawPoints(t, x_center, y_center, scale, phi, omega, fit_param, t_meteor):
    """ Calculate position of a meteor on image with the given simulation parameters. 

    Arguments:
        t: [float or ndarray] Time in seconds.
        x_center: [int] Image X center.
        y_center: [int] Image Y center.
        scale: [float] Image scale in px/deg.
        phi: [int or float] Meteor angle, counterclockwise from Y axis.
        omega: [int or float] Meteor angular velocity in deg/s.
        fit_param: [array of floats] Parameters of the exponentional meteor deceleration function.
        t_meteor: [int or float] Duration of meteor.

    Return:
        (x_meteor, y_meteor): [tuple of floats] X and Y coordinate of the meteor.
    """

    # Unpack variables from tuple
    a = fit_param[0]
    b = fit_param[1]

    # Convert angle to radians
    phi = np.radians(phi)

    # print(t)
    # print(t+t_meteor/2)
    # print(omega*t*scale)
    # print(- a*np.exp(b*(t+t_meteor/2))*scale)

    omega_dec = omega - a*b*np.exp(b*(t+t_meteor/2))
    # print(omega_dec)

    # Calculate distance from centre in pixels
    z = omega_dec*t*scale

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
    sigma_y, noise_scale, offset, fit_param, show_plots, corr_coord=None):
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
        coor_coord: [list of tuples] List of the corrected rolling shutter coordinates (used for plotting, None by default.)

    Return:
        time_coordinates: [list] The time coordinates of both the rolling shutter and model centroids
        centroid_coordinates: [list of tuples] X and Y coordinates of meteor center calculated by centroiding.
        model_coordinates: [list of tuples] X and Y coordinates of meteor center calculated by model.

    """

    # Calculate the amplitude and compensate for the movement loss
    amplitude = 255

    # Total number of frames of the duration of the meteor
    frame_number = int(round(t_meteor*fps))

    # X and Y coordinates of image center
    x_center = img_x/2
    y_center = img_y/2

    # Arrays of model and centroid coordinates
    centroid_coordinates = []
    model_coordinates = []

    # Array of time points
    time_coordinates = []

    # Rolling shutter parameters
    first_run = True
    read_x_encounter_prev = 0
    read_y_encounter_prev = 0


    # Go thorugh all frames
    for i in range(frame_number):

        if corr_coord != None:
            corrx, corry = corr_coord[i]

        # Define time limits
        t_start = -t_meteor/2 + i*(1/fps)

        t_finish = -t_meteor/2 + (i + 1)*(1/fps)

        # print("time limits: {:.4f} {:.4f}".format(t_start, t_finish))

        # Array of points in time defined by step and time limits
        t_arr_iter = np.linspace(t_start, t_finish, img_y)

        # Calculate beginning and ending points of meteor
        x_start, y_start = drawPoints(t_start, x_center, y_center, scale, phi, omega, fit_param, t_meteor)
        x_finish, y_finish = drawPoints(t_finish, x_center, y_center, scale, phi, omega, fit_param, t_meteor)

        # print('-'*20)

        # print(t_start, t_finish)
        # print(t_start + t_meteor/2, t_finish + t_meteor/2)
        # print(omega*t_start*scale, omega*t_finish*scale)
        # print(- fit_param[0]*np.exp(fit_param[1]*(t_start+t_meteor/2))*scale, - fit_param[0]*np.exp(fit_param[1]*(t_finish+t_meteor/2))*scale)

        # print(x_start - x_finish)
        # print(y_start - y_finish)

        # print('-'*20)

        # Add 3 sigma border around them
        x_start, x_finish, y_start, y_finish = calcSigmaWindowLimits(x_start, x_finish, sigma_x, y_start, \
            y_finish, sigma_y, phi)

        if set([x_start, x_finish]) != set(np.clip([x_start, x_finish], 0, img_x)):
            return (-1, -1, -1)
        
        elif set([y_start, y_finish]) != set(np.clip([y_start, y_finish], 0, img_y)):
            return (-1, -1, -1)

        # print(x_start, x_finish)
        # print(y_start, y_finish)

        # Make 2D Gaussian function crop window
        x_window = np.arange(x_start, x_finish)
        y_window = np.arange(y_start, y_finish)
        xx, yy = np.meshgrid(x_window, y_window)

        # Plot gauss window
        # plt.gca().add_patch(patches.Rectangle((x_start, y_start), x_finish - x_start, \
            # y_finish - y_start, fill=False, color='r'))

        # print(x_start, y_start, x_finish, y_finish)


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

            # print(t)

            # Evaluate meteor point and 2D Gaussian
            x, y = drawPoints(t, x_center, y_center, scale, phi, omega, fit_param, t_meteor)
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
        t_mid = (t_start + t_finish) / 2
        x_model, y_model = drawPoints(t_mid, x_center, y_center, scale, phi, omega, fit_param, t_meteor) 
        
        
        # Append model and centroid coordinates to list
        centroid_coordinates.append((x_centr, y_centr))
        model_coordinates.append((x_model, y_model))

        # Append average time to list
        time_coordinates.append(t_mid)

        # Keep track where the reader enountered the meteor at the previous frame
        if rolling_shutter:
            read_y_encounter_prev = read_y_encounter
            read_x_encounter_prev = read_x_encounter


        
        if show_plots:
            
            # Show frame
            plt.imshow(read_image_array, cmap='gray', vmin=0, vmax=255)

            plt.title('Frame number: {}'.format(i))
            
            # print(x_start, y_start, x_finish, y_finish)

            # plt.xkcd()

            # t_finish -= 0.5*1/fps
            t_mid = t_start
            x_model, y_model = drawPoints(t_mid, x_center, y_center, scale, phi, omega, fit_param, t_meteor) 

            # Plot crop window
            plt.gca().add_patch(patches.Rectangle((x_start, y_start), x_finish - x_start, \
                y_finish - y_start, fill=False, color='w'))

            # Plot centroid
            plt.scatter(x_centr, y_centr, c='red', marker='o', label = 'centroid', s = 70)

            # Plot model centre
            plt.scatter(x_model, y_model, c='blue', marker='^', label = 'model', s = 70)

            if corr_coord != None:
                plt.scatter(corrx, corry, c = 'green', marker = 'o', label = 'corrected', s = 70)

            plt.axhline(y=y_model, c='b')

            # (x_st, x_fin, y_st, y_fin) = calcSigmaWindowLimits(x_model, x_model, sigma_x*10, y_model, y_model, sigma_y*10, phi)
            # plt.xlim([x_st, x_fin])
            # plt.ylim([y_fin, y_st])

            plt.legend(loc='upper right')

            plt.savefig('../Figures/Images/centroids/{}i.png'.format(i))

            plt.show()

        
    return (time_coordinates, centroid_coordinates, model_coordinates)


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
    """ Calculates average distance between centroid coordinates and model coordinates for each frame.
    
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


def timeFromAngle(phi, omega, img_x, img_y, scale, fps):
    """ Calculates the time it takes for the meteor to cross the whole image based on the meteor angle. 

    Arguments:
        phi: [int or float] Meteor angle, counterclockwise from Y axis.
        img_x: [int] Size of image X axis. 
        img_y: [int] Size of image Y axis.

    Return:
        t_meteor: [int or float] Duration of meteor.
    """
    
    # Convert angle measure to a value between 0 and 180
    if phi >= 180:
        phi -= 180

    # Convert to radians
    phi = np.radians(phi)
    diag_angle = np.arctan2(img_x, img_y)

    # print("Phi: {:.2f} Diag angle: {:.2f}".format(np.rad2deg(phi), np.rad2deg(diag_angle)))
    
    # Image center coordinates (starting coordinates)
    center_coordinates = (img_x / 2, img_y / 2)

    # If the angle is 0 or 180 deg, the meteor is vertical
    if phi == 0 or phi == np.pi:

        # Form end point
        end_coordinates = (center_coordinates[0], img_y)

    # If the angle is 90 deg, the meteor is horizontal
    elif phi == np.pi/2.:

        end_coordinates = (img_x, center_coordinates[1])

    else:

        if phi <= diag_angle or phi >= 2 * diag_angle:
            
            # Calculate final X coordinate of meteor
            x_end = np.tan(phi) * img_y / 2 + center_coordinates[0]
            end_coordinates = (x_end, img_y)

            # print("X coordinate: {:.2f}".format(x_end))

        else:

            # Calculate final Y coordinate of meteor
            y_end = img_x / (2 * np.tan(phi)) + center_coordinates[1]
            end_coordinates = (img_x, y_end)

            # print("Y coordinate: {:.2f}".format(y_end))

    # Clip end coordinates to image size
    clipped_x = np.clip(end_coordinates[0], 0, img_x)
    clipped_y = np.clip(end_coordinates[1], 0, img_y)
    end_coordinates = (clipped_x, clipped_y)

    # print("End coordinates: ({:.2f}, {:.2f})".format(end_coordinates[0], end_coordinates[1]))

    # Calculate difference between the starting and ending points
    r = centroidDifference(center_coordinates, end_coordinates)

    # Convert omega [deg/s] to [px/s]
    omega_pxs = omega * scale

    # Calculate time from distance and velocity
    t_meteor = r / omega_pxs

    # Cut number to a multiple of image exposition
    t_meteor -= t_meteor % (1 / fps)
    
    return t_meteor

def calculateCorrection(ycentr, img_y, omega_pxs, fps, f = 1):
    ''' Calculates the correction factor for the rolling shutter centroid
        coordinates.

        Arguments:
            ycentr: [int or float] Y coordinate of the meteor centroid.
            img_y: [int] Size of image Y coordinate [px].
            omega_pxs: [int or float] Angular velocity of the meteor [px/s].
            fps: [int] Number of frames taken per second by the camera.
            f: [float] The ratio between the exposure time and frame time.

        Return:
            corr: [float] Correction distance [px].
    '''

    return (f - ycentr / img_y) * (omega_pxs / fps)

def velocityCorrection(omega_roll, phi, img_y, fps):
    ''' Corrects the velocity of a given point.

        Arguments:
            omega_roll: [int or float] Meteor angular velocity [px/s].
            phi: [int or float] Meteor angle on the image.
            img_y: [int] Image Y coordinate size
            fps: [int] Number of frames per second taken by the camera.

        Returns:
            v_corr: [float] Corrected velocity of the point.
    '''

    # Define parameters used for correcting the velocity
    omega_o = img_y * fps

    # Correct velocity using found model
    v_corr = - (np.cos(phi)*omega_roll**2) / (omega_o+np.cos(phi)*omega_roll)

    return v_corr


def getVelocity(time_coordinates, centroid_coordinates):
    ''' Calculates the instanteanous velocities for a given set of temporal and spatial coordinates. 

        Arguments:
            time_coordinates: [array of floats] Temporal coordinates. (s)
            centroid_coordinates: [array of floats] Spatial coordinates (centroid coordinates). (px)

        Returns:
            dist_arr: [array of floats] Array of distances from the starting centroid. (px)
            v_arr: [array of floats] Array of instanteanous velocities. (px/s)

    '''

    num_coord = len(centroid_coordinates)

    r_arr = []
    t_arr = []
    v_arr = []

    dist_arr = []
    start_coord = centroid_coordinates[0]

    # Construct the instantaneous velocity array
    for i in range(num_coord - 1):
        r = centroidDifference(centroid_coordinates[i + 1], centroid_coordinates[i])
        r_arr.append(r)

        t = time_coordinates[i + 1] - time_coordinates[i]
        t_arr.append(t)


    for i in range(num_coord - 1):
        v_arr.append(r_arr[i] / t_arr[i])
    
    print(r_arr)
    print(t_arr)
    print(v_arr)

    # Assume that the initial velocity is equal to the subsequent velocity
    v_arr.insert(0, v_arr[0])

    # Construct the distances array
    for i in range(num_coord):
        ind_coord = centroid_coordinates[i]
        dist_arr.append(centroidDifference(start_coord, ind_coord))

    return dist_arr, v_arr
    

def coordinateCorrection(time_coordinates, centroid_coordinates, img_y, fps, version):
    ''' Corrects the centroid coordinates of a given meteor.

        Arguments:
            t_meteor: [array of ints] Duration of the meteor [s].
            centroid_coordinates_raw: [array of tuples of floats] Uncorrected meteor coordinates. 
            img_y:  [int] Size of image Y coordinate [px].
            fps: [int] Number of frames taken per second by the camera.
            version: [string] The version of the function to be implemented (with or without velocity correction):
                possible values: 'v' or 'v_corr'

        Return:
            centroid_coordinates_corr: [array of tuples of floats] Corrected meteor coordinates. 
    '''

    num_coord = len(centroid_coordinates)

    ### Calculate meteor angle from linear fit ###

    # Extract data from the existing coordinate array
    x_coordinates = []
    y_coordinates = []
    r_coordinates = []


    for i in range(num_coord):
        x_coordinates.append(centroid_coordinates[i][0])
        y_coordinates.append(centroid_coordinates[i][1])

    start_coord = (x_coordinates[0], y_coordinates[0])

    for i in range(num_coord):
        ind_coord = (x_coordinates[i], y_coordinates[i])
        r = centroidDifference(start_coord, ind_coord)
        r_coordinates.append(r)

    # Find meteor angle
    delta_r = r_coordinates[num_coord-1] - r_coordinates[0]
    delta_x = x_coordinates[num_coord-1] - x_coordinates[0]
    delta_y = y_coordinates[num_coord-1] - y_coordinates[0]

    if delta_x <= 0:
        phi = np.arccos(delta_y/delta_r)
    else:
        phi = -np.arccos(delta_y/delta_r)

    print('Meteor angle:', phi)

    if phi < 0:
        phi += 2*np.pi


    print('Meteor slope fit finished.')
    print('Calculated angle: {:.2f}'.format(np.rad2deg(phi)))

    ### Calculate point to point velocity ###

    # Arrays of time and position displacement
    r_arr = []
    t_arr = []

    # Velocity array
    v_arr = []

    # Form delta distance, time and velocity arrays
    for i in range(num_coord - 1):
        t_arr.append(time_coordinates[i + 1] - time_coordinates[i])
        r_arr.append(centroidDifference(centroid_coordinates[i + 1], centroid_coordinates[i]))
    
    for i in range(num_coord - 1):
        v_arr.append(r_arr[i] / t_arr[i])

    # Assume that the first velocity is equal to the subsequent velocity
    v_arr.insert(0, v_arr[0])

    print('Average velocity without smoothening:', np.average(v_arr))

    print('Point to point velocity calculation done.')

    ### Smooth out velocity ###
    
    for i in range(num_coord - 2):
        v_arr[i + 1] = (v_arr[i] + v_arr[i + 2]) / 2

    print('Average velocity after smoothening:', np.average(v_arr))

    print('Velocity smoothened out.')
    
    ### Apply correction to velocity array ###

    if version == 'v_corr':

        for i in range(num_coord):
            v_arr[i] += velocityCorrection(v_arr[i], phi, img_y, fps)
            # print('velocity, correction, product')
            # print('{:.2f} {:.2f} {:.2f}'.format(v_arr[i], velocityCorrection(v_arr[i], phi, img_y, fps), v_arr[i] + velocityCorrection(v_arr[i], phi, img_y, fps)))
            

        print('Velocity correction done. ')

    elif version == 'v':

        print('Skipping velocity correction')

    else:
        
        print('Invalid version flag!')

    
    print('Calculated average velocity: {:.2f}'.format(np.average(v_arr)))

    ### Apply correction to centroid coordinate array ###

    # List of corrected coordinates
    centroid_coordinates_corr = []

    for i in range(num_coord):

        # Centroid coordinates
        x_centr = x_coordinates[i]
        y_centr = y_coordinates[i]

        # Centroid velocity
        omega_pxs = v_arr[i]

        # Calculate the correction for the given set of parameters
        corr = calculateCorrection(y_centr, img_y, omega_pxs, fps)

        # Correct the coordinates
        x_corr = x_centr - np.sin(phi) * corr
        y_corr = y_centr + np.cos(phi) * corr

        centroid_coordinates_corr.append((x_corr, y_corr))

    print('Centroid coordinates corrected.')

    # Return list of corrected coordinates
    return centroid_coordinates_corr


def timeCorrection(centroid_coordinates, img_y, fps, t_meteor, time_mark, t_init=None):
    ''' Corrects the time coordinates of a given meteor, changes the time assignment for each frame.

        Arguments:
            centroid_coordinates: [array of floats] Centroid coordinates of the meteor.
            img_y: [int] Y axis image size.
            fps: [int] Number of frames per second captured by the camera.
            t_meteor: [int or float] Duration of meteor.
            time_mark: [string] Indicates the position of the time mark for each frame. 'start' if the time mark is
                at the start of the frame, 'end' if it is on the end of the frame.

        return:
            time_coordinates_corr: [array of floats] Corrected time coordinates of the meteor.
    '''

    num_coord = len(centroid_coordinates)

    # Initialize array of corrected time coordinates
    time_coordinates_corr = []


    for i in range(num_coord):

        # Define starting time for each frame
        if t_init != None:
            t_start = t_init + i * (1/fps)
        else:
            t_start = -t_meteor/2 + i * (1/fps)

        # Set time offset for different frame time marks
        if time_mark == 'beginning':
            t_start += 0.5 * (1/fps)

        elif time_mark == 'end':
            t_start -= 0.5 * (1/fps)

        elif time_mark == 'middle':
            t_start += 0

        else:
            print('Invalid time mark flag!')

        # Row of the measurement (Y centroid coordinate)
        y_centr = centroid_coordinates[i][1]

        # Calculate the time assignment change
        delta_t = y_centr * (1/fps) / img_y 
        t_start += delta_t

        # print(delta_t)

        time_coordinates_corr.append(t_start)

    print('Time coordinate correction done.')

    return time_coordinates_corr


def getModelfromTime(time_coordinates_corr, img_x, img_y, scale, phi, omega, fit_param, t_meteor):
    ''' Finds the model coordinates equivalent to the temporally corrected time coordinates. 

        Arguments:
            time_coordinates_corr: [array of floats] Corrected time coordinates of the meteor.
            img_x: [int] Size of image X axis. 
            img_y: [int] Size of image Y axis.
            scale: [float] Image scale in px/deg.
            phi: [int or float] Meteor angle, counterclockwise from Y axis.
            omega: [int or float] Meteor angular velocity in deg.
            fit_param: [array of floats] Parameters of the exponentional meteor deceleration function.
            t_meteor: [int or float] Duration of meteor.

        Returns:
            model_coordinates_temp: [array of tuples of floats] The computed model coordinates. (px, px)
    '''

    model_coordinates_temp = []

    for i in range(len(time_coordinates_corr)):

        x_model, y_model = drawPoints(time_coordinates_corr[i], img_x/2, img_y/2, scale, phi, omega, fit_param, t_meteor)
        model_coordinates_temp.append((x_model, y_model))

    return model_coordinates_temp


def meteorCorrection(time_coordinates, centroid_coordinates, img_y, fps, correction_type, frame_timestamp):

    # Define number of coordinates
    num_coord = len(time_coordinates)

    # Define meteor duration
    t_meteor = abs(time_coordinates[num_coord - 1] - time_coordinates[0])
    t_meteor += abs(time_coordinates[0] - time_coordinates[1]) + abs(time_coordinates[num_coord - 1] - time_coordinates[num_coord - 2])

    print('Calculated meteor duration: {:.2f}'.format(t_meteor))

    # Correct temporal or spatial coordinates
    if correction_type == 'temporal':

        time_coordinates_corr = timeCorrection(centroid_coordinates, img_y, fps, t_meteor, frame_timestamp)
        # print(len(time_coordinates_corr))
        
        return time_coordinates_corr
    
    elif correction_type == 'spatial':
    
        centroid_coordinates_corr = coordinateCorrection(time_coordinates, centroid_coordinates, img_y, fps)
        # print(len(centroid_coordinates_corr))
    
        return centroid_coordinates_corr


def getparam(a, v_start, v_finish, t):
    ''' Calculates the second parameter of the exponentional meteor deceleration function. 
    Arugments:
        a: [float or int] First parameter of the exponentional function. 
        v_start: [float or int] Initial velocity of the meteor [deg/s]. 
        v_finish: [float or int] Ending velocity of the meteor [deg/s].
        t: [float or int] Meteor duration [s].

    Return:
        b: [float] The second parameter of the exponentional function. 
    '''

    b = sp.lambertw((v_start - v_finish) / a * t).real / t
    
    return b