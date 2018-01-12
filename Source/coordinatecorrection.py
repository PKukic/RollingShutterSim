def coordinateCorrection(time_coordinates, centroid_coordinates, img_y, fps):
    ''' Corrects the centroid coordinates of a given meteor.

        Arguments:
            t_meteor: [array of ints] Duration of the meteor [s].
            centroid_coordinates_raw: [array of tuples of floats] Uncorrected meteor coordinates. 
            img_y:  [int] Size of image Y coordinate [px].
            fps: [int] Number of frames taken per second by the camera.

        Return:
            centroid_coordinates_corr: [array of tuples of floats] Corrected meteor coordinates. 
    '''

    num_coord = len(centroid_coordinates)

    ### Calculate meteor angle from linear fit ###

    # Extract data from existing coordinate array
    x_coordinates = []
    y_coordinates = []

    for i in range(num_coord):
        x_coordinates.append(centroid_coordinates[i][0])
        y_coordinates.append(centroid_coordinates[i][1])


    # Linear fit function
    def linFit(x, a, b):
        return a * x + b

    # Fit to find slope
    param, pcov = opt.curve_fit(linFit, x_coordinates, y_coordinates)
    a = param[0]
    
    # Check in which direction the meteor is going
    dx = x_coordinates[num_coord - 1] - x_coordinates[0]

    # Calculate meteor angle
    if a < 0:
        phi = -np.arctan(1 / a)
    else:
        phi = np.arctan(a) + np.pi/2
    
    if dx >= 0:
        phi += np.pi

    print('Calculated angle: {:.2f}'.format(np.rad2deg(phi)))

    print('Meteor slope fit finished.')

    ### Calculate point to point velocity ###

    # Define starting time and displacement coordinates
    # t_start = time_coordinates[0]
    # coord_start = centroid_coordinates[0]

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

    # print(len(t_arr), len(r_arr), len(v_arr))

    # print(v_arr)

    print('Point to point velocity calculation done.')

    ### Smooth out velocity ###
    
    for i in range(num_coord - 3):
        v_arr[i + 1] = (v_arr[i] + v_arr[i + 2]) / 2

    print('Velocity smoothened out.')
    
    ### Apply correction to velocity array ###

    for i in range(num_coord - 1):
        v_arr[i] += velocityCorrection(v_arr[i], phi, img_y, fps)

    print('Calculated average velocity: {:.2f}'.format(np.average(v_arr)))

    print('Velocity correction done. ')

    ### Apply correction to centroid coordinate array ###

    # List of corrected coordinates
    centroid_coordinates_corr = []

    # List of correction values
    corr_arr = []

    for i in range(num_coord - 1):

        # Centroid coordinates
        x_centr = x_coordinates[i + 1]
        y_centr = y_coordinates[i + 1]

        # Centroid velocity
        omega_pxs = v_arr[i]

        # Calculate the correction for the given set of parameters
        corr = calculateCorrection(y_centr, img_y, omega_pxs, fps)
        corr_arr.append(corr)

        #print(corr)

        # Correct the coordinates
        x_corr = x_centr + np.sin(phi) * corr
        y_corr = y_centr - np.cos(phi) * corr

        # print('Corrected coordinates: ({:.2f}, {:.2f})'.format(x_corr, y_corr))

        centroid_coordinates_corr.append((x_corr, y_corr))


    print('Correction factor for each centroid point (calculated):')
    for i in range(len(corr_arr)):
        print(corr_arr[i]) 

    print('Centroid coordinates corrected.')

    # Return list of corrected coordinates
    return centroid_coordinates_corr
