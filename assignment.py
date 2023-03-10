import glm
import numpy as np
import cv2 as cv
import pickle

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            # The color is appended to obtain the colors in the 3D reconstruction
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])

    return data, colors


def get_parameters(path):
    """
    gets extrinsic and intrinsic parameters from xml file
    :param path: path where the config.xml is located
    :return: rotation, translation, and distortion vector, and K matrix.
    """

    filename = path + '/config.xml'
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)

    mtx = fs.getNode('mtx').mat()
    rvecs = fs.getNode('rvecs').mat()
    tvecs = fs.getNode('tvecs').mat()
    dist = fs.getNode('dist').mat()

    fs.release()
    return rvecs, tvecs, mtx, dist


def set_voxel_positions(frames, nframe):
    """    Iterates through all the voxels in our 3D space and activates them or not
    depending on whether is 255 or 0 in the binary 2D masks. It creates the
    look-up table.
    :param nframe: the frame in which we are
    :return: data and color of all the activated voxels    """
    width = int(60)
    depth = int(60)
    vid, mask, rvecs, tvecs, mtx, dist, data_mesh = [], [], [], [], [], [], []
    data, colors, tables = [], [], []
    voxels = []
    # Iterates through all  the cams
    for cam in range(1, 5):
        # Load look up table
        with open(f"data/cam{cam}/look_up_table.pickle", "rb") as f:
            # Use the pickle.load() function to deserialize the dictionary from the file
            table = pickle.load(f)
            v = list(table.keys())
            voxels.append(v)
            tables.append(table)

        frame = frames[cam - 1][nframe]
        # Change it to gray and append it to mask
        m = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask.append(m)  # Get the colored mask

        path3 = f'data/cam{cam}/video.png'
        color_frame = cv.imread(path3.format(2))
        vid.append(color_frame)
    count = 0
    for n in range(len(tables[0])):
        voxel = True
        color = []
        for c in range(0, 4):
            pixel_pts = tables[c][voxels[c][n]]
            mask0 = mask[c]
            (heightMask, widthMask) = mask0.shape
            # Avoid pixels outside the boundaries of the 2D images
            if 0 <= pixel_pts[0] < heightMask and 0 <= pixel_pts[1] < widthMask:
                val = mask0[int(pixel_pts[0]), int(pixel_pts[1])]
                color.append(vid[c][int(pixel_pts[0])][int(pixel_pts[1])])
                # If the value is zero in any of the cameras it is not considered
                # as activated in the 3D space.
                if val == 0:
                    voxel = False

        if voxel:
            data.append([(voxels[0][count][0] * block_size / 60 - width) + 40,
                         (voxels[0][count][2] * block_size) / 60,
                         (voxels[0][count][1] * block_size / 60 - depth) + 40])
            # Used for the color reconstruction
            final_color = np.mean(np.array(color), axis=0) / 256
            colors.append(final_color)

        count += 1

    return data, colors


"""
def set_voxel_positions(frames, nframe):

    vid, mask, rvecs, tvecs, mtx, dist, data_mesh = [], [], [], [], [], [], []
    data, colors, tables = [], [], []
    voxels = []
    # Iterates through all  the cams
    for cam in range(1, 5):
        # Load look up table
        with open(f"data/cam{cam}/look_up_table.pickle", "rb") as f:
            # Use the pickle.load() function to deserialize the dictionary from the file
            table = pickle.load(f)
            # voxel = list(table.items())
            # voxels.append(voxel)
            tables.append(table)
        frame = frames[cam-1][nframe]

        # Change it to gray and append it to mask
        m = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask.append(m)

        # Get the colored mask
        path3 = f'data/cam{cam}/video.png'
        color_frame = cv.imread(path3.format(2))
        vid.append(color_frame)

    # Parameters for the 3D space definition
    width = int(60)
    height = int(60)
    depth = int(60)

    # Iterates through all the voxels in our 3D space
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                color = []
                voxel = True
                # Iterates through all cameras
                for c in range(0, 4):

                    # Creates the voxel that we are paying attention to. Multiplied by 40 to enhance visualization.
                    voxels = ((x - width / 2) * 40, (y - height / 2) * 40, -z * 40)

                    pixel_pts = tables[c][voxels]
                    mask0 = mask[c]

                    (heightMask, widthMask) = mask0.shape

                    # Avoid pixels outside the boundaries of the 2D images
                    if 0 <= pixel_pts[0] < heightMask and 0 <= pixel_pts[1] < widthMask:
                        val = mask0[int(pixel_pts[0]), int(pixel_pts[1])]
                        color.append(vid[c][int(pixel_pts[0])][int(pixel_pts[1])])

                        # If the value is zero in any of the cameras it is not considered
                        # as activated in the 3D space.
                        if val == 0:
                            voxel = False

                if voxel:

                    # Summed 40 to compensate the 40 factor multiplication before and maintain the subject on its
                    # initial position
                    data.append(
                        [(x * block_size/2 - width)+40, (z * block_size)/2,
                         (y * block_size/2 - depth)+40])

                    # Used for the color reconstruction
                    final_color = np.mean(np.array(color), axis=0) / 256
                    colors.append(final_color)

    return data, colors
"""


def get_cam_positions():
    """
    Generates dummy camera locations at the 4 corners of the room
    :return: The camera positions
    """

    cameraposition = np.zeros((4, 3, 1))

    for c in range(0, 4):
        cam = c + 1
        path = 'data/cam' + str(cam)

        rvecs, tvecs, mtx, dist = get_parameters(path)

        rmtx, _ = cv.Rodrigues(rvecs)

        # The camera positions are calculated from the extrinsic parameters.
        cameraposition[c] = (-np.dot(np.transpose(rmtx), tvecs / 115))

    cameraposition2 = [[cameraposition[0][0][0], -cameraposition[0][2][0], cameraposition[0][1][0]],
                       [cameraposition[1][0][0], -cameraposition[1][2][0], cameraposition[1][1][0]],
                       [cameraposition[2][0][0], -cameraposition[2][2][0], cameraposition[2][1][0]],
                       [cameraposition[3][0][0], -cameraposition[3][2][0], cameraposition[3][1][0]]]

    # Different colors are assigned to each of the cameras
    colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cameraposition2, colors


def get_cam_rotation_matrices():
    """
    Generates camera rotation matrices, looking down 45 degrees towards the center of the room
    :return: camera rotations matrices.
    """

    cam_rotations = []

    for c in range(0, 4):
        cam = c + 1
        path = 'data/cam' + str(cam)
        rvecs, tvecs, mtx, dist = get_parameters(path)

        rmtx, _ = cv.Rodrigues(rvecs)

        matrix = glm.mat4([
            [rmtx[0][0], rmtx[0][2], rmtx[0][1], tvecs[0][0]],
            [rmtx[1][0], rmtx[1][2], rmtx[1][1], tvecs[1][0]],
            [rmtx[2][0], rmtx[2][2], rmtx[2][1], tvecs[2][0]],
            [0, 0, 0, 1]
        ])

        # Considering that the camera rotation matrices are rotated with glm.
        glm_mat = glm.rotate(matrix, glm.radians(-90), (0, 1, 0))
        glm_mat = glm.rotate(glm_mat, glm.radians(180), (1, 0, 0))

        cam_rotations.append(glm_mat)

    return cam_rotations
