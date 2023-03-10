import cv2 as cv
from assignment import get_parameters
import numpy as np
import pickle

def look_up_table():
    """
    creates look up table
    """

    rvecs, tvecs, mtx, dist = [], [], [], []

    # Parameters for the 3D space definition
    width = int(60)
    height = int(60)
    depth = int(60)

    # Iterates through all  the cams
    for c in range(0, 4):
        cam = c + 1
        path = 'data/cam' + str(cam)
        table = {}
        # Get parameters for
        r, t, m, d = get_parameters(path)

        # Iterates through all the voxels in our 3D space
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    # Iterates through all cameras

                    # Creates the voxel that we are paying attention to. Multiplied by 40 to enhance visualization.
                    voxels = ((x - width / 2) * 40, (y - height / 2) * 40, -z * 40)

                    # Obtain the 2D position on each camera
                    pixel_pts, _ = cv.projectPoints(voxels, r, t, m, d)

                    pixel_pts = np.reshape(pixel_pts, 2)
                    pixel_pts = pixel_pts[::-1]

                    table[voxels] = pixel_pts

        save_path = path + "/look_up_table.pickle"
        # Open a file for writing in binary mode
        with open(save_path, "wb") as f:
            # Use the pickle.dump() function to serialize the dictionary and save it to the file
            pickle.dump(table, f)

        print("Dictionary saved to file.")

    return table


if __name__ == '__main__':
    table = look_up_table()
