import cv2 as cv
import numpy as np
from skimage.filters import threshold_multiotsu


def average_mask(path):
    """
    Get the average mask by using all frames of the videos.
    :param path: the path where the camera folders are located.
    :return: the background model.
    """
    cap = cv.VideoCapture(path + 'background.avi')
    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Initialize variables to keep track of the frames and the sum of the frames
    frame_sum = None

    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Add the frame to the sum of the frames
        if frame_sum is None:
            frame_sum = frame.astype(int)
        else:
            frame_sum += frame.astype(int)

    # Compute the average frame by dividing the sum of the frames by the frame count
    bk_model = (frame_sum / nframes).astype('uint8')

    return bk_model


def background_subtraction(bk_model, path):

    """
    Changes the background to HSV, otsu thresholding is applied on the hue
    to automatize the thresholding obtaining.
    Morphological operations and find contours is applied to optimize the
    shape.
    :param bk_model: background model
    :param path: path where the videos are located.
    :return: the foreground mask.
    """
    # Convert background to HSV
    hsv_background = cv.cvtColor(bk_model, cv.COLOR_BGR2HSV)
    # Load foreground video
    cap = cv.VideoCapture(path + 'video.avi')

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # If there are no more frames in the video, exit the loop
        if not ret:
            break

        # Convert the frame from BGR to HSV color space
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Calculate the absolute difference between the background model and the current frame
        diff = cv.absdiff(hsv_frame, hsv_background)

        # Threshold the difference image for the Hue, Saturation, and Value channels
        ret, hue_mask = cv.threshold(diff[:, :, 0], 100, 255, cv.THRESH_BINARY)
        ret, sat_mask = cv.threshold(diff[:, :, 1], 20, 255, cv.THRESH_BINARY)
        ret, val_mask = cv.threshold(diff[:, :, 2], 100, 255, cv.THRESH_BINARY)

        # Apply otsu thresholding
        hue=diff[:, :, 0]
        sat=diff[:, :, 1]
        val=diff[:, :, 2]

        # Combine the thresholded channels to create a binary mask
        mask = cv.bitwise_xor(hue_mask, cv.bitwise_or(sat_mask, val_mask))

        # Apply morphological operations to the binary mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Find the contours in the binary mask
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Find the contour with the maximum area
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # Create a blank image to draw the contour on
        foreground_mask = np.zeros_like(frame)

        # Draw the contour with the maximum area on the new mask
        cv.drawContours(foreground_mask, [max_contour], -1, (255, 255, 255), cv.FILLED)

        # Show the final mask
        cv.imshow('Foreground Mask', foreground_mask)

        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv.destroyAllWindows()

    return foreground_mask


def gaussian_background_subtraction(path, c):
    """
    Creates the background model using Gaussian opeations, then it is
    applied to the frames of the video and findContours is used to
    perfectionate the shape of the subject.
    :param path: path where the videos are located
    :param c: camera on which we are.
    :return: the final forground binary mask.
    """

    # Create video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    # 486 644 are the height and width of the video files
    frame_width = 486
    frame_height = 644
    fps = 30
    out = cv.VideoWriter('foreground' + str(c) + '.avi', fourcc, fps, (frame_height, frame_width))

    # create background subtractor object
    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=True)

    # capture the background video
    bg_video = cv.VideoCapture(path + 'background.avi')

    # train the background model
    while True:
        ret, frame = bg_video.read()
        if not ret:
            break
        # apply the background subtractor to the frame
        _ = bg_subtractor.apply(frame, learningRate=0.01)

    # capture the foreground video
    fg_video = cv.VideoCapture(path + 'video.avi')

    # loop through each frame of the foreground video
    count=0
    while True:
        ret, frame = fg_video.read()

        if not ret:
            break
        bk_model1 = bg_subtractor.apply(frame, learningRate=0)

        # Automatic segmentation of the images: Apply Multi-Otsu Thresholding and select the second threshold
        thresholds = threshold_multiotsu(bk_model1)

        bk_model = cv.threshold(bk_model1, thresholds[1], 255, cv.THRESH_BINARY)[1]

        # Apply morphological operations to the binary mask
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        # mask = cv.morphologyEx(bk_model, cv.MORPH_CLOSE, kernel)
        # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Find the contours in the binary mask
        contours, hierarchy = cv.findContours(bk_model, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Find the contour with the maximum area
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # Create a blank image to draw the contour on
        final_mask = np.zeros_like(frame)

        # Draw the contour with the maximum area on the new mask
        cv.drawContours(final_mask, [max_contour], -1, (255, 255, 255), cv.FILLED)

        # Show the final mask
        cv.imshow('Foreground Mask', final_mask)

        frame = cv.bitwise_and(frame, final_mask)

        cv.imshow('Foreground', frame)

        # write the every mask as a frame on the video
        out.write(final_mask)

        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    fg_video.release()
    cv.destroyAllWindows()
    out.release()

    return final_mask


if __name__ == '__main__':
    for c in range(1, 5):
        path_cam = f'data/cam{c}/'

        #background_model = average_mask(path_cam)
        #background_subtraction(background_model, path_cam)

        # Only the gaussian background subtraction method is used.
        gaussian_background_subtraction(path_cam,c)
