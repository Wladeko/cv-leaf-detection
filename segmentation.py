import cv2 as cv
import os
import numpy as np
import copy
from tqdm import tqdm


color_path = os.path.join("data", "color")
write_path = os.path.join("data", "prediction_mask")
files = os.listdir(color_path)
color_file_paths = [os.path.join(color_path, file) for file in files]
write_file_paths = [os.path.join(write_path, file) for file in files]


for i, photo in tqdm(
    enumerate(color_file_paths), total=1000, desc="Creating target masks"
):
    # ----------------------------------------thresh

    # Load the image and convert it to the HSV color space
    image = cv.imread(photo)
    # cv.imshow("Photo1", image)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define the range of colors to be detected
    lower_color = np.array([10, 30, 0])
    upper_color = np.array([90, 255, 255])

    # Threshold the image to create a binary mask
    mask = cv.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to fill in gaps and eliminate noise
    mask = cv.dilate(mask, None, iterations=3)
    mask1 = cv.erode(mask, None, iterations=3)

    # Overlay the mask on the original image to highlight the detected leaves
    result = cv.bitwise_and(image, image, mask=mask1)

    # Save the result image
    # cv.imshow("Result2", mask1)

    # ----------------------------------------range

    mask_brown = cv.inRange(hsv, (8, 60, 10), (30, 255, 200))
    mask_yellow_green = cv.inRange(hsv, (10, 39, 64), (86, 255, 255))
    mask = cv.bitwise_or(mask_yellow_green, mask_brown)
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    result = cv.bitwise_and(image, image, mask=mask2)
    # cv.imshow("Result3", mask2)

    # ----------------------------------------greens

    b, g, r = cv.split(image)
    inv_b = 255 - b
    median = cv.medianBlur(inv_b, 5)
    # cv.imshow("Median", median)
    threshold, mask3 = cv.threshold(median, 160, 255, cv.THRESH_BINARY)
    result = cv.bitwise_and(image, image, mask=mask3)
    # cv.imshow("Result4", mask3)

    # ----------------------------------------final mask
    y = len(mask1)
    x = len(mask1[0])
    mask_sum = [
        [int(mask1[i][j]) + int(mask2[i][j]) + int(mask3[i][j]) for j in range(x)]
        for i in range(y)
    ]
    final_mask = [
        [255 if mask_sum[i][j] > 382.5 else 0 for j in range(x)] for i in range(y)
    ]
    final_mask = np.array(final_mask, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)
    final_mask = cv.dilate(final_mask, None, iterations=3)
    final_mask = cv.erode(final_mask, None, iterations=3)

    # cv.imshow("Final", final_mask)
    result = cv.bitwise_and(image, image, mask=final_mask)
    # cv.imshow("Result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite(write_file_paths[i], final_mask)
