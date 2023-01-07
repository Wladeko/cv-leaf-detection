import cv2 as cv
import os
import numpy as np
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
    raw = cv.imread(photo)
    image = cv.GaussianBlur(raw, (3, 3), 1)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define the range of colors to be detected
    lower_color = np.array([10, 30, 30])
    upper_color = np.array([90, 255, 255])

    # Threshold the image to create a binary mask
    mask1 = cv.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to fill in gaps and eliminate noise
    mask1 = cv.dilate(mask1, None, iterations=3)
    mask1 = cv.erode(mask1, None, iterations=3)

    # Overlay the mask on the original image to highlight the detected leaves
    # result1 = cv.bitwise_and(image, image, mask=mask1)

    # ----------------------------------------range

    mask_brown = cv.inRange(hsv, (8, 60, 30), (30, 255, 200))
    mask_yellow_green = cv.inRange(hsv, (10, 39, 64), (86, 255, 255))
    mask = cv.bitwise_or(mask_yellow_green, mask_brown)
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    # result2 = cv.bitwise_and(image, image, mask=mask2)

    # ----------------------------------------greens

    b, g, r = cv.split(image)

    inv_g = 255 - g
    inv_g = cv.medianBlur(inv_g, 3)
    _, leaf_shadow = cv.threshold(inv_g, 220, 255, cv.THRESH_BINARY)
    leaf_shadow = cv.medianBlur(leaf_shadow, 5)

    inv_b = 255 - b
    median = cv.medianBlur(inv_b, 5)
    _, mask3 = cv.threshold(median, 130, 255, cv.THRESH_BINARY)

    mask3 = mask3 - leaf_shadow
    mask3 = cv.medianBlur(mask3, 5)
    # result3 = cv.bitwise_and(image, image, mask=mask3)

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
    final_mask = cv.medianBlur(final_mask, 7)

    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)

    h, w = final_mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flooded_ul = final_mask.copy()
    flooded_dl = final_mask.copy()
    flooded_ur = final_mask.copy()
    flooded_dr = final_mask.copy()
    cv.floodFill(flooded_ul, flood_mask, (0, 0), 255)
    cv.floodFill(flooded_dl, flood_mask, (0, 254), 255)
    cv.floodFill(flooded_ur, flood_mask, (254, 0), 255)
    cv.floodFill(flooded_dr, flood_mask, (254, 254), 255)
    flooded_ul = cv.bitwise_not(flooded_ul)
    flooded_dl = cv.bitwise_not(flooded_dl)
    flooded_ur = cv.bitwise_not(flooded_ur)
    flooded_dr = cv.bitwise_not(flooded_dr)
    mask_u = cv.bitwise_and(flooded_ul, flooded_ur)
    mask_d = cv.bitwise_and(flooded_dl, flooded_dr)
    holes = cv.bitwise_and(mask_u, mask_d)

    final_mask = cv.bitwise_or(final_mask, holes)

    result = cv.bitwise_and(raw, raw, mask=final_mask)

    cv.imwrite(write_file_paths[i], final_mask)
