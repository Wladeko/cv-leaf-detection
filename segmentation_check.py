import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
import copy


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
    # # cv.imshow("Photo1", raw)
    image = cv.GaussianBlur(raw, (3, 3), 1)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define the range of colors to be detected
    lower_color = np.array([10, 30, 30])
    upper_color = np.array([90, 255, 255])

    # Threshold the image to create a binary mask
    mask = cv.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to fill in gaps and eliminate noise
    mask = cv.dilate(mask, None, iterations=3)
    mask1 = cv.erode(mask, None, iterations=3)

    # Overlay the mask on the original image to highlight the detected leaves
    result = cv.bitwise_and(image, image, mask=mask1)

    # Save the result image
    # # cv.imshow("Result1 - Threshold", mask1)

    # ----------------------------------------range

    mask_brown = cv.inRange(hsv, (8, 60, 30), (30, 255, 200))
    mask_yellow_green = cv.inRange(hsv, (10, 39, 64), (86, 255, 255))
    mask = cv.bitwise_or(mask_yellow_green, mask_brown)
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    result = cv.bitwise_and(image, image, mask=mask2)
    # # cv.imshow("Result2 - Mask browen and yellow_green", mask2)

    # ----------------------------------------greens

    b, g, r = cv.split(image)
    inv_b = 255 - b
    inv_g = 255 - g
    inv_g = cv.medianBlur(inv_g, 3)
    _, leaf_shadow = cv.threshold(inv_g, 220, 255, cv.THRESH_BINARY)
    leaf_shadow = cv.medianBlur(leaf_shadow, 5)
    # leaf_shadow = cv.dilate(leaf_shadow, None, iterations=3)
    # leaf_shadow = cv.erode(leaf_shadow, None, iterations=3)
    hori = np.concatenate((r, g, b, inv_g, inv_b), axis=1)
    # # cv.imshow("Analysis rgb", hori)
    # # cv.imshow("Leaf shadow", leaf_shadow)
    median = cv.medianBlur(inv_b, 5)
    # # cv.imshow("Median", median)
    _, mask3 = cv.threshold(median, 130, 255, cv.THRESH_BINARY)
    mask3 = mask3 - leaf_shadow
    mask3 = cv.medianBlur(mask3, 5)
    result = cv.bitwise_and(image, image, mask=mask3)
    # # cv.imshow("Result3 - Inverted blue", mask3)

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
    # # cv.imshow("Final before operations", final_mask)
    final_mask = cv.medianBlur(final_mask, 7)

    kernel = np.ones((5, 5), np.uint8)

    final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)
    # final_mask = cv.dilate(final_mask, None, iterations=3)
    # final_mask = cv.erode(final_mask, None, iterations=3)

    h, w = final_mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flooded_ul = final_mask.copy()
    flooded_dl = final_mask.copy()
    flooded_ur = final_mask.copy()
    flooded_dr = final_mask.copy()
    cv.floodFill(flooded_ul, flood_mask, (0, 0), 255)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(flooded_dl, flood_mask, (0, 254), 255)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(flooded_ur, flood_mask, (254, 0), 255)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(flooded_dr, flood_mask, (254, 254), 255)
    # cv.imshow("flooded_ul", flooded_ul)
    # cv.imshow("flooded_dl", flooded_dl)
    # cv.imshow("flooded_ur", flooded_ur)
    # cv.imshow("flooded_dr", flooded_dr)
    flooded_ul = cv.bitwise_not(flooded_ul)
    flooded_dl = cv.bitwise_not(flooded_dl)
    flooded_ur = cv.bitwise_not(flooded_ur)
    flooded_dr = cv.bitwise_not(flooded_dr)
    # cv.imshow("flooded_ul_not", flooded_ul)
    mask_u = cv.bitwise_and(flooded_ul, flooded_ur)
    # cv.imshow("mask_u", mask_u)
    mask_d = cv.bitwise_and(flooded_dl, flooded_dr)
    # cv.imshow("mask_d", mask_d)
    holes = cv.bitwise_and(mask_u, mask_d)
    # holes = cv.bitwise_not(mask_all)
    # cv.imshow("flooded", holes)

    final_mask = cv.bitwise_or(final_mask, holes)

    # cv.imshow("Final", final_mask)
    result = cv.bitwise_and(raw, raw, mask=final_mask)
    # cv.imshow("Result", result)
    hori = np.concatenate((raw, result), axis=1)
    # cv.imshow("Analysis", hori)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite(write_file_paths[i], final_mask)
