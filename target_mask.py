import cv2 as cv
import os
import numpy as np
import copy
from tqdm import tqdm


segmented_path = os.path.join("data", "segmented")
write_path = os.path.join("data", "target_mask")
files = os.listdir(segmented_path)
segmented_file_paths = [os.path.join(segmented_path, file) for file in files]
write_file_paths = [os.path.join(write_path, file) for file in files]

for i, photo in tqdm(
    enumerate(segmented_file_paths), total=1000, desc="Creating target masks"
):
    img = cv.imread(photo)

    b, g, r = cv.split(img)

    median = cv.medianBlur(g, 5)

    threshold, thresh = cv.threshold(median, 15, 255, cv.THRESH_BINARY)

    # kernel = np.ones((7, 7), np.uint8)
    # closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    # height, width, _ = img.shape
    # img_blue = copy.copy(img)

    # for i in range(height):
    #     for j in range(width):
    #         # img[i, j] is the RGB pixel at position (i, j)
    #         # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
    #         if img_blue[i, j].sum() <= 10:
    #             img_blue[i, j] = [255, 0, 0]

    # masked = cv.bitwise_and(img_blue, img_blue, mask=thresh)

    # masked_inv = cv.bitwise_and(img_blue, img_blue, mask=cv.bitwise_not(thresh))

    # hori = np.concatenate((img, img_blue, masked, masked_inv), axis=1)
    # cv.imshow("Analysis", hori)
    # cv.imshow("Mask", thresh)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    cv.imwrite(write_file_paths[i], thresh)
