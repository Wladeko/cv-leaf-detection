import cv2 as cv
import os
import numpy as np
import copy
from tqdm import tqdm


def calculate_area(img):
    area = 0
    for i in img:
        area += sum(i)
    return area


predict_path = os.path.join("data", "prediction_mask")
target_path = os.path.join("data", "target_mask")
files_predict = os.listdir(predict_path)
files_target = os.listdir(target_path)
predict_file_paths = [os.path.join(predict_path, file) for file in files_predict]
target_file_paths = [
    os.path.join(target_path, file.replace(".JPG", "_final_masked.jpg"))
    for file in files_predict
]

iou = []
dice = []

for i, file in tqdm(
    enumerate(predict_file_paths), total=1000, desc="Calculating metrics"
):
    target_mask = cv.imread(file, cv.IMREAD_GRAYSCALE)
    _, target_mask = cv.threshold(target_mask, 15, 255, cv.THRESH_BINARY)
    predict_mask = cv.imread(target_file_paths[i], cv.IMREAD_GRAYSCALE)
    _, predict_mask = cv.threshold(predict_mask, 15, 255, cv.THRESH_BINARY)

    intersection = cv.bitwise_and(target_mask, predict_mask)
    union = cv.bitwise_or(target_mask, predict_mask)

    i_area = calculate_area(intersection)
    u_area = calculate_area(union)
    iou.append(i_area / u_area)

    a = calculate_area(target_mask)
    b = calculate_area(predict_mask)
    dice.append(2 * i_area / (a + b))

print(f"Mean IoU: {sum(iou)/len(iou)}\tMean Dice: {sum(dice)/len(dice)}")
