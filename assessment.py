import cv2 as cv
import os
import numpy as np
import copy
from tqdm import tqdm


def calculate_area(img):
    normalize = img / 255
    normalize = normalize.astype("int32")
    return sum(sum(normalize))


predict_path = os.path.join("data", "prediction_mask")
target_path = os.path.join("data", "target_mask")
photo_path = os.path.join("data", "color")
files_predict = os.listdir(predict_path)
files_target = os.listdir(target_path)
predict_file_paths = [os.path.join(predict_path, file) for file in files_predict]
target_file_paths = [
    os.path.join(target_path, file.replace(".JPG", "_final_masked.jpg"))
    for file in files_predict
]
photo_file_paths = [os.path.join(photo_path, file) for file in files_predict]

iou = []
dice = []
images = []

for i, file in tqdm(
    enumerate(predict_file_paths), total=1000, desc="Calculating metrics"
):
    predict_mask = cv.imread(file, cv.IMREAD_GRAYSCALE)
    _, predict_mask = cv.threshold(predict_mask, 15, 255, cv.THRESH_BINARY)
    target_mask = cv.imread(target_file_paths[i], cv.IMREAD_GRAYSCALE)
    _, target_mask = cv.threshold(target_mask, 15, 255, cv.THRESH_BINARY)
    photo = cv.imread(photo_file_paths[i])

    intersection = cv.bitwise_and(target_mask, predict_mask)
    union = cv.bitwise_or(target_mask, predict_mask)

    # cv.imshow("Target", target_mask)
    # cv.imshow("Predict", predict_mask)
    # cv.imshow("Intersection", intersection)
    # cv.imshow("Union", union)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    i_area = calculate_area(intersection)
    u_area = calculate_area(union)
    # print(f"{i_area} / {u_area}")
    iou_value = i_area / u_area
    iou.append(iou_value)

    a = calculate_area(target_mask)
    b = calculate_area(predict_mask)
    dice_value = 2 * i_area / (a + b)
    dice.append(dice_value)

    images.append(
        (
            iou_value,
            dice_value,
            {
                "predict": cv.bitwise_and(photo, photo, mask=predict_mask),
                "target": cv.bitwise_and(photo, photo, mask=target_mask),
                "photo": photo,
            },
        )
    )

# min_iou_index = iou.index(min(iou))
# max_iou_index = iou.index(max(iou))
# min_dice_index = iou.index(min(dice))
# max_dice_index = iou.index(max(dice))

# min_iou_mask = cv.imread(predict_file_paths[min_iou_index], cv.IMREAD_GRAYSCALE)
# _, min_iou_mask = cv.threshold(min_iou_mask, 15, 255, cv.THRESH_BINARY)

# max_iou_mask = cv.imread(predict_file_paths[max_iou_index], cv.IMREAD_GRAYSCALE)
# _, max_iou_mask = cv.threshold(max_iou_mask, 15, 255, cv.THRESH_BINARY)

# min_dice_mask = cv.imread(predict_file_paths[min_iou_index], cv.IMREAD_GRAYSCALE)
# _, min_iou_mask = cv.threshold(min_iou_mask, 15, 255, cv.THRESH_BINARY)

# min_iou_mask = cv.imread(predict_file_paths[min_iou_index], cv.IMREAD_GRAYSCALE)
# _, min_iou_mask = cv.threshold(min_iou_mask, 15, 255, cv.THRESH_BINARY)

images_sorted = sorted(images)

print(
    f"Mean IoU: {round(sum(iou)/len(iou)*100, 4)}%\tMean Dice: {round(sum(dice)/len(dice)*100, 4)}%"
)
print(f"Min IoU: {round(min(iou)*100, 4)}%\tMin Dice: {round(min(dice)*100, 4)}%")
print(f"Max IoU: {round(max(iou)*100, 4)}%\tMax Dice: {round(max(dice)*100, 4)}%")


for i, photo in tqdm(enumerate(images_sorted), total=1000, desc="Saving comparison"):
    hori1 = np.concatenate(
        (
            photo[2]["predict"],
            photo[2]["target"],
            photo[2]["photo"],
        ),
        axis=1,
    )
    text = f"IoU: {round(photo[0]*100, 2)}%   Dice: {round(photo[1]*100, 2)}%"
    org = (20, 20)
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 255, 255)
    thickness = 1
    cv.putText(hori1, text, org, font, fontScale, color, thickness)
    cv.imwrite(f"data/comparison/{i+1}-worst-to-best.jpg", hori1)
