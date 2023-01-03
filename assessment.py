import cv2 as cv
import os
import numpy as np
import copy
from tqdm import tqdm


predict_path = os.path.join("data", "prediction_mask")
target_path = os.path.join("data", "target_mask")
files_predict = os.listdir(predict_path)
files_target = os.listdir(target_path)
predict_file_paths = [os.path.join(predict_path, file) for file in files_predict]
target_file_paths = [os.path.join(target_path, file) for file in files_target]
