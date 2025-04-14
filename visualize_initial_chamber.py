import json
import os
import random

import cv2
import matplotlib.pyplot as plt
from iterwrap import iterate_wrapper
from tqdm import tqdm

from common.args import feat_recog_args
from feat_recognize.initial_chamber import ProloculusDetector

detector = ProloculusDetector()

images = os.listdir("dataset/common/filtered_images/")

# Randomly sample 10 images from the list
# sampled_images = random.sample(images, min(5, len(images)))
# sampled_images = images[:5]
sampled_images = ["Fusulina_kirovi_1_1.png"]

# os.makedirs("visualize_tools/initial_chamber", exist_ok=True)

for image in tqdm(sampled_images):
    image_path = os.path.join("dataset/common/filtered_images/", image)
    initial_chamber_center = detector.detect_initial_chamber(image_path, threshold=0.0)
