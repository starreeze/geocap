Feature recognition for specific part of a fossil image.

### Initial Chamber

The `detect_initial_chamber` function in `initial_chamber.py` uses `cv2.houghcircle` to detect the circle in the central region of the image.

### Volutions

The `VolutionCounter` class in `volution_counter.py` is designed to detect and measure the volutions with a "adsorb-scan" strategy. Call the `count_volutions` method with the input image to get the volutions-related features, it also returns whether the initial chamber was detected with a high confidence level.

### Usage Example

The `recognize_feature` function in `recognize.py` is an example of feature recognition. It first recognizes the volutions in the fossil image, and try to detect the initial chamber twice with different confidence level.