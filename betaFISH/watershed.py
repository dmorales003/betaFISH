# segmentation algorithm of fluorescence
# channel based on watershed. threshold
# may be altered to improve segmentation

import cv2
import numpy as np

kernel = np.ones((3,3),np.uint8)

def segmentCells(filename, threshold):
    image = cv2.imread(filename)
    image_bw = image[:,:,0]
    thresh = cv2.adaptiveThreshold(
        src=image_bw,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=threshold,
        C=-1)
    opening = cv2.morphologyEx(
        src=thresh,
        op=cv2.MORPH_OPEN,
        kernel=kernel,
        iterations=1)
    surebg = cv2.dilate(
        src=opening,
        kernel=kernel,
        iterations=10)
    dist_transform = cv2.distanceTransform(
        src=opening,
        distanceType=cv2.DIST_L2,
        maskSize=5)
    ret_,surefg = cv2.threshold(
        src=dist_transform,
        thresh=0.1*dist_transform.max(),
        maxval=255,
        type=0)
    surefg = np.uint8(surefg)
    surebg = np.uint8(surebg)
    unknown = cv2.subtract(surebg,surefg)
    ret2,markers = cv2.connectedComponents(surefg)
    markers = markers + 10
    markers[unknown==255] = 0
    markers = cv2.watershed(image, markers)
    contours,hierarchy = cv2.findContours(
        image = markers,
        mode=cv2.RETR_CCOMP,
        method=cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return contours,hierarchy

def main():
    pass

if __name__ == '__main__':
    main()