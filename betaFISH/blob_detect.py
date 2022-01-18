# blob detection algorithm based on 
# scikitimage: laplacian of gaussian 
# threshold at 0.01 for sufficient coverage

import cv2
import numpy as np
from math import sqrt
from skimage.feature import blob_log

def detectBlobs(filename, threshold):
    image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    spots = blob_log(
        image=image,
        max_sigma=10,
        min_sigma=1,
        threshold=threshold
    )
    spots[:,2] = spots[:,2]*sqrt(2)
    return spots

def count(filename, fieldnames, contours, hierarchy, spots):    
    # empty list for dictionary objects
    data = []
    treatment, stain = filename.split('_')[0:2]
    # treatment = filename.split('_')[0:1]
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            area = cv2.contourArea(contours[i])
            values = [filename, treatment, stain, i, area]
            for channel in spots:
                number = 0# count number of spots in each cell
                for blob in channel:
                    y, x = blob[:2]
                    if cv2.pointPolygonTest(contours[i],(x, y),True) < -0:
                        continue
                    number += 1
                values.append(number)
            data.append(dict(zip(fieldnames, values)))
        else:
            continue
    return data

def blobSize(filename, fieldnames, contours, hierarchy, spots):    
    # empty list for dictionary objects
    spotSizes = []
    treatment, stain = filename.split('_')[0:2]
    ch = fieldnames[5:]
    columns = fieldnames[0:5] + ['channel', 'spot#', 'radius']
    # treatment = filename.split('_')[0:1]
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            area = cv2.contourArea(contours[i])
            for j,channel in enumerate(spots):
                for k,blob in enumerate(channel):
                    y, x, r = blob
                    if cv2.pointPolygonTest(contours[i],(x, y),True) < -0:
                        continue
                    values = [filename, treatment, stain, i, area, ch[j], k, r]
                    spotSizes.append(dict(zip(columns, values)))
        else:
            continue
    return columns, spotSizes

def main():
    pass

if __name__ == '__main__':
    main()
