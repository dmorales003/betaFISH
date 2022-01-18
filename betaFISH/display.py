# assorted commands for displaying data acquired
# from segmentation and blob detection imagess

import cv2
from random import randint
from imgparse import image_list
import matplotlib.pyplot as plt

def show_blobs(filename, blobs, output=None):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    image = cv2.imread(filename,cv2.IMREAD_ANYDEPTH)
    ax.imshow(image)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle(
            (x,y), 
            r, 
            color='red', 
            linewidth=1,
            fill=False
        )
        ax.add_patch(c)
    plt.show()


def show_cells(filename, contours, hierarchy):
    image = cv2.imread(filename)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if (hierarchy[0][i][3] == -1 and 
            area > 100 and 
            area < 10000):
            cv2.drawContours(
                image,
                contours,
                i,
                (randint(0,255),
                 randint(0,255),
                 randint(0,255)),
                -1)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    plt.show()

def save_blobs(filename, blobs, output, color):
    image = cv2.imread(filename)
    for blob in blobs:
        y, x, r = blob
        cv2.drawMarker(
            image,
            (int(x), int(y)),
            color = color,
            markerType = cv2.MARKER_CROSS,
            markerSize = 20,
            thickness = 1,
            line_type=cv2.LINE_AA)
    cv2.imwrite(output, image)

def save_cells(filename, contours, hierarchy, output):
    image = cv2.imread(filename)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if (hierarchy[0][i][3] == -1 and 
            area > 100 and 
            area < 10000):
            cv2.drawContours(
                image,
                contours,
                i,
                (randint(0,255),
                 randint(0,255),
                 randint(0,255)),
                -1)
    cv2.imwrite(output, image)

#show an imported image
def show_image(image, cmap='Greys_r'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(image,cmap=cmap)

def show_random(directory, channel):
    images = image_list(directory,channel)
    a_number = randint(0,len(images))
    image = cv2.imread(images[a_number],cv2.IMREAD_ANYDEPTH)
    show_image(image)

def multi_image(nrows, ncols, images, size = (16,8)):
    plt.figure(figsize=size)
    for i, image in enumerate(images):
        plt.subplot(nrows, ncols, i)
        plt.title('Figure {}'.format(i))
        plt.imshow(image)
    plt.show()

def main():
    pass

if __name__ == '__main__':
    main()