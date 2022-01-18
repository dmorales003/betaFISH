# collection of methods for handling images
# based on OpenCV Python Package

import os

def image_list(directory, channel='filter0'):
    image_list = set()
    for root, dir, filename in os.walk(directory):
        for file in filename:
            name = file.split('_')
            if file.endswith(
                ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')
                ) and f'{channel}' in name:
                image_list.add(os.path.join(root,file))                           
    print(f'{len(image_list)} number of image sets found.')
    return list(image_list)

def main():
    pass

if __name__ == '__main__':
    main()

