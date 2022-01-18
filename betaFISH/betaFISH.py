# Segmentation and blob detection of fluorescence microscopy images
# for smFISH optimized for pill-shaped bacteria such as E. coli
# and Burkholderia. Parsing through images requires distinct 
# identifiers for cell image and smFISH channels separated by '_'
# example file name: 'treatment_stain_#_channel_x_y_z_Fr1.tiff'

script = 'betaFISH'
ver = '1.3'

# global packages
import os
import csv
import timeit
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
# local pacakages
from imgparse import image_list
from watershed import segmentCells
from display import save_blobs, save_cells
from blob_detect import detectBlobs, count, blobSize

class Fisher:

    def __init__(self, filepath, identifier, channels, export, cellThresh, spotThresh):
        # file name parser
        self.filepath = filepath
        self.channels = channels
        self.directory = os.path.split(self.filepath)[0]
        self.file = os.path.split(self.filepath)[1]
        self.outdir = os.path.join(self.directory,'betaFISH_output')
        self.optdir = os.path.join(self.directory,'betaFISH_optimize')
        self.index = self.file.split(identifier)
        self.fieldnames = ['image','treatment','stain','cell','area'] + channels
        # boolean variables
        self.export = export
        # parameter variables
        self.cellThresh = cellThresh
        self.spotThresh = len(self.channels)*[spotThresh]
        self.colors = [(0,255,0),(0,0,255),(255,0,0),(255,255,0),
                        (255,0,255),(0,255,255),(255,255,255)]

    def _features(self):
        # tabulate data returns list of dictionaries
        self.data = count(
            self.file, 
            self.fieldnames, 
            self.contours, 
            self.hierarchy, 
            self.spots)
        return self.data

    def _spotFeatures(self):
        self.spotData = blobSize(
            self.file, 
            self.fieldnames, 
            self.contours, 
            self.hierarchy, 
            self.spots)
        return self.spotData

    def _write(self):
        # output ._features() as a csv file
        self.resultsfile = os.path.join(self.outdir,'report.csv')
        self.spotsfile = os.path.join(self.outdir, 'spotsreport.csv')
        if os.path.exists(self.resultsfile):
            with open(self.resultsfile, 'a+', newline='') as output:
                write = csv.DictWriter(output,self.fieldnames)
                write.writerows(self._features())
        else:
            with open(self.resultsfile, 'w', newline='') as output:
                write = csv.DictWriter(output,self.fieldnames)
                write.writeheader()
                write.writerows(self._features())
        output.close()
        cols, spotSizes = self._spotFeatures()
        if os.path.exists(self.spotsfile):
            with open(self.spotsfile, 'a+', newline='') as spotOut:
                write = csv.DictWriter(spotOut, cols)
                write.writerows(spotSizes)
        else:
            with open(self.spotsfile, 'w', newline='') as spotOut:
                write = csv.DictWriter(spotOut, cols)
                write.writeheader()
                write.writerows(spotSizes)
        spotOut.close()

    def _export(self):
        # if export == True, returns segmented and blob images
        save_cells(
            self.filepath, 
            self.contours, 
            self.hierarchy, 
            os.path.join(self.outdir,self.file))
        for ch,spot,color in zip(self.channels, self.spots, self.colors):
            save_blobs(
                os.path.join(self.directory,(ch).join(self.index)),
                spot,
                os.path.join(self.outdir,(ch).join(self.index)),
                color)

    def file_maker(self):
        pass

    def _optimize(self):
        # optimize cell watershed segmentation
        cellThreshFile = os.path.join(self.optdir,'cell_optimize.csv')
        thresh_range = list(range(3,256,4))
        keys = ['file']+sorted(thresh_range)
        values = [self.index[0]]
        for i in thresh_range:
            self.contours, self.hierarchy = segmentCells(self.filepath,i)
            save_cells(
                self.filepath,
                self.contours,
                self.hierarchy, 
                os.path.join(self.optdir,(f'cell_{i}').join(self.index))
            )
            values.append(len(self.contours))
        data = dict(zip(keys,values))
        if os.path.exists(cellThreshFile):
            with open(cellThreshFile, 'a+', newline='') as ct:
                writer = csv.DictWriter(ct, keys)
                writer.writerow(data)
        else:
            with open(cellThreshFile,'w') as ct:
                writer = csv.DictWriter(ct, fieldnames=keys)
                writer.writeheader()
                writer.writerow(data)
        ct.close()
        # optimize blob detection
        spotThreshFile = os.path.join(self.optdir,'spot_optimize.csv')
        data = []        
        thresh_range = [x/100000 for x in range(500,5125,125)]
        keys = ['file', 'channel'] + thresh_range
        for channel,color in zip(self.channels, self.colors):
            values = [self.index[0], channel]
            for j in thresh_range:
                image = os.path.join(self.directory,(channel).join(self.index))
                blobs = detectBlobs(image,j)
                save_blobs(
                    image, 
                    blobs, 
                    os.path.join(self.optdir,(channel+f'_{j}').join(self.index)),
                    color
                    )
                values.append(len(blobs))
            data.append(dict(zip(keys,values)))
        if os.path.exists(spotThreshFile):
            with open(spotThreshFile, 'a+', newline='') as st:
                write = csv.DictWriter(st, keys)
                for line in data:
                    write.writerow(line)
        else:
            with open(spotThreshFile, 'w') as st:
                writer = csv.DictWriter(st, fieldnames=keys)
                writer.writeheader()
                for line in data:
                    writer.writerow(line)
        st.close()


    def _run(self):
        # perform segmentation and analysis
        self.contours, self.hierarchy = segmentCells(self.filepath,self.cellThresh)
        self.spots = []
        for channel,thresh in zip(self.channels,self.spotThresh):
            self.spots.append(
                detectBlobs(os.path.join(self.directory,(channel).join(self.index)), 
                thresh))
        self._write()
        
        if self.export == True:
            self._export()
    


def runOptimizer(directory, identifier, channels, export, optimizer):
    optimize_folder = os.path.join(directory,'betaFISH_optimize')
    if os.path.exists(optimize_folder) == False:
        os.mkdir(optimize_folder)

    images = image_list(directory, identifier)
    # fraction = round(len(images)/10)
    random.seed(42)
    subsample = random.sample(images, optimizer)
    for image in subsample:
        fish = Fisher(image, identifier, channels, export, None, None)
        fish._optimize()



def runFisher(directory, identifier, channels, export, cellThresh, spotThresh):
    # generate results folder
    export_folder = os.path.join(directory,'betaFISH_output')
    if os.path.exists(export_folder) == False:
        os.mkdir(export_folder)
    
    # acquire list of files to parse through
    images = image_list(directory, identifier)
    
    # tqdm compatible script
    for image in tqdm(images):
        fish = Fisher(image, identifier, channels, export, cellThresh, spotThresh)
        fish._run()



def main():

    initial = timeit.default_timer()

    inputs = argparse.ArgumentParser(description=\
        '%s version %s. Input a directory of images to return a csv file '
        'tabulating cells detected per image and the spots per cell for specified '
        'fluorescence channels. Use \'-i\' to specify the string token referencing '
        'the cell image for segmentation. smFISH channels are imported through the '
        'inclusion of the \'-c\' argument followed by the string token for that channel. '
        'The threshold for both the cell segmentation and blob detection may be altered'
        'via the \'-t\' & \'-s\' arguments, respectively. If resulting segmentation '
        'and spots localization images are desired, images will can be exported to the '
        'output folder if \'-e\' is flagged.' % (script, ver))
    req = inputs.add_argument_group('required arguments')
    req.add_argument('-d', '--directory', action='store', required=True,
                        help='Provide directory that image files are located.')
    req.add_argument('-i', '--identifier', action='store', required=True, type=str,
                        help='Provide primary string for fluorescence channel of cell identifier.')
    req.add_argument('-c', '--channels', action='append', required=True,
                        help='Provide primary strings for fluorescence channels for smFISH')
    inputs.add_argument('-e', '--export', action='store_true', default=False,
                        help='Flag if images of segmentation and spot detection desired for export.')
    inputs.add_argument('-t', '--threshold', action='store', default=99, type=int,
                        help='Set threshold for watershed semgentation.')
    inputs.add_argument('-s', '--spotThresholds', action='store', default=0.1, type=float,
                        help='Set thresholds for blob detection')
    inputs.add_argument('-O', '--Optimizer', action='store', default=0, type=int,
                        help='Flag to optimize counting of cells and spots, select number of images to analyze.')


    # Command line values 
    args = inputs.parse_args()
    directory = args.directory
    identifier = args.identifier
    channels = args.channels
    export = args.export
    cellThresh = args.threshold
    spotThresh = args.spotThresholds
    optimizer = args.Optimizer
    
    if type(optimizer) != int:
        print('Please select a positive integer.')
    elif optimizer < 0:
        print('Please select a positive number.')
    elif optimizer > 0:
        runOptimizer(directory,identifier,channels,export,optimizer)
    else:
        runFisher(directory, identifier, channels, export, cellThresh, spotThresh)

    # timeit enabled
    print(f'Program completed in {(timeit.default_timer()-initial)/60} minutes.')

if __name__ == '__main__':
    main()

