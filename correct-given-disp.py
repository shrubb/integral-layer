import numpy as np
import cv2

def processGivenDisparity(disp, imgLeft):
    if disp.dtype == np.uint16:
        disp = disp.astype(np.int16)
    assert(disp.dtype == np.int16)
    mask = (disp <= 48)
    disp //= 48
    
    noConfidenceFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    noConfidenceFilter.setLambda(8000)
#     noConfidenceFilter.setSigmaColor(0.75)

    dispFilteredNoConfidence = noConfidenceFilter.filter(disp, imgLeft)
    
    reduceConstant = dispFilteredNoConfidence.max() * 1.01 / 255.0
    assert dispFilteredNoConfidence.max() / reduceConstant <= 255.0
    
    dispFilteredNoConfidenceFloat = dispFilteredNoConfidence.astype(np.float64)
    dispFilteredNoConfidenceFloat[dispFilteredNoConfidenceFloat <= reduceConstant] = reduceConstant
    dispFilteredNoConfidenceFloat /= reduceConstant
    dispFilteredNoConfidenceUInt8 = dispFilteredNoConfidenceFloat.astype(np.uint8)
    
    dispFilteredNoConfidenceInpainted = cv2.inpaint(
        dispFilteredNoConfidenceUInt8, 
        mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    
    np.copyto(dispFilteredNoConfidenceFloat, dispFilteredNoConfidenceInpainted, 'unsafe', where=mask)
    dispFilteredNoConfidenceFloat *= reduceConstant
    np.copyto(dispFilteredNoConfidence, dispFilteredNoConfidenceFloat, 'unsafe')
    
    dispFilteredNoConfidence -= (dispFilteredNoConfidence.min() - 2)
        
    return dispFilteredNoConfidence.astype(np.uint16)

cityscapes_base = '/media/hpc4_Raid/e_burkov/Datasets/Cityscapes/'
leftImgDir = cityscapes_base + 'leftImg8bitOriginal/'
givenDispDir = cityscapes_base + 'disparity/'
fixedDispDir = cityscapes_base + 'disparityFixedOrig/'
ourDispDir = cityscapes_base + 'disparityOur/'

import os
from sys import stdout
count = 0

for subset in os.listdir(leftImgDir):
    for city in os.listdir(leftImgDir + subset):
        for file in os.listdir(leftImgDir + subset + '/' + city):
            imgLeft = cv2.imread(leftImgDir + subset + '/' + city + '/' + file)
            disp = cv2.imread(givenDispDir + subset + '/' + city + '/' + file.replace('leftImg8bit', 'disparity'), cv2.IMREAD_ANYDEPTH)

            os.makedirs(fixedDispDir + subset + '/' + city, exist_ok=True)
            cv2.imwrite(fixedDispDir + subset + '/' + city + '/' + file.replace('leftImg8bit', 'disparity'), processGivenDisparity(disp, imgLeft))
            count += 1
            print(count)
            stdout.flush()