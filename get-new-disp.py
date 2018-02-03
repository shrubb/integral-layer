import numpy as np
import cv2

def getNewDisparity(imgLeft, imgRight):
    leftMatcher = cv2.StereoBM_create(128)
    leftFilter = cv2.ximgproc.createDisparityWLSFilter(leftMatcher)
    rightMatcher = cv2.ximgproc.createRightMatcher(leftMatcher)

    leftGray  = cv2.cvtColor(imgLeft , cv2.COLOR_BGR2GRAY)
    rightGray = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

    dispLeft  = leftMatcher .compute(leftGray, rightGray)
    dispRight = rightMatcher.compute(rightGray, leftGray)
    
    leftFilter.setSigmaColor(1.45)
    dispFiltered = leftFilter.filter(dispLeft, imgLeft, None, dispRight)

    reduceConstant = dispFiltered.max() * 1.01 / 255.0
    assert dispFiltered.max() / reduceConstant <= 255.0

    dispFilteredFloat = dispFiltered.astype(np.float64)
    dispFilteredFloat[dispFilteredFloat <= reduceConstant] = reduceConstant
    dispFilteredFloat /= reduceConstant
    dispFilteredUInt8 = dispFilteredFloat.astype(np.uint8)

    mask = (dispFiltered <= 0).astype(np.uint8)

    dispFilteredInpainted = cv2.inpaint(
        dispFilteredUInt8, 
        mask, 3, cv2.INPAINT_TELEA)

    np.copyto(dispFilteredFloat, dispFilteredInpainted, 'unsafe', where=mask.astype(bool))
    dispFilteredFloat *= reduceConstant
    np.copyto(dispFiltered, dispFilteredFloat, 'unsafe')
    
    dispFiltered -= (dispFiltered.min() - 2)
    dispFiltered //= 2
    
    return dispFiltered.astype(np.uint16)

cityscapes_base = '/media/hpc4_Raid/e_burkov/Datasets/Cityscapes/'
leftImgDir = cityscapes_base + 'leftImg8bitOriginal/'
rightImgDir = cityscapes_base + 'rightImg8bit/'
fixedDispDir = cityscapes_base + 'disparityFixedOrig/'
ourDispDir = cityscapes_base + 'disparityOur/'

import os
from sys import stdout
count = 0

for subset in os.listdir(leftImgDir):
    for city in os.listdir(leftImgDir + subset):
        for file in os.listdir(leftImgDir + subset + '/' + city):
            imgLeft  = cv2.imread(leftImgDir + subset + '/' + city + '/' + file)
            imgRight = cv2.imread(rightImgDir + subset + '/' + city + '/' + file.replace('leftImg8bit', 'rightImg8bit'))

            os.makedirs(ourDispDir + subset + '/' + city, exist_ok=True)
            cv2.imwrite(ourDispDir + subset + '/' + city + '/' + file.replace('leftImg8bit', 'disparity'), getNewDisparity(imgLeft, imgRight))
            count += 1
            print(count)
            stdout.flush()