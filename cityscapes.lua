require 'paths'
require 'image'

cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local cityscapes = {}

cityscapes.mean = torch.FloatTensor{0.28470638394356, 0.32577008008957, 0.28766867518425}
cityscapes.std  = torch.FloatTensor{0.18671783804893, 0.1899059265852,  0.18665011227131}
cityscapes.dsize = {512, 256}
cityscapes.nClasses = 19
-- precomputed class frequencies
cityscapes.classProbs = torch.FloatTensor {
    0.36869695782661,
    0.060849856585264,
    0.22824048995972,
    0.0065539856441319,
    0.0087727159261703,
    0.012273414991796,
    0.0020779478363693,
    0.0055127013474703,
    0.1592865139246,
    0.011578181758523,
    0.040189824998379,
    0.012189572677016,
    0.0013512192526832,
    0.069945447146893,
    0.0026745572686195,
    0.0023519159294665,
    0.0023290426470339,
    0.00098657899070531,
    0.0041390685364604,
}
-- cityscapes.classWeights = cityscapes.classProbs:clone():pow(-1/2.5)
cityscapes.classWeights = cityscapes.classProbs:clone():add(1.10):log():pow(-1)

function cityscapes.loadNames(kind) -- `kind` is 'train' or 'val'
    --[[ returns:
        {
            {'train/aachen/aachen_000000_000019_leftImg8bit.png',
             'train/aachen/aachen_000000_000019_gtFine_labelIds.png},
            ...,
        }
    --]]
    
    local retval = {}
    local labelBase = cityscapes.relative .. 'gtFine/' .. kind .. '/'
    local imageBase = cityscapes.relative .. 'leftImg8bit/' .. kind .. '/'
    
    -- iterate over cities
    for city in paths.iterdirs(imageBase) do
        for file in paths.iterfiles(imageBase .. city) do
            local gtFile = file:gsub('leftImg8bit', 'gtFine_labelIds')
            table.insert(retval, 
                {
                    image  = kind .. '/' .. city .. '/' .. file,
                    labels = kind .. '/' .. city .. '/' .. gtFile
                })
        end
    end
    
    table.sort(retval, function(a,b) return a.image < b.image end)
    return retval
end

function cityscapes.calcMean(files)
    local retval = torch.FloatTensor{0, 0, 0}
    
    for i,names in ipairs(files) do
        local imgFile = cityscapes.relative .. 'leftImg8bit/' .. names.image
        retval:add(cv.imread{imgFile, cv.IMREAD_COLOR}:view(-1, 3):float():div(255):mean(1):squeeze())

        if i % 100 == 0 then print(i); collectgarbage() end
    end
    
    return retval:div(#files)
end

function cityscapes.calcStd(files, mean)
    local retval = torch.FloatTensor{0, 0, 0}
    
    for i,names in ipairs(files) do
        local imgFile = cityscapes.relative .. 'leftImg8bit/' .. names.image
        local img = cv.imread{imgFile, cv.IMREAD_COLOR}:view(-1, 3):float():div(255)
        local squareDiff = img:add(-mean:view(1,3):expandAs(img)):pow(2):mean(1):squeeze()
        retval:add(squareDiff)

        if i % 100 == 0 then print(i); collectgarbage() end
    end
    
    return retval:div(#files):sqrt()
end

-- Time for loading 1 picture:
-- OpenCV: 0.1926669716835 seconds
-- Torch : 0.15436439037323 seconds
function cityscapes.loadSample(files)
    local imagePath  = cityscapes.relative .. 'leftImg8bit/' .. files.image
    local labelsPath = cityscapes.relative .. 'gtFine/' .. files.labels

    -- load image
    local img = cv.imread{imagePath, cv.IMREAD_COLOR}
    cv.cvtColor{img, img, cv.COLOR_BGR2RGB}
    if img:size(1) ~= cityscapes.dsize[2] or 
       img:size(2) ~= cityscapes.dsize[1] then
        img = cv.resize{img, cityscapes.dsize, interpolation=cv.INTER_CUBIC}
    end
    img = img:float():div(255):permute(3,1,2):clone()

    -- normalize image globally
    for ch = 1,3 do
        img[ch]:add(-cityscapes.mean[ch])
        img[ch]:div(cityscapes.std[ch])
    end

    -- load labels
    local labels = cv.imread{labelsPath, cv.IMREAD_GRAYSCALE}
    if labels:size(1) ~= cityscapes.dsize[2] or
       labels:size(2) ~= cityscapes.dsize[1] then
        labelsOriginal = cv.resize{labels, cityscapes.dsize, interpolation=cv.INTER_NEAREST}
    
        if labels:size(1) == 512 and labels:size(2) == 1024 then
            labels = labelsOriginal
        else
            local labelsTorch = torch.ByteTensor(labelsOriginal:size()):fill(255)
            -- shift labels according to
            -- https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L61
            local classMap = {
                7, 8, 11, 12, 13, 17, 19, 20, 21, 
                22, 23, 24, 25, 26, 27, 28, 31, 32, 33}
            for torchClass, originalClass in ipairs(classMap) do
                labelsTorch[labelsOriginal:eq(originalClass)] = torchClass
            end

            labels = labelsTorch
        end
    end

    return img, labels
end

local labelToColor = {
    [ 1] = {128, 64,128}, -- road
    [ 2] = {244, 35,232}, -- sidewalk
    [ 3] = { 70, 70, 70}, -- building
    [ 4] = {102,102,156}, -- wall
    [ 5] = {190,153,153}, -- fence
    [ 6] = {153,153,153}, -- pole
    [ 7] = {250,170, 30}, -- traffic light
    [ 8] = {220,220,  0}, -- traffic sign
    [ 9] = {107,142, 35}, -- vegetation
    [10] = {152,251,152}, -- terrain
    [11] = { 70,130,180}, -- sky
    [12] = {220, 20, 60}, -- person
    [13] = {255,  0,  0}, -- rider
    [14] = {  0,  0,142}, -- car
    [15] = {  0,  0, 70}, -- truck
    [16] = {  0, 60,100}, -- bus
    [17] = {  0, 80,100}, -- train
    [18] = {  0,  0,230}, -- motorcycle
    [19] = {119, 11, 32}, -- bicycle
}

for label, color in ipairs(labelToColor) do
    -- color[3], color[1] = color[1], color[3]
    for k = 1,3 do
        color[k] = color[k] / 255
    end
end

require 'nn'

function cityscapes.renderLabels(labels, img, blendCoeff)
    
    local retval = torch.FloatTensor(3, cityscapes.dsize[2], cityscapes.dsize[1]):zero()
    for label, color in ipairs(labelToColor) do
        local mask = nn.utils.addSingletonDimension(labels:eq(label))
        for ch = 1,3 do
            retval[ch][mask] = color[ch]
        end
    end
    
    if img then
        local labelsBlendCoeff = blendCoeff or 0.62
        retval:mul(labelsBlendCoeff)
        
        local MIN = img:min()
        local MAX = img:max() - MIN
        retval:add((1 - labelsBlendCoeff) / MAX, img)
        retval:add(- MIN * (1 - labelsBlendCoeff) / MAX)
    end
    
    return retval
end

function cityscapes.calcClassProbs(trainFiles)
    local counts = torch.DoubleTensor(19):zero()

    local classMap = {
        7, 8, 11, 12, 13, 17, 19, 20, 21, 
        22, 23, 24, 25, 26, 27, 28, 31, 32, 33}
    for i, files in ipairs(trainFiles) do
        local labelsPath = cityscapes.relative .. 'gtFine/' .. files.labels
        local labels = cv.imread{labelsPath, cv.IMREAD_GRAYSCALE}
        for classTorch, classOrigin in ipairs(classMap) do
            counts[classTorch] = counts[classTorch] + labels:eq(classOrigin):sum()
        end
        if i % 100 == 0 then print(i); collectgarbage() end
    end

    return counts:div(counts:sum()):float()
end

function cityscapes.labelsToEval(labels)
    local retval = torch.ByteTensor(cityscapes.dsize[2], cityscapes.dsize[1])

    local classMap = {
        7, 8, 11, 12, 13, 17, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 31, 32, 33}
    for torchClass, evalClass in pairs(classMap) do
        retval[labels:eq(torchClass)] = evalClass
        -- print('Setting ' .. evalClass)
    end

    return retval
end

ffi = require 'ffi'

local C_lib = ffi.load('C/lib/libcityscapes-c.so')

ffi.cdef [[
void updateConfusionMatrix(
    long *confMatrix, long *predictedLabels,
    unsigned char *labels, int numPixels,
    int nClasses);
]]

function cityscapes.updateConfusionMatrix(confMatrix, predictedLabels, labels)
    -- confMatrix:      long, 19x19
    -- predictedLabels: long, 128x256
    -- labels:          byte, 128x256
    C_lib.updateConfusionMatrix(
        torch.data(confMatrix), torch.data(predictedLabels),
        torch.data(labels), predictedLabels:nElement(),
        cityscapes.nClasses)
end

local classCategories = {
    {1, 2},                   -- flat
    {3, 4, 5},                -- construction
    {6, 7, 8},                -- object
    {9, 10},                  -- nature
    {11},                     -- sky
    {12, 13},                 -- human
    {14, 15, 16, 17, 18, 19}, -- vehicle
}

local function validAverage(t)
    local sum, count = 0, 0
    t:apply(
        function(x)
            if x == x then -- if x is not nan
                sum = sum + x
                count = count + 1
            end
        end
    )

    return count > 0 and (sum / count) or 0
end

function cityscapes.calcIoU(confMatrix)
    local IoUclass = torch.FloatTensor(19)
    for classIdx = 1,IoUclass:nElement() do
        local TP = confMatrix[{classIdx, classIdx}]
        local FN = confMatrix[{classIdx, {}}]:sum() - TP
        local FP = confMatrix[{{}, classIdx}]:sum() - TP
        IoUclass[classIdx] = TP / (TP + FP + FN)
    end

    local IoUcategory = torch.FloatTensor(#classCategories)
    for categoryIdx, category in ipairs(classCategories) do
        local TP = 0
        for _,classIdx in ipairs(category) do
            TP = TP + confMatrix[{classIdx, classIdx}]
        end

        local FN, FP = -TP, -TP
        for _,classIdx in ipairs(category) do
            FN = FN + confMatrix[{classIdx, {}}]:sum()
            FP = FP + confMatrix[{{}, classIdx}]:sum()
        end

        IoUcategory[categoryIdx] = TP / (TP + FP + FN)
    end

    return validAverage(IoUclass), validAverage(IoUcategory), IoUclass, IoUcategory
end

return cityscapes
