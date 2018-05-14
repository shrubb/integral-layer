require 'paths'
require 'image'

cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local cityscapes = {}

cityscapes.mean = torch.FloatTensor{0.49384835362434, 0.45716208219528, 0.43320959806442} * 255
cityscapes.std  = torch.FloatTensor{0.29184621572495, 0.28711155056953, 0.27946510910988} * 255
cityscapes.dsize = {728, 528}
cityscapes.nClasses = 37
-- precomputed class frequencies
cityscapes.classProbs = torch.FloatTensor {
    0.27086761593819,   
    0.23402366042137,   
    0.028874693438411,  
    0.041853778064251,  
    0.1166203096509,    
    0.033175464719534,  
    0.074613481760025,  
    0.026766657829285,  
    0.031676657497883,  
    0.0061814510263503, 
    0.0068180141970515, 
    0.0091995866969228, 
    0.0040137958712876, 
    0.023720165714622,  
    0.003696559695527,  
    0.010313637554646,  
    0.0055921101011336, 
    0.0067539755254984, 
    0.0066420380026102, 
    0.00040074778371491,    
    0.0031655754428357, 
    0.010683725588024,  
    0.0037833638489246, 
    0.003201603423804,  
    0.0021454386878759, 
    0.0033536942210048, 
    0.0018207511166111, 
    0.00026947970036417,    
    0.0060982936993241, 
    0.0064957421272993, 
    0.0012123491615057, 
    0.0011886606225744, 
    0.0030936673283577, 
    0.0045012501068413, 
    0.0028987068217248, 
    0.0020554747898132, 
    0.0022278418764472,
}
-- cityscapes.classWeights = cityscapes.classProbs:clone():pow(-1/2.5)
cityscapes.classWeights = cityscapes.classProbs:clone():add(1.10):log():pow(-1)
-- add "unlabeled" class with zero weight
cityscapes.classWeights = torch.cat(cityscapes.classWeights, torch.FloatTensor{0})

function cityscapes.loadNames(kind)
    --[[
        `kind`: 'train' or 'val'
        
        returns:
        {5050, 5051, 5052, ..., }
    --]]
    
    local retval = {}
    local first, last

    if kind == 'train' then
        first, last = 5051, 10335
    elseif kind == 'val' then
        first, last = 1, 5050
    end

    for k = first,last do
        local labelsDir = ('%slabels/%s/img-%06d.png')
            :format(cityscapes.relative, kind, k)
        local imageDir = ('%simages/%s/img-%06d.jpg')
            :format(cityscapes.relative, kind, k > 5050 and (k-5050) or k)
        
        assert(paths.filep(labelsDir), labelsDir .. ' wasn\'t found')
        assert(paths.filep(imageDir), imageDir .. ' wasn\'t found')

        table.insert(retval, {image=imageDir, labels=labelsDir})
    end

    return retval
end

function cityscapes.calcMean(files)
    local retval = torch.FloatTensor{0, 0, 0}
    
    for i,file in ipairs(files) do
        local imgFile = cityscapes.relative .. file.image
        retval:add(cv.imread{imgFile, cv.IMREAD_COLOR}:view(-1, 3):float():div(255):mean(1):squeeze())

        if i % 100 == 0 then print(i); collectgarbage() end
    end
    
    retval[1], retval[3] = retval[3], retval[1]
    return retval:div(#files)
end

function cityscapes.calcStd(files, mean)
    local retval = torch.FloatTensor{0, 0, 0}
    
    for i,file in ipairs(files) do
        local imgFile = cityscapes.relative .. file.image
        local img = cv.imread{imgFile, cv.IMREAD_COLOR}:view(-1, 3):float():div(255)
        local squareDiff = img:add(-mean:view(1,3):expandAs(img)):pow(2):mean(1):squeeze()
        retval:add(squareDiff)

        if i % 100 == 0 then print(i); collectgarbage() end
    end
    
    retval[1], retval[3] = retval[3], retval[1]
    return retval:div(#files):sqrt()
end

function cityscapes.loadSample(file, _, augment)
    local imagePath  = cityscapes.relative .. file.image
    local labelsPath = cityscapes.relative .. file.labels

    -- load image
    local img = cv.imread{imagePath, cv.IMREAD_COLOR}[{{2,-2}, {2,-2}}]
    if img:size(1) ~= cityscapes.dsize[2] or 
       img:size(2) ~= cityscapes.dsize[1] then
        img = cv.resize{img, cityscapes.dsize, interpolation=cv.INTER_CUBIC}
    end

    -- load labels
    local labels = cv.imread{labelsPath, cv.IMREAD_ANYCOLOR}[{{2,-2}, {2,-2}}]
    assert(labels:nDimension() == 2)
    if labels:size(1) ~= cityscapes.dsize[2] or
       labels:size(2) ~= cityscapes.dsize[1] then
        labels = cv.resize{labels, cityscapes.dsize, interpolation=cv.INTER_NEAREST}
    end
    labels[labels:eq(0)] = cityscapes.nClasses+1

    if augment then
        local flip = torch.random(2) == 1
        local maybeFlipMatrix = torch.eye(3):double()
        if flip then
            maybeFlipMatrix[{1,1}] = -1
            maybeFlipMatrix[{1,3}] = labels:size(2)
        end
        
        local angle = (math.random() * 2 - 1) * 7
        local scaleFactor = math.random() * 0.24 + 0.88
        local imageCenter = {labels:size(2) / 2, labels:size(1) / 2}
        local rotationMatrix = torch.eye(3):double()
        rotationMatrix[{{1,2}}]:copy(cv.getRotationMatrix2D{imageCenter, angle, scaleFactor})
        
        local transformationMatrix = torch.mm(maybeFlipMatrix, rotationMatrix)[{{1,2}}]
        
        img = cv.warpAffine{
            img, transformationMatrix, flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT}
        labels = cv.warpAffine{
            labels, transformationMatrix, flags=cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT, borderValue={cityscapes.nClasses+1}}
        
        if torch.random(2) == 1 then
            local blurSigma = math.random() * 2.3 + 0.7
            cv.GaussianBlur{img, {5, 5}, blurSigma, dst=img, borderType=cv.BORDER_REFLECT}
        end
    end

    cv.cvtColor{img, img, cv.COLOR_BGR2RGB}
    img = img:permute(3,1,2):float()
    -- normalize image globally
    for ch = 1,3 do
        img[ch]:add(-cityscapes.mean[ch])
        img[ch]:mul(1/cityscapes.std[ch])
    end

    return img, labels:long()
end

local labelToColor = {
    {166,83,94},
    {206,182,242},
    {0,238,255},
    {170,255,0},
    {229,145,115},
    {242,0,97},
    {65,0,242},
    {38,153,145},
    {124,166,41},
    {77,60,57},
    {230,172,195},
    {0,27,102},
    {191,255,251},
    {242,206,61},
    {255,0,0},
    {102,0,54},
    {102,129,204},
    {0,230,153},
    {217,206,163},
    {191,48,48},
    {51,26,39},
    {0,92,230},
    {29,115,52},
    {102,77,26},
    {76,19,19},
    {242,61,182},
    {13,28,51},
    {40,51,38},
    {242,129,0},
    {86,45,89},
    {0,85,128},
    {115,140,105},
    {51,27,0},
    {143,48,191},
    {115,191,230},
    {195,230,172},
    {217,166,108},
    {129,105,140},
    {38,64,77},
    {41,64,16},
    {127,51,0}
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
    local counts = torch.DoubleTensor(cityscapes.nClasses):zero()

    for i,file in ipairs(trainFiles) do
        local labelsPath = cityscapes.relative .. file.labels
        local labels = cv.imread{labelsPath, cv.IMREAD_GRAYSCALE}
        for class = 1,37 do
            counts[class] = counts[class] + labels:eq(class):sum()
        end
        if i % 100 == 0 then print(i); collectgarbage() end
    end

    return counts:div(counts:sum()):float()
end

function cityscapes.labelsToEval(labels)
    return labels
end

ffi = require 'ffi'

local C_lib = ffi.load('C/lib/libcityscapes-c.so')

ffi.cdef [[
void updateConfusionMatrix(
    long *confMatrix, long *predictedLabels,
    long *labels, int numPixels,
    int nClasses);
]]

function cityscapes.updateConfusionMatrix(confMatrix, predictedLabels, labels)
    -- confMatrix:      long, 19x19
    -- predictedLabels: long, 128x256
    -- labels:          byte, 128x256
    assert(confMatrix:type() == 'torch.LongTensor')
    assert(confMatrix:size(1) == confMatrix:size(2) and confMatrix:size(1) == cityscapes.nClasses)

    assert(predictedLabels:type() == 'torch.LongTensor')
    assert(labels:type() == 'torch.LongTensor')
    assert(predictedLabels:nElement() == labels:nElement())
    assert(predictedLabels:isContiguous() and labels:isContiguous())

    C_lib.updateConfusionMatrix(
        torch.data(confMatrix), torch.data(predictedLabels),
        torch.data(labels), predictedLabels:nElement(),
        cityscapes.nClasses)
end

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
	-- returns: mean IoU, pixel acc, IoU by class
    local IoUclass = torch.FloatTensor(cityscapes.nClasses)
    local classAcc = torch.FloatTensor(cityscapes.nClasses)
    local pixelAcc = 0

    for classIdx = 1,IoUclass:nElement() do
        local TP = confMatrix[{classIdx, classIdx}]
        local FN = confMatrix[{classIdx, {}}]:sum() - TP
        local FP = confMatrix[{{}, classIdx}]:sum() - TP
        
        IoUclass[classIdx] = TP / (TP + FP + FN)
        classAcc[classIdx] = TP / (TP + FN)
        pixelAcc = pixelAcc + TP
    end

    pixelAcc = pixelAcc / confMatrix:sum()

    return validAverage(IoUclass), pixelAcc, validAverage(classAcc), IoUclass, classAcc
end

return cityscapes
