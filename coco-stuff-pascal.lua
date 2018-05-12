require 'paths'
require 'image'

cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local dataset = {}

dataset.mean = torch.FloatTensor{0.47049099206924, 0.44712817668915, 0.40792506933212} * 255
dataset.std  = torch.FloatTensor{0.27841508388519, 0.2738970220089, 0.28862416744232} * 255
dataset.dsize = {512, 512}
dataset.nClasses = 21 -- all PASCAL VOC 2012 segmentation classes + "background"

-- precomputed class frequencies
dataset.classProbs = torch.FloatTensor {
    0.0043220477268013,
    0.0020987215202244,
    0.0020858630336653,
    0.0035373760958584,
    0.0020328517319741,
    0.0099920090565802,
    0.0080248412006723,
    0.0081511776939684,
    0.0085582885561291,
    0.0035301265201594,
    0.0411748268623,
    0.0059333851547132,
    0.0047445264454333,
    0.0064907405201876,
    0.11959019917812,
    0.0031072816502775,
    0.0025029589748986,
    0.0077598138855773,
    0.010089911488045,
    0.0046456479120634,
    0.74162740479235,
}
-- dataset.classWeights = dataset.classProbs:clone():pow(-1/2.5)
dataset.classWeights = dataset.classProbs:clone():add(1.10):log():pow(-1)
-- add "unlabeled" class with zero weight
dataset.classWeights = torch.cat(dataset.classWeights, torch.FloatTensor{0})

function dataset.loadNames(kind)
    --[[
        `kind`: 'train', 'val' or 'test'

        returns:
        {
            {image  = 'train2017/000000000071.jpg',
             labels = 'annotations/train2017_pascal/000000000071.png'},
            ...,
        }
    --]]
    
    assert(dataset.relative:sub(-1, -1) == '/')

    local retval = {} 
    local imageBase = kind .. '2017/'
    local labelBase = 'annotations/' .. kind .. '2017_pascal/'
    
    for gtFile in paths.iterfiles(dataset.relative .. labelBase) do
        table.insert(retval, 
            {
                image  = imageBase .. gtFile:gsub('png', 'jpg'),
                labels = labelBase .. gtFile,
            })
    end
    
    -- doesn't really matter, but just in case
    table.sort(retval, function(a,b) return a.image < b.image end)
    return retval
end

require 'xlua'
function dataset.calcMean(files)
    local retval = torch.DoubleTensor{0, 0, 0}
    
    for i,sample in ipairs(files) do
        local imgFile = dataset.relative .. sample.image
        local localMean = cv.imread{imgFile, cv.IMREAD_COLOR}:view(-1, 3):float():div(255):mean(1)
        retval:add(localMean:squeeze():double())

        xlua.progress(i, #files)
    end
    
    retval[1], retval[3] = retval[3], retval[1]
    return retval:div(#files):float()
end

function dataset.calcStd(files, mean)
    local retval = torch.DoubleTensor{0, 0, 0}

    for i,sample in ipairs(files) do
        local imgFile = dataset.relative .. sample.image
        local img = cv.imread{imgFile, cv.IMREAD_COLOR}:view(-1, 3):float():div(255)
        local squareDiff = img:add(-mean:view(1,3):expandAs(img)):pow(2):mean(1)
        retval:add(squareDiff:squeeze():double())

        xlua.progress(i, #files)
    end

    retval[1], retval[3] = retval[3], retval[1]
    return retval:div(#files):sqrt():float()
end

-- `crop`: false/nil, 'random' or 'center'
function dataset.loadSample(files, crop, augment)
    local imagePath  = dataset.relative .. files.image
    local labelsPath = dataset.relative .. files.labels

    local labels = cv.imread{labelsPath, cv.IMREAD_GRAYSCALE}
    local img = cv.imread{imagePath, cv.IMREAD_COLOR}

    if augment then
        local flip = torch.random(2) == 1
        local maybeFlipMatrix = torch.eye(3):double()
        if flip then
            maybeFlipMatrix[{1,1}] = -1
            maybeFlipMatrix[{1,3}] = labels:size(2)
        end
        
        local angle = (math.random() * 2 - 1) * 4.5
        local scaleFactor = math.random() * 1.5 + 0.5
        local imageCenter = {labels:size(2) / 2, labels:size(1) / 2}
        local rotationMatrix = torch.eye(3):double()
        rotationMatrix[{{1,2}}]:copy(cv.getRotationMatrix2D{imageCenter, angle, scaleFactor})
        
        local transformationMatrix = torch.mm(maybeFlipMatrix, rotationMatrix)[{{1,2}}]
        
        img = cv.warpAffine{
            img, transformationMatrix, flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT, borderValue={nClasses+1}}
        labels = cv.warpAffine{
            labels, transformationMatrix, flags=cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT, borderValue={nClasses+1}}
        
        if torch.random(2) == 1 then
            local blurSigma = math.random() * 2.3 + 0.7
            cv.GaussianBlur{img, {5, 5}, blurSigma, dst=img, borderType=cv.BORDER_REFLECT}
        end
    end

    if crop then
        -- determine crop boundaries
        local large,small = img:size(1), img:size(2)
        local heightIsLarge = true
        if large < small then
            large,small = small,large
            heightIsLarge = not heightIsLarge
        end

        local startIdx
        if crop == 'random' then
            startIdx = torch.random(1, 1+large-small)
        elseif crop == 'center' then
            startIdx = 1 + math.floor((large-small) / 2)
        else
            error('Unknown crop type')
        end
        local narrowTable = {startIdx, startIdx+small-1}

        local cropIndices = {{}, {}, {}}
        cropIndices[heightIsLarge and 1 or 2] = narrowTable

        -- do the crop
        img = img[cropIndices]
        cropIndices[3] = nil
        labels = labels[cropIndices]

        -- resize the cropped image to `dataset.dsize`
        local interpolation = (dataset.dsize[1] > small) and cv.INTER_LINEAR or cv.INTER_AREA
        img = cv.resize{img, dataset.dsize, interpolation=interpolation}
        labels = cv.resize{labels, dataset.dsize, interpolation=cv.INTER_NEAREST}
    end

    cv.cvtColor{img, img, cv.COLOR_BGR2RGB}
    img = img:permute(3,1,2):float()
    -- normalize image globally
    for ch = 1,3 do
        img[ch]:add(-dataset.mean[ch])
        img[ch]:mul(1/dataset.std[ch])
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

function dataset.renderLabels(labels, img, blendCoeff)
    
    local retval = torch.FloatTensor(3, dataset.dsize[2], dataset.dsize[1]):zero()
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

function dataset.calcClassProbs(files)
    local counts = torch.DoubleTensor(dataset.nClasses):zero()

    for i,sample in ipairs(files) do
        local labelsPath = dataset.relative .. sample.labels
        local labels = cv.imread{labelsPath, cv.IMREAD_GRAYSCALE}
        local scaleFactor = 640 / math.max(labels:size(1), labels:size(2))

        local mask = torch.ByteTensor(labels:size())

        local min = labels:min()
        torch.eq(mask, labels, 93)
        labels[mask] = 0
        local max = labels:max()

        for class = min,max do
            torch.eq(mask, labels, class)
            counts[class] = counts[class] + scaleFactor * mask:sum()
        end
        xlua.progress(i, #files)
    end

    return counts:div(counts:sum()):float()
end

function dataset.labelsToEval(labels)
    return labels + 91
end

ffi = require 'ffi'

local C_lib = ffi.load('C/lib/libcityscapes-c.so')

ffi.cdef [[
void updateConfusionMatrix(
    long *confMatrix, long *predictedLabels,
    long *labels, int numPixels,
    int nClasses);
]]

function dataset.updateConfusionMatrix(confMatrix, predictedLabels, labels)
    -- confMatrix:      long, nClasses x nClasses
    -- predictedLabels: long, dsize
    -- labels:          long, dsize
    assert(confMatrix:type() == 'torch.LongTensor')
    assert(confMatrix:size(1) == confMatrix:size(2) and confMatrix:size(1) == dataset.nClasses)

    assert(predictedLabels:type() == 'torch.LongTensor')
    assert(labels:type() == 'torch.LongTensor')
    assert(predictedLabels:nElement() == labels:nElement())
    assert(predictedLabels:isContiguous() and labels:isContiguous())

    C_lib.updateConfusionMatrix(
        torch.data(confMatrix), torch.data(predictedLabels),
        torch.data(labels), predictedLabels:nElement(),
        dataset.nClasses)
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

function dataset.calcIoU(confMatrix)
    local IoUclass = torch.FloatTensor(dataset.nClasses)
    for classIdx = 1,IoUclass:nElement() do
        local TP = confMatrix[{classIdx, classIdx}]
        local FN = confMatrix[{classIdx, {}}]:sum() - TP
        local FP = confMatrix[{{}, classIdx}]:sum() - TP
        IoUclass[classIdx] = TP / (TP + FP + FN)
    end

    return validAverage(IoUclass), IoUclass
end

return dataset
