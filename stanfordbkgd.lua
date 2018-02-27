require 'paths'
require 'image'

cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local dataset = {}

dataset.mean = torch.FloatTensor{0.48159265518188, 0.49047175049782, 0.47479340434074}
dataset.std  = torch.FloatTensor{0.24885559082031, 0.24857203662395, 0.2740965783596}
dataset.dsize = {320, 240}
dataset.nClasses = 8
-- precomputed class frequencies
dataset.classProbs = torch.FloatTensor {
    0.1473,
    0.1412,
    0.2240,
    0.0698,
    0.0410,
    0.2266,
    0.0136,
    0.1365
}
-- dataset.classWeights = dataset.classProbs:clone():pow(-1/2.5)
dataset.classWeights = dataset.classProbs:clone():add(1.10):log():pow(-1)

local function loadData(dataset)
    dataset.labels = torch.load(dataset.relative .. 'labels_th.t7'):add(1):byte()
    dataset.labels[dataset.labels:eq(0)] = 255

    dataset.images = torch.FloatTensor(dataset.labels:size(1), 3, 240, 320)
    local imagePath = dataset.relative .. 'images_resized/%04d.jpg'

    for k = 0,714 do
        dataset.images[k+1]:copy(image.load(imagePath:format(k)))
    end

    for ch = 1,3 do
        local mean, std = dataset.mean[ch], dataset.std[ch]
        dataset.images[{{}, ch}]:add(-mean):div(std)
    end

    collectgarbage()
end

function dataset.loadNames(kind, foldIdx) -- `kind` is 'train' or 'val', `foldIdx` is 1 to 5
    --[[ returns:
        {1, 2, 3, ..., 571, 572}
    --]]
    local nFolds = 5
    foldIdx = foldIdx or 1
    assert(1 <= foldIdx and foldIdx <= 5)
    
    if not dataset.images or not dataset.labels then
        loadData(dataset)
    end

    local retval = {}
    local foldSize = 715 / nFolds
    for i = 1,715 do
        local isInTrainFold = (i <= foldSize*(foldIdx-1) or i > foldSize*foldIdx)
        if (kind == 'train') and     isInTrainFold or
           (kind ==   'val') and not isInTrainFold then
            table.insert(retval, i)
        end
    end

    return retval
end

function dataset.loadSample(idx)
    return dataset.images[idx]:clone(), dataset.labels[idx]:clone()
end

local labelToColor = {
    [1] = {0.0, 1.0, 1.0}, -- sky
    [2] = {1.0, 1.0, 0.0}, -- tree
    [3] = {0.5, .25, 0.5}, -- road
    [4] = {0.0, 1.0, 0.0}, -- grass
    [5] = {0.0, 0.0, 1.0}, -- water
    [6] = {0.5, 0.0, 0.0}, -- building
    [7] = {1.0, 1.0, 1.0}, -- mountain
    [8] = {1.0, 0.0, 0.0}  -- foreground object
}

require 'nn'

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

function dataset.calcClassProbs()
    local counts = torch.DoubleTensor(dataset.nClasses):zero()

    if not dataset.labels then
        loadData(dataset)
    end

    for c = 1,dataset.nClasses do
        counts[c] = dataset.labels:eq(c):sum()
    end

    return counts:div(counts:sum()):float()
end

ffi = require 'ffi'

local C_lib = ffi.load('C/lib/libcityscapes-c.so')

ffi.cdef [[
void updateConfusionMatrix(
    long *confMatrix, long *predictedLabels,
    unsigned char *labels, int numPixels,
    int nClasses);
]]

function dataset.updateConfusionMatrix(confMatrix, predictedLabels, labels)
    -- confMatrix:      long, 19x19
    -- predictedLabels: long, 128x256
    -- labels:          byte, 128x256
    assert(predictedLabels:type() == 'torch.LongTensor')
    assert(labels         :type() == 'torch.ByteTensor')
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

function dataset.calcAcc(confMatrix)
    -- returns: pixel accuracy, mean class accuracy
    local classAcc = confMatrix:sum(2):squeeze():float() -- total GT pixels of each class
    local totalPixels = confMatrix:sum()
    local totalCorrect = 0
    for c = 1,dataset.nClasses do
        local classCorrect = confMatrix[{c, c}]
        totalCorrect = totalCorrect + classCorrect
        classAcc[c] = classCorrect / classAcc[c]
    end

    return totalCorrect / totalPixels, validAverage(classAcc)
end

return dataset
