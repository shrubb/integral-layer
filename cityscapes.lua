require 'paths'
require 'image'

cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local cityscapes = {}

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

cityscapes.mean = torch.FloatTensor{0.28470638394356, 0.32577008008957, 0.28766867518425}
cityscapes.std  = torch.FloatTensor{0.18671783804893, 0.1899059265852,  0.18665011227131}

-- Time for loading 1 picture:
-- OpenCV: 0.1926669716835 seconds
-- Torch : 0.15436439037323 seconds
function cityscapes.loadSample(files)
    local imagePath  = cityscapes.relative .. 'leftImg8bit/' .. files.image
    local labelsPath = cityscapes.relative .. 'gtFine/' .. files.labels
    
    -- load image
    local img = image.loadPNG(imagePath)
    -- normalize image globally
    for ch = 1,3 do
        img[ch]:add(-cityscapes.mean[ch])
        img[ch]:div(cityscapes.std[ch])
    end

    -- load labels
    local labelsOriginal = cv.imread{labelsPath, cv.IMREAD_GRAYSCALE}
    local labelsTorch    = torch.ByteTensor(labelsOriginal:size()):fill(255)
    -- shift labels according to
    -- https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L61
    local classMap = {
        7, 8, 11, 12, 13, 17, 19, 20, 21, 
        22, 23, 24, 25, 26, 27, 28, 31, 32, 33}
    for torchClass, originalClass in ipairs(classMap) do
        labelsTorch[labelsOriginal:eq(originalClass)] = torchClass
    end

    return img, labelsTorch
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

function cityscapes.renderLabels(labels, img, blendCoeff)
    
    local retval = torch.FloatTensor(3, 1024, 2048):zero()
    for label, color in ipairs(labelToColor) do
        local mask = nn.utils.addSingletonDimension(labels:eq(label))
        for ch = 1,3 do
            retval[ch][mask] = color[ch]
        end
    end
    
    if img then
        local labelsBlendCoeff = blendCoeff or 0.62
        retval:mul(labelsBlendCoeff)
        
        img = img:clone()
        img:add(-img:min())
        img:div(img:max())
        img:mul(1 - labelsBlendCoeff)
        retval:add(img)
    end
    
    return retval
end

return cityscapes