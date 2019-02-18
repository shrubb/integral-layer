-- ERFNet^-
local w, h, nClasses = ...
assert(w)
assert(h)
assert(nClasses)

require 'nn'
require 'cudnn'
require 'IntegralZeroPadding'
local BoxConvolution = IntegralSmartNorm

local SpatialConvolution = cudnn.SpatialConvolution
local SpatialDilatedConvolution = cudnn.SpatialDilatedConvolution
local SpatialFullConvolution = cudnn.SpatialFullConvolution
local ReLU = cudnn.ReLU
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
local SpatialMaxPooling = cudnn.SpatialMaxPooling

collectgarbage()

local function downsampler(inChannels, outChannels, dropoutProb)
    dropoutProb = dropoutProb or 0

    return nn.Sequential()
        :add(nn.Concat(2)
            :add(SpatialMaxPooling(2,2, 2,2, 0,0))
            :add(SpatialConvolution(inChannels, outChannels-inChannels, 3,3, 2,2, 1,1)))
        :add(SpatialBatchNormalization(outChannels))
        :add(dropoutProb ~= 0 and nn.SpatialDropout(dropoutProb) or nn.Identity())
        :add(ReLU(true))
end

local function upsampler(inChannels, outChannels)
    return nn.Sequential()
        :add(SpatialFullConvolution(inChannels, outChannels, 3,3, 2,2, 1,1, 1,1))
        :add(SpatialBatchNormalization(outChannels))
        :add(ReLU(true))
end

local function nonBt1D(inChannels, dropoutProb, dilation)
    dilation = dilation or 1
    dropoutProb = dropoutProb or 0

    local convs = nn.Sequential()
        :add(SpatialConvolution(inChannels, inChannels, 3,1, 1,1, 1,0))
        :add(ReLU(true))
        :add(SpatialConvolution(inChannels, inChannels, 1,3, 1,1, 0,1))
        :add(SpatialBatchNormalization(inChannels))
        :add(ReLU(true))

    if dilation == 1 then convs
        :add(SpatialConvolution(inChannels, inChannels, 3,1, 1,1, 1,0))
        :add(ReLU(true))
        :add(SpatialConvolution(inChannels, inChannels, 1,3, 1,1, 0,1))
    else convs
        :add(SpatialDilatedConvolution(inChannels, inChannels, 3,1, 1,1, dilation,0, dilation,dilation))
        :add(ReLU(true))
        :add(SpatialDilatedConvolution(inChannels, inChannels, 1,3, 1,1, 0,dilation, dilation,dilation))
    end

    convs
        :add(SpatialBatchNormalization(inChannels))
        :add(dropoutProb ~= 0 and nn.SpatialDropout(dropoutProb) or nn.Identity())

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(convs))
        :add(nn.CAddTable())
        :add(ReLU(true))
end

local function bottleneckBox(inChannels, numBoxes, dropoutProb, h, w)
    assert(inChannels % numBoxes == 0)
    local btChannels = inChannels / numBoxes

    dropoutProb = dropoutProb or 0

    local mainBranch = nn.Sequential()
        :add(SpatialConvolution(inChannels, btChannels, 1,1, 1,1))
        :add(SpatialBatchNormalization(btChannels))

        :add(BoxConvolution(btChannels, numBoxes, h, w))
        
        -- :add(SpatialConvolution(inChannels, inChannels, 1,1, 1,1))
        :add(SpatialBatchNormalization(inChannels))
        
        :add(dropoutProb ~= 0 and nn.SpatialDropout(dropoutProb) or nn.Identity())

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(mainBranch))
        :add(nn.CAddTable())
        :add(ReLU(true))
end

local model = nn.Sequential()

model:add(downsampler( 3, 16, 0.0 ))
model:add(downsampler(16, 64, 0.03))

model:add(nonBt1D(64, 0.03))

model:add(downsampler(64, 128, 0.3))

model:add(nonBt1D(128, 0.3,  2))
model:add(nonBt1D(128, 0.3,  4))


model:add(nonBt1D(128, 0.3,  2))
model:add(nonBt1D(128, 0.3,  4))

model:add(upsampler(128, 64))
model:add(nonBt1D(64))

model:add(upsampler(64, 16))
model:add(nonBt1D(16))

model:add(SpatialFullConvolution(16, nClasses+1, 3,3, 2,2, 1,1, 1,1))

return model, {}
