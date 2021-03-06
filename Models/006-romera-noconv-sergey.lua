local w, h, nClasses = ...
assert(w)
assert(h)
assert(nClasses)

require 'nn'
require 'cudnn'

local SpatialConvolution = cudnn.SpatialConvolution
local SpatialDilatedConvolution = nn.SpatialDilatedConvolution
local SpatialFullConvolution = cudnn.SpatialFullConvolution
local ReLU = cudnn.ReLU
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
local SpatialMaxPooling = cudnn.SpatialMaxPooling

collectgarbage()

local function downsampler(inChannels, outChannels, dropoutProb, noBN)
    dropoutProb = dropoutProb or 0

    return nn.Sequential()
        :add(noBN and nn.Identity() or SpatialBatchNormalization(inChannels))
        :add(noBN and nn.Identity() or ReLU(true))
        :add((noBN or dropoutProb == 0) and nn.Identity() or nn.SpatialDropout(dropoutProb))
        :add(nn.Concat(2)
            :add(SpatialMaxPooling(2,2, 2,2, 0,0))
            :add(SpatialConvolution(inChannels, outChannels-inChannels, 3,3, 2,2, 1,1)))
end

local function upsampler(inChannels, outChannels, noBN)
    return nn.Sequential()
        :add(noBN and nn.Identity() or SpatialBatchNormalization(inChannels))
        :add(noBN and nn.Identity() or ReLU(true))
        :add(SpatialFullConvolution(inChannels, outChannels, 3,3, 2,2, 1,1, 1,1))
end

local function nonBt1D(inChannels, dropoutProb, dilation)
    dilation = dilation or 1
    dropoutProb = dropoutProb or 0

    local convs = nn.Sequential()
        :add(SpatialBatchNormalization(inChannels))
        :add(ReLU(true))
        :add(SpatialConvolution(inChannels, inChannels, 3,1, 1,1, 1,0))
        :add(ReLU(true))
        :add(SpatialConvolution(inChannels, inChannels, 1,3, 1,1, 0,1))

        :add(SpatialBatchNormalization(inChannels))
        :add(ReLU(true))
        :add(dropoutProb ~= 0 and nn.SpatialDropout(dropoutProb) or nn.Identity())

    if dilation == 1 then convs
        :add(SpatialConvolution(inChannels, inChannels, 3,1, 1,1, 1,0))
        :add(ReLU(true))
        :add(SpatialConvolution(inChannels, inChannels, 1,3, 1,1, 0,1))
    else convs
        :add(SpatialDilatedConvolution(inChannels, inChannels, 3,1, 1,1, dilation,0, dilation,dilation))
        :add(ReLU(true))
        :add(SpatialDilatedConvolution(inChannels, inChannels, 1,3, 1,1, 0,dilation, dilation,dilation))
    end

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(convs))
        :add(nn.CAddTable())
        :add(ReLU(true))
end

local model = nn.Sequential()

model:add(downsampler( 3, 16, 0.0 , true))
model:add(downsampler(16, 64, 0.03))

model:add(nonBt1D(64, 0.03))
model:add(nonBt1D(64, 0.03))
model:add(nonBt1D(64, 0.03))
model:add(nonBt1D(64, 0.03))
model:add(nonBt1D(64, 0.03))

model:add(downsampler(64, 128, 0.3))

-- convs go here

model:add(upsampler(128, 64))

model:add(nonBt1D(64))
model:add(nonBt1D(64))

model:add(upsampler(64, 16))

model:add(nonBt1D(16))
model:add(nonBt1D(16))

model:add(SpatialFullConvolution(16, nClasses, 3,3, 2,2, 1,1, 1,1))

model:add(nn.View(nClasses, w*h):setNumInputDims(3))
model:add(nn.Transpose({2, 1}):setNumInputDims(2))

return model