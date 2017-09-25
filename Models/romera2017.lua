local w, h, nClasses = ...
assert(w)
assert(h)
assert(nClasses)

require 'nn'
require 'nngraph'

local SpatialConvolution = nn.SpatialDilatedConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local ReLU = nn.ReLU
local SpatialBatchNormalization = nn.SpatialBatchNormalization

collectgarbage()

local function downsampler(inChannels, outChannels, dropoutProb)
    dropoutProb = dropoutProb or 0

    return nn.Sequential()
        :add(nn.Concat(2)
            :add(nn.SpatialMaxPooling(2,2, 2,2, 0,0))
            :add(SpatialConvolution(inChannels, outChannels-inChannels, 3,3, 2,2, 1,1)))
        :add(SpatialBatchNormalization(outChannels))
        :add(dropoutProb ~= 0 and nn.SpatialDropout(dropoutProb) or nn.Identity())
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

        :add(SpatialConvolution(inChannels, inChannels, 3,1, 1,1, dilation,0, dilation,dilation))
        :add(ReLU(true))
        :add(SpatialConvolution(inChannels, inChannels, 1,3, 1,1, 0,dilation, dilation,dilation))
        :add(SpatialBatchNormalization(inChannels))
        :add(dropoutProb ~= 0 and nn.SpatialDropout(dropoutProb) or nn.Identity())

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(convs))
        :add(nn.CAddTable())
        :add(nn.ReLU(true))
end

-- the following vars are intentionally global for debugging
inData = nn.Identity()()

downsample1 = downsampler( 3, 16, 0.0 )(inData)
downsample2 = downsampler(16, 64, 0.03)(downsample1)

nonBt1D01 = nonBt1D(64, 0.03)(downsample2)
nonBt1D02 = nonBt1D(64, 0.03)(nonBt1D01)
nonBt1D03 = nonBt1D(64, 0.03)(nonBt1D02)
nonBt1D04 = nonBt1D(64, 0.03)(nonBt1D03)
nonBt1D05 = nonBt1D(64, 0.03)(nonBt1D04)

downsample3 = downsampler(64, 128, 0.3)(nonBt1D05)

nonBt1D06 = nonBt1D(128, 0.3,  2)(downsample3)
nonBt1D07 = nonBt1D(128, 0.3,  4)(nonBt1D06)
nonBt1D08 = nonBt1D(128, 0.3,  8)(nonBt1D07)
nonBt1D09 = nonBt1D(128, 0.3, 16)(nonBt1D08)
nonBt1D10 = nonBt1D(128, 0.3,  2)(nonBt1D09)
nonBt1D11 = nonBt1D(128, 0.3,  4)(nonBt1D10)
nonBt1D12 = nonBt1D(128, 0.3,  8)(nonBt1D11)
nonBt1D13 = nonBt1D(128, 0.3, 16)(nonBt1D12)

upsample3 = SpatialFullConvolution(128, 64, 3,3, 2,2, 1,1, 1,1)(nonBt1D13)
upsample3Rect = ReLU(true)(SpatialBatchNormalization(64)(upsample3))
nonBt1D14 = nonBt1D(64)(upsample3Rect)
nonBt1D15 = nonBt1D(64)(nonBt1D14)

upsample2 = SpatialFullConvolution( 64, 16, 3,3, 2,2, 1,1, 1,1)(nonBt1D15)
upsample2Rect = ReLU(true)(SpatialBatchNormalization(16)(upsample2))
nonBt1D16 = nonBt1D(16)(upsample2Rect)
nonBt1D17 = nonBt1D(16)(nonBt1D16)

upsample1 = SpatialFullConvolution(16, nClasses, 3,3, 2,2, 1,1, 1,1)(nonBt1D17)

reshape = nn.View(nClasses, w*h):setNumInputDims(3)(upsample1)
transpose = nn.Transpose({2, 1}):setNumInputDims(2)(reshape)

return nn.gModule({inData}, {transpose})