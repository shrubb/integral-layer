local w, h, nClasses = ...
assert(w)
assert(h)
assert(nClasses)

require 'nn'
require 'cudnn'
require 'IntegralSmartNorm'

local SpatialConvolution = cudnn.SpatialConvolution
local SpatialDilatedConvolution = nn.SpatialDilatedConvolution
local SpatialFullConvolution = cudnn.SpatialFullConvolution
local ReLU = cudnn.ReLU
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
local SpatialMaxPooling = cudnn.SpatialMaxPooling

collectgarbage()

local model = nn.Sequential()

model
    :add(nn.Concat(2)
        :add(IntegralSmartNorm(3, 140, h, w))
        :add(nn.Sequential()
            :add(SpatialConvolution(3, 32, 3,3, 1,1, 1,1))
            :add(ReLU(true))))
    :add(SpatialBatchNormalization(3*140+32))
    :add(SpatialConvolution(3*140+32, 20, 1,1,1,1))
    :add(SpatialBatchNormalization(20))
    :add(ReLU(true))
    
    :add(nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.Concat(2)
                :add(IntegralSmartNorm(20, 28, h, w))
                :add(nn.Sequential()
                    :add(SpatialConvolution(20, 32, 3,3, 1,1, 1,1))
                    :add(ReLU(true))))
            :add(SpatialBatchNormalization(20*28+32))
            :add(SpatialConvolution(20*28+32, 20, 1,1,1,1)))
        :add(nn.Identity()))
    :add(nn.CAddTable())
    :add(SpatialBatchNormalization(20))
    :add(ReLU(true))

    :add(nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.Concat(2)
                :add(IntegralSmartNorm(20, 28, h, w))
                :add(nn.Sequential()
                    :add(SpatialConvolution(20, 32, 3,3, 1,1, 1,1))
                    :add(ReLU(true))))
            :add(SpatialBatchNormalization(20*28+32))
            :add(SpatialConvolution(20*28+32, 20, 1,1,1,1)))
        :add(nn.Identity()))
    :add(nn.CAddTable())
    :add(SpatialBatchNormalization(20))
    :add(ReLU(true))

    :add(SpatialConvolution(20, nClasses, 1,1,1,1))

model:add(nn.View(nClasses, w*h):setNumInputDims(3))
model:add(nn.Transpose({2, 1}):setNumInputDims(2))

return model