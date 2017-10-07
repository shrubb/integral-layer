local w, h, nClasses = ...
assert(w)
assert(h)
assert(nClasses)

require 'nn'
require 'cudnn'

function nn.Module.setGroupSparsityMark(self, l, r)
    self.intPlanesStartIdx = l
    self.intPlanesEndIdx   = r
    return self
end

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
        :add(SpatialDilatedConvolution(3, 512, 3,3, 1,1, 3,3, 3,3))
        :add(SpatialConvolution(3, 32, 3,3, 1,1, 1,1)))
    :add(SpatialBatchNormalization(512+32))
    :add(ReLU(true))
    :add(SpatialConvolution(512+32, 20, 1,1,1,1))
    :add(SpatialBatchNormalization(20))
    :add(ReLU(true))
    
    :add(nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.Concat(2)
                :add(SpatialDilatedConvolution(20, 20*28, 3,3, 1,1, 7,7, 7,7))
                :add(SpatialConvolution(20, 32, 3,3, 1,1, 1,1)))
            :add(SpatialBatchNormalization(20*28+32))
            :add(ReLU(true))
            :add(SpatialConvolution(20*28+32, 20, 1,1,1,1)))
        :add(nn.Identity()))
    :add(nn.CAddTable(true))
    :add(SpatialBatchNormalization(20))
    :add(ReLU(true))

    :add(nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.Concat(2)
                :add(SpatialDilatedConvolution(20, 20*28, 3,3, 1,1, 14,14, 14,14))
                :add(SpatialConvolution(20, 32, 3,3, 1,1, 1,1)))
            :add(SpatialBatchNormalization(20*28+32))
            :add(ReLU(true))
            :add(SpatialConvolution(20*28+32, 20, 1,1,1,1)))
        :add(nn.Identity()))
    :add(nn.CAddTable(true))
    :add(SpatialBatchNormalization(20))
    :add(ReLU(true))

    :add(SpatialConvolution(20, nClasses, 1,1,1,1))

model:add(nn.View(nClasses, w*h):setNumInputDims(3))
model:add(nn.Transpose({2, 1}):setNumInputDims(2))

return model