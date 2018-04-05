-- Total operations: 103995795

local w, h, nClasses = ...
assert(w)
assert(h)
assert(nClasses)

require 'cudnn'
require 'cunn'
require 'IntegralSmartNorm'
local utils = paths.dofile'utils.lua'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = cudnn.SpatialMaxPooling
local SBatchNorm = cudnn.SpatialBatchNormalization

local function createModel(opt)
   assert(opt and opt.depth)
   assert(opt and opt.num_classes)
   assert(opt and opt.widen_factor)

   local function Dropout()
      return nn.Dropout(opt and opt.dropout or 0,nil,true)
   end

   local depth = opt.depth

   local blocks = {}

   local function intBlock(inChannels, btChannels, numBoxes, dropoutProb, h, w)
      -- assert(inChannels % numBoxes == 0)
      -- local btChannels = inChannels / numBoxes

      dropoutProb = dropoutProb or 0

      local mainBranch = nn.Sequential()
         :add(Convolution(inChannels, btChannels, 1,1, 1,1))
         :add(SBatchNorm(btChannels))
         :add(ReLU(true))

         :add(IntegralSmartNorm(btChannels, numBoxes, h, w))
        
         -- :add(SpatialConvolution(inChannels, inChannels, 1,1, 1,1))
         :add(SBatchNorm(inChannels))
        
         :add(dropoutProb ~= 0 and nn.SpatialDropout(dropoutProb) or nn.Identity())

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(mainBranch))
         :add(nn.CAddTable())
         :add(ReLU(true))
   end
   
   local function wide_basic(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,0,0},
         {3,3,1,1,0,0},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()     

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(true))
            convs:add(nn.SpatialReflectionPadding(1,1, 1,1))
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            convs:add(nn.SpatialReflectionPadding(1,1, 1,1))
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
         end
      end
     
      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)
     
      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride, h, w)
      local s = nn.Sequential()

      assert(nOutputPlane % 4 == 0)
      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(intBlock(nOutputPlane, nOutputPlane / 4, 4, 0.15, h / stride, w / stride))
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      model:add(nn.SpatialReflectionPadding(1,1, 1,1))
      model:add(Convolution(3,nStages[1], 3,3, 1,1)) -- one conv at the beginning (spatial size: 32x32)

      
      model:add(intBlock(nStages[1], nStages[1], 1, 0.15, h, w))
      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1, h  , w  )) -- Stage 1 (spatial size: 32x32)

      model:add(intBlock(nStages[2], nStages[2] / 4, 4, 0.15, h, w))
      model:add(layer(wide_basic, nStages[2], nStages[3], n, 2, h, w)) -- Stage 2 (spatial size: 16x16)

      model:add(intBlock(nStages[3], nStages[3] / 4, 4, 0.15, h/2, w/2))
      model:add(layer(wide_basic, nStages[3], nStages[4], n, 2, h/2, w/2)) -- Stage 3 (spatial size: 8x8)

      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], opt.num_classes))
   end

   utils.DisableBias(model)
   utils.MSRinit(model)
   utils.FCinit(model)
   
   -- model:get(1).gradInput = nil

   return model
end

return createModel{
   num_classes  = nClasses,
   depth        = 16,
   widen_factor = 2,
   dropout      = 0.3,
}
