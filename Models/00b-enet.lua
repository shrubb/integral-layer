-- ENet

-- Total operations: 4 650 434 560  
-- Convolutions do 97.67%  
-- Integral layers do 0.00%   
-- Total number of parameters: 1803634

local w, h, nClasses = ...
assert(w)
assert(h)
assert(nClasses)

require 'nn'
require 'cudnn'

local SpatialConvolution = cudnn.SpatialConvolution
local SpatialDilatedConvolution = cudnn.SpatialDilatedConvolution
local SpatialFullConvolution = cudnn.SpatialFullConvolution
local ReLU = cudnn.ReLU
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
local SpatialMaxPooling = cudnn.SpatialMaxPooling

collectgarbage()

local model = nn.Sequential()

local ct = 0
function _bottleneck(internal_scale, use_relu, asymetric, dilated, input, output, downsample)
   local internal = output / internal_scale
   local input_stride = downsample and 2 or 1

   local sum = nn.ConcatTable()

   local main = nn.Sequential()
   local other = nn.Sequential()
   sum:add(main):add(other)

   main:add(SpatialConvolution(input, internal, input_stride, input_stride, input_stride, input_stride, 0, 0):noBias())
   main:add(SpatialBatchNormalization(internal, 1e-3))
   if use_relu then main:add(nn.PReLU(internal)) end
   if not asymetric and not dilated then
      main:add(SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
   elseif asymetric then
      local pad = (asymetric-1) / 2
      main:add(SpatialConvolution(internal, internal, asymetric, 1, 1, 1, pad, 0):noBias())
      main:add(SpatialConvolution(internal, internal, 1, asymetric, 1, 1, 0, pad))
   elseif dilated then
      main:add(SpatialDilatedConvolution(internal, internal, 3, 3, 1, 1, dilated, dilated, dilated, dilated))
   else
      assert(false, 'You shouldn\'t be here')
   end
   main:add(SpatialBatchNormalization(internal, 1e-3))
   if use_relu then main:add(nn.PReLU(internal)) end
   main:add(SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0):noBias())
   main:add(SpatialBatchNormalization(output, 1e-3))
   main:add(nn.SpatialDropout((ct < 5) and 0.01 or 0.1))
   ct = ct + 1

   other:add(nn.Identity())
   if downsample then
      other:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   end
   if input ~= output then
      other:add(nn.Padding(1, output-input, 3))
   end

   return nn.Sequential():add(sum):add(nn.CAddTable()):add(nn.PReLU(output))
end

local _ = require 'moses'
local bottleneck = _.bindn(_bottleneck, 4, true, false, false)
local cbottleneck = _.bindn(_bottleneck, 4, true, false, false)
local xbottleneck = _.bindn(_bottleneck, 4, true, 7, false)
local wbottleneck = _.bindn(_bottleneck, 4, true, 5, false)
local dbottleneck = _.bindn(_bottleneck, 4, true, false, 2)
local xdbottleneck = _.bindn(_bottleneck, 4, true, false, 4)
local xxdbottleneck = _.bindn(_bottleneck, 4, true, false, 8)
local xxxdbottleneck = _.bindn(_bottleneck, 4, true, false, 16)
local xxxxdbottleneck = _.bindn(_bottleneck, 4, true, false, 32)

local initial_block = nn.ConcatTable(2)
initial_block:add(SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
initial_block:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(initial_block)                                         -- 128x256
model:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
model:add(SpatialBatchNormalization(16, 1e-3))
model:add(nn.PReLU(16))
model:add(bottleneck(16, 64, true))                              -- 64x128
for i = 1,4 do
   model:add(bottleneck(64, 64))
end
model:add(bottleneck(64, 128, true))                             -- 32x64
for i = 1,2 do
   model:add(cbottleneck(128, 128))
   model:add(dbottleneck(128, 128))
   model:add(wbottleneck(128, 128))
   model:add(xdbottleneck(128, 128))
   model:add(cbottleneck(128, 128))
   model:add(xxdbottleneck(128, 128))
   model:add(wbottleneck(128, 128))
   model:add(xxxdbottleneck(128, 128))
end

-- SpatialMaxUnpooling requires nn modules...
model:apply(function(module)
   if module.modules then
      for i,submodule in ipairs(module.modules) do
         if torch.typename(submodule):match('cudnn.SpatialMaxPooling') then
            module.modules[i] = nn.SpatialMaxPooling(2, 2, 2, 2) -- TODO: make more flexible
         end
      end
   end
end)

-- find pooling modules
local pooling_modules = {}
model:apply(function(module)
   if torch.typename(module):match('nn.SpatialMaxPooling') then
      table.insert(pooling_modules, module)
   end
end)
assert(#pooling_modules == 3, 'There should be 3 pooling modules')

-- kill gradient
-- local grad_killer = nn.Identity()
-- function grad_killer:updateGradInput(input, gradOutput)
--    return self.gradInput:resizeAs(gradOutput):zero()
-- end
-- model:add(grad_killer)

-- decoder:

function bottleneck(input, output, upsample, reverse_module)
   local internal = output / 4
   local input_stride = upsample and 2 or 1

   local module = nn.Sequential()
   local sum = nn.ConcatTable()
   local main = nn.Sequential()
   local other = nn.Sequential()
   sum:add(main):add(other)

   main:add(SpatialConvolution(input, internal, 1, 1, 1, 1, 0, 0):noBias())
   main:add(SpatialBatchNormalization(internal, 1e-3))
   main:add(ReLU(true))
   if not upsample then
      main:add(SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
   else
      main:add(SpatialFullConvolution(internal, internal, 3, 3, 2, 2, 1, 1, 1, 1))
   end
   main:add(SpatialBatchNormalization(internal, 1e-3))
   main:add(ReLU(true))
   main:add(SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0):noBias())
   main:add(SpatialBatchNormalization(output, 1e-3))

   other:add(nn.Identity())
   if input ~= output or upsample then
      other:add(SpatialConvolution(input, output, 1, 1, 1, 1, 0, 0):noBias())
      other:add(SpatialBatchNormalization(output, 1e-3))
      if upsample and reverse_module then
         other:add(nn.SpatialMaxUnpooling(reverse_module))
      end
   end

   if upsample and not reverse_module then
      main:remove(#main.modules) -- remove BN
      return main
   end
   return module:add(sum):add(nn.CAddTable()):add(ReLU(true))
end

--model:add(bottleneck(128, 128))
model:add(bottleneck(128, 64, true, pooling_modules[3]))         -- 32x64
model:add(bottleneck(64, 64))
model:add(bottleneck(64, 64))
model:add(bottleneck(64, 16, true, pooling_modules[2]))          -- 64x128
model:add(bottleneck(16, 16))
model:add(SpatialFullConvolution(16, nClasses+1, 2, 2, 2, 2))

return model