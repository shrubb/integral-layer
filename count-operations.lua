local exact = false
if exact then print('WARNING: computing ops for exact ints version') end

require 'nngraph'
require 'cudnn'
require 'cunn'
require 'IntegralZeroPadding'

local nClasses = 18
local w, h = assert(tonumber(arg[1])), assert(tonumber(arg[2]))
local modelFile = assert(arg[3])
local totalOps = tonumber(arg[4] or '')

if modelFile:find('.lua') then
    model = assert(loadfile(modelFile))(w, h, nClasses)
else
    model = torch.load(modelFile)
end

local hasCudnn = false
model:apply(function(m) hasCudnn = hasCudnn or torch.type(m):find('cudnn') end)
if hasCudnn then
    model:cuda()
else
    model:float()
end

model:evaluate()
sample = torch.FloatTensor(1, 3, h, w):typeAs(model)
model:forward(sample)

local totalOps, convOps, integralOps = 0, 0, 0
local paramsCount = 0

for iter = 1,2 do
    local intIdx = 1
    local intNumRedundantPlanes = {}

    convOps = 0
    integralOps = 0

    for _, module in ipairs(model:listModules()) do
        local kind = torch.type(module)
        local currentOps

        if     torch.isTypeOf(module, nn.Container) or kind:find('View') or kind:find('Reshape') or
               kind:find('Identity') or kind:find('JoinTable') or kind:find('BatchNorm') or
               kind:find('Padding') or kind == 'nn.Narrow' or kind == 'nn.Transpose' or
               kind:find('Dropout') or kind:find('Contiguous') or kind:find('LogSoftMax') then
            currentOps = 0

        elseif kind:find('Convolution') then
            currentOps = 
                module.kW * module.kH * module.output:nElement() *
                (kind:find('DepthWise') and 1 or (module.nInputPlane / (module.groups or 1))) +
                (module.bias and module.output:nElement() or 0)
            
            if module.kW * module.kH == 1 and module.nInputPlane > 256 and
                intNumRedundantPlanes[intIdx] then

                -- hope it's a post-Integral convolution
                assert(intNumRedundantPlanes[intIdx] < module.nInputPlane)
                
                currentOps =
                    module.output:nElement() *
                    (module.nInputPlane - intNumRedundantPlanes[intIdx] + 1)
            end

            intIdx = intIdx + 1
            
            convOps = convOps + currentOps

        elseif kind:find('Pooling') or kind:find('Unpooling') then
            currentOps = (module.kW or 1) * (module.kH or 1) * module.output:nElement()

        elseif kind:find('ReLU') or kind:find('MulConstant') or
               kind:find('CAddTable') or kind == 'nn.Mean' or
               kind == 'nn.SpatialAdaptiveMaxPooling' then
            currentOps = module.output:nElement()

        elseif kind:find('Linear') then
            currentOps = module.weight:size(1) * (module.weight:size(2) + (module.bias and 1 or 0))

        elseif kind:find('SpatialCrossMapLRN') then
            currentOps = module.output:nElement() * math.ceil(module.size / 2)

        elseif kind == 'nn.SpatialUpSamplingBilinear' then
            currentOps = module.output:nElement() * 15

        elseif kind:find('Integral') then
            local function as(k, stride) -- apply stride
                return math.ceil(k / stride)
            end

            currentOps = 
                module.h * module.w * module.nInputPlane + -- integral images
                as(module.h, module.strideH) * as(module.w, module.strideW) *
                    module.nInputPlane * module.nWindows * (exact and 25 or 5)

            if intNumRedundantPlanes[intIdx] then
                currentOps =
                    module.h * module.w * module.nInputPlane +
                    module.h * module.w * 
                    (module.nInputPlane * module.nWindows - intNumRedundantPlanes[intIdx]) * 5
            end

            integralOps = integralOps + currentOps

        else
            error('NYI: ' .. kind)

        end

        if iter == 1 then
            totalOps = totalOps + currentOps

            if not torch.isTypeOf(module, nn.Container) and not kind:find('BatchNorm') then
                for _,w in ipairs(module:parameters() or {}) do
                    paramsCount = paramsCount + w:nElement()
                end
            end
        else
            local info = ('%-36s% 12d operations'):format(kind, currentOps)
            if currentOps > 0 then
                info = info .. (' (%.2f%%)'):format(currentOps / totalOps * 100)
            end
            if kind:find('Convolution') then
                info = info .. (' %dx%d, %3d -> %3d, stride %dx%d, dilation %d')
                    :format(
                        module.kW, module.kH, module.nInputPlane, module.nOutputPlane, 
                        module.dW, module.dH, module.dilationH or 1)
            end
            print(info)
        end
    end
end

print('')
print('w x h = ' .. w .. ' x ' .. h .. ' = ' .. (w*h))
print(model.output:size())
print('')
print('Total operations: ' .. totalOps)
print(('Convolutions do %.2f%%'):format(convOps / totalOps * 100))
print(('Integral layers do %.2f%%'):format(integralOps / totalOps * 100))
print('Total number of parameters: ' .. paramsCount)
