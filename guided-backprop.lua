torch.setdefaulttensortype('torch.FloatTensor')
require 'cutorch'
require 'cudnn'
require 'cunn'
optnet = require 'optnet'
require 'IntegralSmartNorm'

cv = require 'cv'
require 'cv.imgproc'

cudnn.fastest = false
cudnn.benchmark = false

cityscapes = require 'cityscapes'

cityscapes.relative = '../../Datasets/Cityscapes/'
nClasses = cityscapes.nClasses -- 19
h, w = 512, 1024
cityscapes.dsize = {w, h}

trainFiles = cityscapes.loadNames('train')
valFiles = cityscapes.loadNames('val')

img, labels = cityscapes.loadSample(valFiles[13])

outputDir = 'Guided backprop/' .. os.date() .. '/'
os.execute('mkdir "' .. outputDir .. '" -p')

baseImage = torch.FloatTensor(3, h*3+4, w):fill(1)
baseImage[{{}, {1, h}}]:copy(img)
for c = 1,3 do
    baseImage[{c, {1, h}}]:mul(cityscapes.std[c])
    baseImage[{c, {1, h}}]:add(cityscapes.mean[c])
end
assert(0 <= baseImage:min() and baseImage:max() <= 1)

function drawMark(img, x, y)
    color = {1, 0, 0}
    for c = 1,3 do
        for i = -10,10 do
            for hShift = -1,1 do
                img[{c, x-i, y-i+hShift}] = color[c]
                img[{c, x+i, y-i+hShift}] = color[c]
            end
        end
    end
end

outputImages = {}

gridSizeH, gridSizeW = 9,10

pixelCoords = {}
for hIdx = 1,gridSizeH do
    for wIdx = 1,gridSizeW do
        local x = math.floor(h/(gridSizeH+1) * hIdx)
        local y = math.floor(w/(gridSizeW+1) * wIdx)

        if labels[{x, y}] ~= 20 then
            table.insert(pixelCoords, {x,y})

            local outputImage = baseImage:clone()
            drawMark(outputImage, x, y)
            table.insert(outputImages, outputImage)
        end
    end
end

-- assert(#arg == 2)

function mergeDropout(conv, dropout)
    conv.weight:mul(1-dropout.p)
    if conv.bias then conv.bias:mul(1-dropout.p) end
end

function absorbDropout(model)
    local i = 1
    while i <= #model.modules do
        layer_type = torch.type(model.modules[i])

        if torch.isTypeOf(model.modules[i], nn.Container) then
            absorbDropout(model.modules[i])
        elseif layer_type:find('BatchNormalization') or layer_type:find('Convolution') or 
               layer_type:find('Linear') then
            if torch.type(model.modules[i+1]):find('SpatialDropout') then
                mergeDropout(model.modules[i], model.modules[i+1])
            end
        elseif (layer_type:find('Dropout')) then
            model:remove(i)
            i = i - 1
        end
        i = i + 1
    end
end

-- https://github.com/Kaixhin/Atari/blob/52044699c09dad3b53358b94f3042bfe1c8ba46f/modules/GuidedReLU.lua
local GuidedReLU, parent = torch.class('nn.GuidedReLU', 'nn.ReLU')

function GuidedReLU:__init(p)
  parent.__init(self, p)
  self.guide = false
end

function GuidedReLU:updateOutput(input)
  return parent.updateOutput(self, input)
end

function GuidedReLU:updateGradInput(input, gradOutput)
  parent.updateGradInput(self, input, gradOutput)
  if self.guide then
    -- Only backpropagate positive error signals
    self.gt = self.gt or gradOutput:new(gradOutput:size())
    self.gradInput:cmul(torch.gt(self.gt, gradOutput, 0))
  end
  return self.gradInput
end

function GuidedReLU:salientBackprop()
  self.guide = true
end

function GuidedReLU:normalBackprop()
  self.guide = false
end

-- https://github.com/Kaixhin/Atari/blob/52044699c09dad3b53358b94f3042bfe1c8ba46f/modules/GuidedReLU.lua
local GuidedPReLU, parent = torch.class('nn.GuidedPReLU', 'nn.PReLU')

function GuidedPReLU:__init(p)
  parent.__init(self, p)
  self.guide = false
end

function GuidedPReLU:updateOutput(input)
  return parent.updateOutput(self, input)
end

function GuidedPReLU:updateGradInput(input, gradOutput)
  parent.updateGradInput(self, input, gradOutput)
  if self.guide then
    -- Only backpropagate positive error signals
    self.gt = self.gt or gradOutput:new(gradOutput:size())
    self.gradInput:cmul(torch.gt(self.gt, gradOutput, 0))
  end
  return self.gradInput
end

function GuidedPReLU:salientBackprop()
  self.guide = true
end

function GuidedPReLU:normalBackprop()
  self.guide = false
end

classNames = {
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
}

-- ***********************************************************************

require 'gnuplot'
require 'xlua'

input = nn.utils.addSingletonDimension(img:cuda())
gradOutput = torch.CudaTensor(h, w, 19):zero()
plots = {{}, {}}

for modelIdx, modelPath in ipairs{arg[1], arg[2]} do
    net = torch.load(modelPath .. '/net.t7')[1]
    if not torch.type(net:get(#net)):find('LogSoftMax') then
        net:add(nn.Contiguous())
        net:add(nn.View(-1, nClasses))
        net:add(nn.LogSoftMax())
    end
    net:cuda()

    absorbDropout(net)

    net:replace(function(m)
        if torch.typename(m):find('BatchNormalization') then
            return cudnn.convert(m, nn)
        elseif torch.typename(m):find('%.ReLU') then
            return nn.GuidedReLU():type(m:type())
        elseif torch.typename(m):find('PReLU') then
            local retval = nn.GuidedPReLU(m.nOutputPlane):type(m:type())
            retval.weight:copy(m.weight)
            return retval
        else
            return m
        end
    end)

    for _, m in ipairs(net:findModules('nn.GuidedReLU')) do
        -- m:salientBackprop()
        m:normalBackprop()
    end
    for _, m in ipairs(net:findModules('nn.GuidedPReLU')) do
        -- m:salientBackprop()
        m:normalBackprop()
    end

    local poolings   = net:findModules('nn.SpatialMaxPooling')
    local unpoolings = net:findModules('nn.SpatialMaxUnpooling')
    for i = 1,#unpoolings do
        unpoolings[i].pooling = poolings[#poolings-i+1]
    end

    local ints = net:findModules('IntegralSmartNorm')
    for _,int in ipairs(ints) do
        int.saveMemoryIntegralInput = true
        int.saveMemoryIntegralGradOutput = true
        int.saveMemoryUpdateGradInput = true
        int.saveMemoryAccGradParameters = true
    end

    -- optnet.optimizeMemory(net, input, {inplace=true, mode='training', reuseBuffers=true})
    for _,m in ipairs(net:listModules()) do
        m.gradWeight = nil
        m.gradBias = nil
    end

    collectgarbage()

    net:evaluate()
    net:forward(input)

    for pixelIdx, coords in ipairs(pixelCoords) do
        xlua.progress(pixelIdx, #pixelCoords)

        x,y = table.unpack(coords)
        local classIdx = labels[{x, y}]
        local className = classNames[classIdx]

        for _,int in ipairs(ints) do int._backwardDone = false end

        gradOutput[{x, y, classIdx}] = 1
        net:updateGradInput(input, gradOutput:view(h*w, 19))
        gradOutput[{x, y, classIdx}] = 0

        local outputHeatmap = outputImages[pixelIdx]:narrow(2, 1+(h+2)*modelIdx, h)
        local gradInt

        do
            outputHeatmap[1]:copy(net.gradInput:float():abs():sum(2))
            gradInt = cv.integral{outputHeatmap[1]}
            outputHeatmap[1]:add(-outputHeatmap[1]:min())
            outputHeatmap[1]:div(outputHeatmap[1]:max())
            outputHeatmap:copy(image.y2jet(outputHeatmap[1]*255 + 1))
        end

        -- do
        --     outputHeatmap:copy(net.gradInput)
        --     outputHeatmap:add(-outputHeatmap:min())
        --     outputHeatmap:div(outputHeatmap:max())
        -- end

        if modelIdx == 2 then
            image.save(
                outputDir .. ('%03d-%s.png'):format(pixelIdx, className),
                outputImages[pixelIdx])
        end

        local function clipInt(t)
            return {
                math.max(math.min(t[1], h+1), 1),
                math.max(math.min(t[2], w+1), 1),
            }
        end

        local graph = {}
        for winW = 0,2*w do
            local winH = winW * (h / w)
            local windowSum = 
                gradInt[clipInt{x+winH/2+1, y+winW/2+1}] -
                gradInt[clipInt{x-winH/2  , y+winW/2+1}] -
                gradInt[clipInt{x+winH/2+1, y-winW/2  }] +
                gradInt[clipInt{x-winH/2  , y-winW/2  }]
            table.insert(graph, windowSum)
        end
        graph = torch.Tensor(graph) / gradInt:view(-1)[-1]

        table.insert(plots[modelIdx], graph)

        if modelIdx == 2 then
            gnuplot.figure()
            gnuplot.raw('set term png') --postscript color eps')
            gnuplot.raw('set output \'' .. outputDir .. ('%03d-%s-1.png\''):format(pixelIdx, className))
            gnuplot.xlabel('Sum window size around target pixel')
            gnuplot.ylabel('Fraction of gradient values covered by the window')
            gnuplot.grid('on')
            gnuplot.plot(
                {arg[1]:sub(31,33), torch.range(0,2*w), plots[1][pixelIdx], '-'},
                {arg[2]:sub(31,33), torch.range(0,2*w), plots[2][pixelIdx], '-'}
            )
            
            gnuplot.plotflush()
        end
    end

    net = nil
    collectgarbage()
    collectgarbage()
end
