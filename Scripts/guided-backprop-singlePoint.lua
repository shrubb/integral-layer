-- some variables are **intentionally** global
torch.setdefaulttensortype('torch.FloatTensor')
require 'cutorch'
require 'cudnn'
require 'cunn'

cv = require 'cv'
require 'cv.imgproc'

cudnn.fastest = false
cudnn.benchmark = false

local saveMemory = false
local stopAtBlock = 1000

cityscapes = require 'cityscapes'

cityscapes.relative = '../../Datasets/Cityscapes/'
nClasses = cityscapes.nClasses -- 19
h, w = 512, 1024
cityscapes.dsize = {w, h}

trainFiles = cityscapes.loadNames('train')
valFiles = cityscapes.loadNames('val')

local evalFiles = {} --25
for k = 1,#valFiles,10 do table.insert(evalFiles, valFiles[k]) end

math.randomseed(666)
local numPoints = 40 --10
-- coords = {}
-- for _ = 1,numPoints do
--     table.insert(coords, {math.random(1,h), math.random(1,w)})
-- end

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

assert(#arg % 2 == 0, 'Must give a name to every model')
local modelNames = {}
for k = #arg/2+1,#arg do
    table.insert(modelNames, arg[k])
    arg[k] = nil
end
assert(#arg == #modelNames)

-- ***********************************************************************

require 'gnuplot'
require 'xlua'

graphs = {}

input = torch.CudaTensor(1, 3, h, w)
gradOutput = torch.CudaTensor()

for modelIdx, modelPath in ipairs(arg) do
    _G['IntegralSmartNorm'] = nil
    _G['IntegralSymmetric'] = nil
    debug.getregistry()['IntegralSmartNorm'] = nil
    debug.getregistry()['IntegralZeroPadding'] = nil
    debug.getregistry()['IntegralSymmetric'] = nil
    package.loaded['IntegralSmartNorm'] = nil
    package.loaded['IntegralZeroPadding'] = nil

    require (modelPath:find('zp') and 'IntegralZeroPadding' or 'IntegralSmartNorm')

    net = torch.load(modelPath .. '/net.t7')[1]
    
    while not torch.type(net:get(#net)):find('FullConvolution') and 
          not torch.type(net:get(#net)):find('Bilinear') do
        net:remove()
    end
    net:add(nn.Narrow(2, 1, nClasses))
    net:add(cudnn.SpatialLogSoftMax())
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
        int.saveMemoryIntegralInput = saveMemory
        int.saveMemoryIntegralGradOutput = saveMemory
        int.saveMemoryUpdateGradInput = saveMemory
        int.saveMemoryAccGradParameters = saveMemory
    end

    for _,m in ipairs(net:listModules()) do
        m.gradWeight = nil
        m.gradBias = nil
    end

    assert((stopAtBlock or stopAtIntBlock) and not (stopAtBlock and stopAtIntBlock))

    if stopAtBlock then
        for _ = #net,stopAtBlock+1,-1 do
            net:remove()
        end
    end

    -- if stopAtIntBlock then
    --     local endBlockIdx
    --     local currentIntBlocks = 0
    --     for k = 1,#net do
    --         local hasInt = false
    --         for _,m in ipairs(net:get(k):listModules()) do
    --             if torch.type(m):find('Integral') then
    --                 hasInt = true
    --                 break
    --             end
    --         end

    --         if hasInt then
    --             currentIntBlocks = currentIntBlocks + 1
    --         end

    --         if currentIntBlocks >= stopAtIntBlock then
    --             endBlockIdx = k
    --             break
    --         end
    --     end
 
    --     for _ = #net,endBlockIdx+1,-1 do
    --         net:remove()
    --     end
    -- end

    collectgarbage()

    local graphsByClass = {}

    xlua.progress(0, #evalFiles)
    net:evaluate()

    for fileIdx, evalFile in ipairs(evalFiles) do
        local img, _ = cityscapes.loadSample(evalFile)
        input:copy(img)
        net:forward(input)
        gradOutput:resize(net.output:size()):zero()
        local downsamplingX, downsamplingY = h / net.output:size(3), w / net.output:size(4)

        for classIdx = 1,gradOutput:size(2) do
            for _,int in ipairs(ints) do int._backwardDone = false end

            for pointIdx = 1,numPoints do
                local x,y
                if coords then
                    x,y = table.unpack(coords[pointIdx])
                else
                    x,y = math.random(1,h), math.random(1,w)
                end

                gradOutput[{1, classIdx, x / downsamplingX, y / downsamplingY}] = 1
                net:updateGradInput(input, gradOutput)
                gradOutput[{1, classIdx, x / downsamplingX, y / downsamplingY}] = 0

                local gradInt = cv.integral{net.gradInput[1]:float():abs():sum(1)[1]}

                local function clipInt(t)
                    return {
                        math.max(math.min(t[1], h+1), 1),
                        math.max(math.min(t[2], w+1), 1),
                    }
                end

                local graph = {}
                for winW = 0,2*w+2 do
                    local winH = winW * (h / w)
                    local windowSum = 
                        gradInt[clipInt{x+winH/2+1, y+winW/2+1}] -
                        gradInt[clipInt{x-winH/2  , y+winW/2+1}] -
                        gradInt[clipInt{x+winH/2+1, y-winW/2  }] +
                        gradInt[clipInt{x-winH/2  , y-winW/2  }]
                    table.insert(graph, windowSum)
                end
                graph = torch.Tensor(graph):div(gradInt:view(-1)[-1])

                if not graphsByClass[classIdx] then
                    graphsByClass[classIdx] = graph
                else
                    graphsByClass[classIdx]:add(graph)
                end
            end
        end

        xlua.progress(fileIdx, #evalFiles)
    end

    local average = graphsByClass[1]:clone():zero()
    local validGraphs = 0
    local function hasNaN(x) return x:ne(x):sum() > 0 end
    for k = 1,#graphsByClass do
        if not hasNaN(graphsByClass[k]) then
            graphsByClass[k]:div(numPoints * #evalFiles)
            average:add(graphsByClass[k])
            validGraphs = validGraphs + 1
        end
    end
    average:div(validGraphs)
    table.insert(graphsByClass, average)
    table.insert(graphs, graphsByClass)

    net = nil
    collectgarbage()
    collectgarbage()
end

-- **************************************** Plot *******************************

local allModelNumbers = {}
for k = 1,#arg do
    local modelNumber = arg[k]:sub(31,33)
    table.insert(allModelNumbers, modelNumber)
end
allModelNumbers = table.concat(allModelNumbers, '-')

outputDir = 'Scripts/ReceptiveFieldSize/' .. allModelNumbers .. '-' .. 
            numPoints .. 'pts' .. (coords and '/' or 'random/')
os.execute('mkdir "' .. outputDir .. '" -p')

local classNames = {
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
    'average',
}

local file = io.open(outputDir .. 'data.txt', 'w')
file:write('(')

file:write('[')
for k = 1,#arg do file:write('\'', modelNames[k], '\',') end
file:write('],')

file:write('[')
for k = 1,#arg do
    file:write('[')
    graphs[k][#graphs[k]]:apply(function(x) file:write(x, ',') end)
    file:write('],')
end
file:write('],')

file:write(')')
file:close()

for classIdx = 1,#graphs[1] do
    className = 
        ('%03d'):format(classIdx) .. 
        (classNames[classIdx] and ('-'..classNames[classIdx]) or '')

    local plotTable = {}
    for k = 1,#arg do
        table.insert(plotTable, {modelNames[k], torch.range(0,2*w+2), graphs[k][classIdx], '-'})
    end

    gnuplot.figure()
    gnuplot.raw('set term png') --postscript color eps')
    gnuplot.raw('set output \'' .. outputDir .. className .. '.png\'')
    gnuplot.xlabel('Sum window size around target pixel')
    gnuplot.ylabel('Fraction of gradient values covered by the window')
    gnuplot.title(className)
    gnuplot.grid('on')
    gnuplot.plot(plotTable)

    gnuplot.plotflush()
end
