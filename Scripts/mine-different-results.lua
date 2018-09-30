torch.setdefaulttensortype('torch.FloatTensor')
require 'cutorch'
require 'cudnn'
require 'cunn'
require 'IntegralZeroPadding'

local saveMemory = false

cv = require 'cv'
require 'cv.imgproc'

cudnn.fastest = false
cudnn.benchmark = false

dataset = require 'cityscapes'

dataset.relative = '../../Datasets/Cityscapes/'
nClasses = dataset.nClasses -- 19
h, w = 512, 1024
dataset.dsize = {w, h}

evalFiles = dataset.loadNames('val')
-- for k = 11,500 do evalFiles[k] = nil end

outputDir = 'Scripts/DifferentResults/' .. os.date() .. '/'
os.execute('mkdir "' .. outputDir .. '" -p')

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

-- ***********************************************************************

local numOutputImages = assert(tonumber(arg[3]))

input = torch.CudaTensor(1, 3, h, w)
results, resultsCorrect = {}, {}

for modelIdx, modelPath in ipairs{arg[1], arg[2]} do
    
    local currentCorrect = torch.LongTensor(#evalFiles)
    local currentResults = torch.LongTensor(#evalFiles, h, w)

    net = torch.load(modelPath .. '/net.t7')[1]

    while not torch.type(net:get(#net)):find('FullConvolution') and 
          not torch.type(net:get(#net)):find('Bilinear') do
        net:remove()
    end
    net:cuda()

    absorbDropout(net)
    net:replace(function(m)
        if torch.typename(m):find('BatchNormalization') then
            return cudnn.convert(m, nn)
        else
            return m
        end
    end)

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

    collectgarbage()
    net:evaluate()

    local maxVal, maxInd = torch.FloatTensor(), torch.LongTensor()

    require 'xlua'
    xlua.progress(0, #evalFiles)

    for sampleIdx, evalFile in ipairs(evalFiles) do
        local img, labels = dataset.loadSample(evalFile)
        input:copy(img)
        local prediction = net:forward(input):float()
        torch.max(maxVal, maxInd, prediction, 2)

        labels = labels:long()
        maxInd[labels:eq(dataset.nClasses+1)] = dataset.nClasses + 1
        currentCorrect[sampleIdx] = maxInd:eq(labels):sum()
        currentResults[sampleIdx]:copy(maxInd)

        xlua.progress(sampleIdx, #evalFiles)
    end

    table.insert(results, currentResults)
    table.insert(resultsCorrect, currentCorrect)

    net = nil
    collectgarbage()
    collectgarbage()
end

local advantage = resultsCorrect[2]-resultsCorrect[1]
collectgarbage()
local val, ind = advantage:sort(true)
for outputImageIdx = 1,numOutputImages do
    local img, _ = dataset.loadSample(evalFiles[ind[outputImageIdx]])

    for modelIdx = 1,2 do
        local fileName = ('%s/%01d-%03d-%06d.png')
            :format(outputDir, modelIdx, outputImageIdx, resultsCorrect[modelIdx][ind[outputImageIdx]])
        local outputImage = dataset.renderLabels(results[modelIdx][ind[outputImageIdx]], img, 0.7)
        image.savePNG(fileName, outputImage)
    end
end
