require 'cudnn'; print('')
require 'IntegralZeroPadding'
require 'gnuplot'

net = torch.load(arg[1])[1]
modules = net:listModules()

numberOfInts = 0

plotTable = {}

function addLayerToDistribution(preConv, preBN, int, postBN, postConv)
    numberOfInts = numberOfInts + 1

    -- get parameters in percent of h and w
    int:_reparametrize(true)
    -- int.xMin:div(int.h)
    -- int.xMax:div(int.h)
    -- int.yMin:div(int.w)
    -- int.yMax:div(int.w)

    local height = int.xMax - int.xMin + 1
    local width  = int.yMax - int.yMin + 1
    local area = torch.cmul(height, width):view(-1)

    local preWeight = preConv.weight
        :squeeze():abs():max(2):cmul(preBN.weight:abs())
        :expand(int.nInputPlane, int.nWindows):contiguous():view(-1)
    local postWeight = postConv.weight
        :abs():transpose(1,2):contiguous()
        :max(3):max(4):squeeze()

    importance = postWeight:sum(2):squeeze():cmul(postBN.weight:abs())
    -- importance = torch.sort(postWeight, 2)[{{}, postConv.nOutputPlane * 0.75}] -- take 75th percentile
    importance:cmul(preWeight)

    assert(importance:nElement() == area:nElement())

    table.insert(plotTable, {importance, area, '+'})
end

for intModuleIdx,intModule in ipairs(modules) do
    if torch.type(intModule) == 'IntegralSmartNorm' then
        
        local postBN = modules[intModuleIdx+1]
        local  preBN = modules[intModuleIdx-2]
        assert(torch.type(postBN):find('BatchNorm'))
        assert(torch.type( preBN):find('BatchNorm'))

        local postConv
        for convModuleIdx = intModuleIdx+1,#modules do
            
            local convModule = modules[convModuleIdx]
            if torch.type(convModule):find('SpatialConvolution') and
               convModule.nInputPlane == intModule.nInputPlane * intModule.nWindows then
        
                postConv = convModule
                break
            end
        end

        if not postConv then
            print('WARNING: could not find a following conv for modules['..intModuleIdx..']')
        end

        local preConv
        for convModuleIdx = intModuleIdx-1,1,-1 do
            
            local convModule = modules[convModuleIdx]
            if torch.type(convModule):find('SpatialConvolution') and
               convModule.nOutputPlane == intModule.nInputPlane then
                
                assert(convModule.kH == 1 and convModule.kW == 1)
                preConv = convModule
                break
            end
        end

        if not preConv then
            print('WARNING: could not find a preceding conv for modules['..intModuleIdx..']')
        end

        if preConv and postConv then
            addLayerToDistribution(preConv, preBN, intModule, postBN, postConv)
        end
    end
end

gnuplot.figure(numberOfInts)
gnuplot.logscale(true)
gnuplot.plot(plotTable)
gnuplot.xlabel('Filter Importance')
gnuplot.ylabel('Filter size, pixels')
gnuplot.grid(true)

local modelName = arg[1]
local slashIdx = modelName:find('/')
modelName = modelName:sub(slashIdx+1, -1)
local slashIdx = modelName:find('/')
modelName = modelName:sub(slashIdx+1, -8):gsub('/', '.')

local file = assert(io.open('Scripts/BoxDistributions/box-distribution-' .. modelName .. '.txt', 'w'))
file:write('[')

for intIdx,t in ipairs(plotTable) do
    file:write('[')
    
    local importance, area = t[1], t[2]
    for k = 1,importance:nElement() do
        file:write('[', importance[k], ',', area[k], '],')
    end

    file:write('],')
end

file:write(']')
file:close()