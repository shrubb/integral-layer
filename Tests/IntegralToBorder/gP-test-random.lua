require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntegralToBorder'

local seed = os.time()
-- seed = 1518018971
torch.manualSeed(seed)
math.randomseed(seed)

local targetParam = 'yMax'
print('The parameter to test is ' .. targetParam)
local targetParamGrad = 'grad' .. targetParam:sub(1,1):upper() .. targetParam:sub(2,-1)

local CUDA = true
local dtype = CUDA and 'torch.CudaTensor' or 'torch.FloatTensor'

if CUDA then
    require 'cunn'
end

for iter = 1,(arg[1] or 1) do

batchSize = 2
h,w = math.random(2, 8), math.random(2, 8)
strideH, strideW = 1,1
print('h, w = ' .. h .. ', ' .. w)
print('stride = ' .. strideH .. ', ' .. strideW)

local function applyStride(k, stride)
    return math.ceil(k / stride)
end

int = IntegralToBorder(2, 12, h, w, strideH, strideW):type(dtype)

int.exact = true
int.smart = true
int.replicate = true
int.normalize = false
crit = nn.MSECriterion():type(dtype)

img = torch.rand(batchSize, int.nInputPlane, h, w):type(dtype)
target = torch.rand(
    batchSize, 
    int.nInputPlane*int.nWindows, 
    applyStride(h,strideH), 
    applyStride(w,strideW)) :mul(2):add(-1):type(dtype)

local paramPlane, paramWin = 
    math.random(1,int.nInputPlane), math.random(1,int[targetParam]:size(2))
print('paramPlane, paramWin = ' .. paramPlane .. ', ' .. paramWin)

--[[
local function rand(a,b)
    return torch.rand(1)[1] * (b-a) + a
end

for planeIdx = 1,int.nInputPlane do
    for winIdx = 1,int.nWindows do
        local xMin, yMin = rand(-h+1, h-2), rand(-w+1, w-2)
        local xMax, yMax = rand(xMin+1, h-1), rand(yMin+1, w-1)

        int.xMin[planeIdx][winIdx] = xMin
        int.xMax[planeIdx][winIdx] = xMax
        int.yMin[planeIdx][winIdx] = yMin
        int.yMax[planeIdx][winIdx] = yMax

        print('int.xMin[' .. planeIdx .. '][' .. winIdx .. '] = ' .. xMin)
        print('int.xMax[' .. planeIdx .. '][' .. winIdx .. '] = ' .. xMax)
        print('int.yMin[' .. planeIdx .. '][' .. winIdx .. '] = ' .. yMin)
        print('int.yMax[' .. planeIdx .. '][' .. winIdx .. '] = ' .. yMax)
        print('')
    end
end
int:_reparametrize(false)
--]]

if true or iter == 7 then

local paramsBefore = {}
for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
    local size = int[param]:size(2)
    paramsBefore[param] = int[param][paramPlane][math.min(paramWin,size)]
end

int:forward(img)

for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
    -- assert(paramsBefore[param] == tonumber(ffi.new('float', int[param][paramPlane][paramWin])),
    local size = int[param]:size(2)
    assert(math.abs(paramsBefore[param] - int[param][paramPlane][math.min(paramWin,size)]) < 1e-6,
        param .. ' was changed: ' .. paramsBefore[param]*int.reparametrization .. ' became ' .. 
        int[param][paramPlane][math.min(paramWin,size)]*int.reparametrization)
end

target:add(int.output)

loss = crit:forward(int.output, target)
gradOutput = crit:updateGradInput(int.output, target)

int:zeroGradParameters()
int:backward(img, gradOutput)
-- do return end

params = {}
loss = {}
deriv = {}
derivM = {}

local k = 1
local step = 0.04
local innerStep = int.exact and 0.005 or 1

local lowerLimit, upperLimit

int:_reparametrize(true)
if targetParam:find('Max') then
    if paramWin > int[targetParam:gsub('Max', 'Min')]:size(2) then
        lowerLimit = targetParam == 'xMax' and -h or -w
    else
        lowerLimit = int[targetParam:gsub('Max', 'Min')][paramPlane][paramWin]
    end
    upperLimit = targetParam == 'xMax' and h or w
else
    lowerLimit = targetParam == 'xMin' and -h or -w
    if paramWin > int[targetParam:gsub('Min', 'Max')]:size(2) then
        upperLimit = targetParam == 'xMin' and h or w
    else
        upperLimit = int[targetParam:gsub('Min', 'Max')][paramPlane][paramWin]
    end
    innerStep = -innerStep
end
int:_reparametrize(false)

timer = torch.Timer()
for param = lowerLimit,upperLimit,step do

    int:_reparametrize(true)
    int[targetParam][paramPlane][paramWin] = param
    int:_reparametrize(false)

    pred = int:forward(img)

    int:_reparametrize(true)
    local dirtyFixWindowsFired = math.abs(int[targetParam][paramPlane][paramWin] - param) > 1e-6
    int:_reparametrize(false)
    
---[[
    if not dirtyFixWindowsFired then
        params[k] = param
        loss[k] = crit:forward(pred, target)

        int:zeroGradParameters()
        int:backward(img, crit:updateGradInput(pred, target))
        derivM[k] = int[targetParamGrad][paramPlane][paramWin]
        
        -- int.xMax[paramPlane][paramWin] = param + innerStep
        -- valFront = crit:forward(int:forward(img), target)
        -- int.xMax[paramPlane][paramWin] = param - innerStep
        -- valBack = crit:forward(int:forward(img), target)
        
        -- deriv[k] = (valFront - valBack) / (2 * innerStep)

        int:_reparametrize(true)
        int[targetParam][paramPlane][paramWin] = param + innerStep
        int:_reparametrize(false)
        valFront = crit:forward(int:forward(img), target)
        deriv[k] = (valFront - loss[k]) / innerStep * int.reparametrization
        
        k = k + 1
    end
    --]]
end
if CUDA then cutorch.synchronize() end
print(timer:time().real .. ' seconds')

-- loss[#loss] = nil
-- params[#params] = nil
-- derivM[#derivM] = nil
---[[
require 'gnuplot'

gnuplot.figure(iter)

-- if os.getenv('CUDA_VISIBLE_DEVICES') then
    gnuplot.raw('set term postscript eps')
    gnuplot.raw('set output \'Tests/gP-test-random-' .. iter .. '.eps\'')
-- end

gnuplot.plot(
    {'Loss', torch.Tensor(params), torch.Tensor(loss), '-'},
    {'Diff', torch.Tensor(params), torch.Tensor(deriv), '-'},
    {'Manual', torch.Tensor(params), torch.Tensor(derivM), '-'}
)

gnuplot.grid(true)
--]]
end -- if
end -- for

print('Random seed was ' .. seed)