require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntegralVarScale'

local seed = os.time()
seed = 1517324044
print('Random seed is ' .. seed)
torch.manualSeed(seed)
math.randomseed(seed)

local testType = 'inner' -- 'corner' | 'border' | 'inner'
local CUDA = true
local dtype = CUDA and 'torch.CudaTensor' or 'torch.FloatTensor'

if CUDA then
    require 'cunn'
end

for iter = 1,(arg[1] or 1) do

strideH, strideW = 1, 1
h,w = math.random(1+strideH, 200), math.random(1+strideW, 200)
print('h, w = ' .. h .. ', ' .. w)
print('stride = ' .. strideH .. ', ' .. strideW)

local function applyStride(k, stride)
    return math.ceil(k / stride)
end

require 'nn'
scaleProcessor = nn.Identity()
int = IntegralVarScale(2, 2, h, w, strideH, strideW, scaleProcessor)

int.exact = true
int.smart = true
int.replicate = true
int.normalize = false

if testType == 'inner' then
    targetX = math.random(2, h-1)
    targetY = math.random(2, w-1)
elseif testType == 'corner' then
    targetX = ({1,h})[math.random(1,2)]
    targetY = ({1,w})[math.random(1,2)]
elseif testType == 'border' then
    if math.random(1,2) == 1 then
        -- vertical border
        targetX = math.random(2, h-1)
        targetY = ({1,w})[math.random(1,2)]
    else
        -- horizontal border
        targetX = ({1,h})[math.random(1,2)]
        targetY = math.random(2, w-1)
    end
end
targetPlane = math.random(1, int.nInputPlane)

print('targetX, targetY, targetPlane = ' .. targetX .. ', ' .. targetY .. ', ' .. targetPlane)

crit = nn.MSECriterion():type(dtype)

img = torch.rand(int.nInputPlane, h, w):type(dtype)
scales = torch.rand(h, w):mul(2):add(0.1):type(dtype)
target = torch.rand(int.nInputPlane*int.nWindows, applyStride(h,strideH), applyStride(w,strideW)):add(-0.5):mul(0.1):type(dtype)

scaleProcessor:forward(scales)

-- img = nn.utils.addSingletonDimension(img)
-- target = nn.utils.addSingletonDimension(target)

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

-- int2 = IntegralSmartNorm(2, 2, h, w, strideH, strideW)
-- int2.xMin:copy(int)
-- int2.xMax:copy(int)
-- int2.yMin:copy(int)
-- int2.yMax:copy(int)
-- int2.exact = int.exact
-- int2.smart = int.smart
-- int2.replicate = int.replicate
-- int2.normalize = int.normalize
-- int2:type(dtype)

if iter == 14 or true then
int:type(dtype)
int:forward(img)
target:add(int.output)

loss = crit:forward(int.output, target)
gradOutput = crit:updateGradInput(int.output, target)

int:zeroGradParameters()
int:updateGradInput(img, gradOutput) -- backward

params = {}
loss = {}
deriv = {}
derivM = {}

local k = 1
local step = 0.1
local innerStep = int.exact and 0.015 or 1

for param = -5,5,step do
    img[{targetPlane, targetX, targetY}] = param
    pred = int:forward(img)

    params[k] = param
    loss[k] = crit:forward(pred, target)

    int:zeroGradParameters()
    int:updateGradInput(img, crit:updateGradInput(pred, target))
    derivM[k] = int.gradInput[{targetPlane, targetX, targetY}]
    
    img[{targetPlane, targetX, targetY}] = param + innerStep
    valFront = crit:forward(int:forward(img), target)
    img[{targetPlane, targetX, targetY}] = param - innerStep
    valBack = crit:forward(int:forward(img), target)
    
    deriv[k] = (valFront - valBack) / (2 * innerStep)

    -- img[{targetPlane, targetX, targetY}] = param + innerStep
    -- valFront = crit:forward(int:forward(img), target)
    -- deriv[k] = (valFront - loss[k]) / innerStep
    
    k = k + 1
end

-- loss[#loss] = nil
-- params[#params] = nil
-- derivM[#derivM] = nil
-- 
require 'gnuplot'

gnuplot.figure(iter)

-- if os.getenv('CUDA_VISIBLE_DEVICES') then
--     gnuplot.raw('set term postscript eps')
--     gnuplot.raw('set output \'gI-test-random.eps\'')
-- end

gnuplot.plot(
    {'Loss', torch.Tensor(params), torch.Tensor(loss), '-'},
    {'Diff', torch.Tensor(params), torch.Tensor(deriv), '-'},
    {'Manual', torch.Tensor(params), torch.Tensor(derivM), '-'}
)

gnuplot.grid(true)
end
-- sys.sleep(0.05)
end -- for

print('Seed was ' .. seed)