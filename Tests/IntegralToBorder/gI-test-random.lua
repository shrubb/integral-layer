require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntegralToBorder'

local seed = os.time()
-- seed = 1518704634
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
batchSize = 2
h,w = math.random(1+strideH, 7), math.random(1+strideW, 7)
print('h, w = ' .. h .. ', ' .. w)
print('stride = ' .. strideH .. ', ' .. strideW)

local function applyStride(k, stride)
    return math.ceil(k / stride)
end

int = IntegralToBorder(2, 2, h, w, strideH, strideW)

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
targetBatchIdx, targetPlane = math.random(1, batchSize), math.random(1, int.nInputPlane)

print('targetX, targetY, targetBatchIdx, targetPlane = ' .. targetX .. ', ' .. targetY .. ', ' .. targetBatchIdx .. ',' .. targetPlane)

crit = nn.MSECriterion():type(dtype)

img = torch.rand(batchSize, int.nInputPlane, h, w):type(dtype)
target = torch.rand(batchSize, int.nInputPlane*int.nWindows, applyStride(h,strideH), applyStride(w,strideW)):add(-0.5):mul(0.1):type(dtype)

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
    img[{targetBatchIdx, targetPlane, targetX, targetY}] = param
    pred = int:forward(img)

    params[k] = param
    loss[k] = crit:forward(pred, target)

    int:zeroGradParameters()
    int:updateGradInput(img, crit:updateGradInput(pred, target))
    derivM[k] = int.gradInput[{targetBatchIdx, targetPlane, targetX, targetY}]
    
    img[{targetBatchIdx, targetPlane, targetX, targetY}] = param + innerStep
    valFront = crit:forward(int:forward(img), target)
    img[{targetBatchIdx, targetPlane, targetX, targetY}] = param - innerStep
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
    gnuplot.raw('set term postscript eps')
    gnuplot.raw('set output \'Tests/gI-test-random-' .. iter .. '.eps\'')
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

print('Random seed was ' .. seed)