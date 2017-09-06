require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntegralSmartNorm'

local seed = os.time()
-- seed = 1504693562
print('Random seed is ' .. seed)
torch.manualSeed(seed)
math.randomseed(seed)

local targetParam = 'yMin'
print('The parameter to test is ' .. targetParam)
local targetParamGrad = 'grad' .. targetParam:sub(1,1):upper() .. targetParam:sub(2,-1)

local h,w = math.random(2, 100), math.random(2, 100)
print('h, w = ' .. h .. ', ' .. w)

local CUDA = false
local dtype = CUDA and 'torch.CudaTensor' or 'torch.FloatTensor'

int = IntegralSmartNorm(2, 2, h, w)

int.exact = true
int.smart = true
int.replicate = true
int.normalize = true
crit = nn.MSECriterion():type(dtype)

img = torch.rand(int.nInputPlane, h, w):type(dtype)
target = torch.rand(int.nInputPlane*int.nWindows, h, w):mul(2):add(-1):type(dtype)

local paramPlane, paramWin = math.random(1,int.nInputPlane), math.random(1,int.nWindows)
print('paramPlane, paramWin = ' .. paramPlane .. ', ' .. paramWin)

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

initialParam = int[targetParam][paramPlane][paramWin]

local paramsBefore = {}
for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
    paramsBefore[param] = int[param][paramPlane][paramWin]
end

int:forward(img)

for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
    assert(paramsBefore[param] == tonumber(ffi.new('float', int[param][paramPlane][paramWin])),
        param .. ' was changed')
end

target:add(int.output)

loss = crit:forward(int.output, target)
gradOutput = crit:updateGradInput(int.output, target)

int:zeroGradParameters()
int:backward(img, gradOutput)

params = {}
loss = {}
deriv = {}
derivM = {}

local k = 1
local step = 0.1
local innerStep = 0.004

local lowerLimit, upperLimit

if targetParam:find('Max') then
    lowerLimit = int[targetParam:gsub('Max', 'Min')][paramPlane][paramWin]
    upperLimit = targetParam == 'xMax' and h or w
else
    lowerLimit = targetParam == 'xMin' and -h or -w
    upperLimit = int[targetParam:gsub('Min', 'Max')][paramPlane][paramWin]
    innerStep = -innerStep
end

for param = lowerLimit,upperLimit,step do
    int[targetParam][paramPlane][paramWin] = param
    pred = int:forward(img)

    if int[targetParam][paramPlane][paramWin] == tonumber(ffi.new('float', param)) then
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

        int[targetParam][paramPlane][paramWin] = param + innerStep
        valFront = crit:forward(int:forward(img), target)
        deriv[k] = (valFront - loss[k]) / innerStep
        
        k = k + 1
    end
end

-- loss[#loss] = nil
-- params[#params] = nil
-- derivM[#derivM] = nil

require 'gnuplot'

gnuplot.plot(
    {'Loss', torch.Tensor(params), torch.Tensor(loss), '-'},
    {'Diff', torch.Tensor(params), torch.Tensor(deriv), '-'},
    {'Manual', torch.Tensor(params), torch.Tensor(derivM), '-'}
)

gnuplot.grid(true)