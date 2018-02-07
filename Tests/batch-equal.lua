require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntegralSmartNorm'

local seed = os.time()
seed = 1518018971
torch.manualSeed(seed)
math.randomseed(seed)

local CUDA = true
local dtype = CUDA and 'torch.CudaTensor' or 'torch.FloatTensor'

if CUDA then
    require 'cunn'
end

batchSize = 2
h,w = math.random(2, 4), math.random(2, 4)
strideH, strideW = 1,1
print('h, w = ' .. h .. ', ' .. w)
print('stride = ' .. strideH .. ', ' .. strideW)

local function applyStride(k, stride)
    return math.ceil(k / stride)
end

int = IntegralSmartNorm(2, 2, h, w, strideH, strideW):type(dtype)

int.exact = true
int.smart = true
int.replicate = true
int.normalize = false

img = torch.rand(batchSize, int.nInputPlane, h, w):type(dtype)

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

local paramsBefore = {}
for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
	paramsBefore[param] = {}
	for paramWin = 1,int.xMin:nElement() do
	    paramsBefore[param][paramWin] = int[param]:view(-1)[paramWin]
	end
end

output1 = int:forward(img[{{1,1}}]):clone()

for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
	for paramWin = 1,int.xMin:nElement() do
	    -- assert(paramsBefore[param] == tonumber(ffi.new('float', int[param][paramPlane][paramWin])),
	    assert(math.abs(paramsBefore[param][paramWin] - int[param]:view(-1)[paramWin]) < 1e-6,
	        param .. ' was changed: ' .. paramsBefore[param][paramWin]*int.reparametrization .. ' became ' .. int[param]:view(-1)[paramWin]*int.reparametrization)
	end
end

output2 = int:forward(img[{{2,2}}]):clone()
outputBoth = int:forward(img):clone()

-- print('Need:'); print(output2[1])
-- print('Have:'); print(outputBoth[2])

print('Diff first: ' .. (output1[1] - outputBoth[1]):abs():max())
print('Diff second: ' .. (output2[1] - outputBoth[2]):abs():max())
print('Diff second vs first: ' .. (output1[1] - outputBoth[2]):abs():max())