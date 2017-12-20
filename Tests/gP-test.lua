require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntegralSmartNorm'

torch.manualSeed(666)
local h,w = 2,2
local strideH, strideW = 2, 2

int = IntegralSmartNorm(1, 1, h, w, strideH, strideW)
pX, pY = 2,1

int.exact = true
int.smart = true
int.replicate = true
int.normalize = false
crit = nn.MSECriterion()

img = torch.rand(int.nInputPlane, h, w)
target = torch.rand(int.nInputPlane*int.nWindows, applyStride(h, strideH), applyStride(w, strideW))

-- img[1][pX][pY] = 

-- int.xMin[1][2] = -1
-- int.xMax[1][2] = -0.5
-- int.yMin[1][2] = 0
-- int.yMax[1][2] = 0.25

-- int.xMin[1][2] = 1.5
-- int.xMax[1][2] = 1.5
-- int.yMin[1][2] = -1
-- int.yMax[1][2] = -1

local paramPlane, paramWin = 1,1

int.xMin[1][1] = 0
int.xMax[1][1] = 0
int.yMin[1][1] = 0
int.yMax[1][1] = 0

-- int.xMin[1][2] = 10
-- int.xMax[1][2] = 10
-- int.yMin[1][2] = 0
-- int.yMax[1][2] = 0

-- int.xMin[2][1] = 0
-- int.xMax[2][1] = 0
-- int.yMin[2][1] = 10
-- int.yMax[2][1] = 10

-- int.xMin[2][2] = -10
-- int.xMax[2][2] = -10
-- int.yMin[2][2] = 0
-- int.yMax[2][2] = 0

print('params before:')
print(int.xMin[paramPlane][paramWin], int.xMax[paramPlane][paramWin])
print(int.yMin[paramPlane][paramWin], int.yMax[paramPlane][paramWin])

int:forward(img)

print('params after:')
print(int.xMin[paramPlane][paramWin], int.xMax[paramPlane][paramWin])
print(int.yMin[paramPlane][paramWin], int.yMax[paramPlane][paramWin])

loss = crit:forward(int.output, target)
gradOutput = crit:updateGradInput(int.output, target)

print('input:')
print(img[1])
print('output:')
print(int.output[1])

print('gradOutput:')
print(gradOutput[1])

int:zeroGradParameters()
int:backward(img, gradOutput)

print('my grad (need 1.96149):')
print(int.gradYMin[paramPlane][paramWin])
-- do return end

params = {}
loss = {}
deriv = {}
derivM = {}

local k = 1
local step = 0.02
local innerStep = -0.005

for param = -w+1,int.yMax[paramPlane][paramWin],step do
    int.yMin[paramPlane][paramWin] = param
    pred = int:forward(img)

    if int.yMin[paramPlane][paramWin] == tonumber(ffi.new('float', param)) then
        params[k] = param
        loss[k] = crit:forward(pred, target)

        int:zeroGradParameters()
        int:backward(img, crit:updateGradInput(pred, target))
        derivM[k] = int.gradYMin[paramPlane][paramWin]
        
        -- int.xMax[paramPlane][paramWin] = param + innerStep
        -- valFront = crit:forward(int:forward(img), target)
        -- int.xMax[paramPlane][paramWin] = param - innerStep
        -- valBack = crit:forward(int:forward(img), target)
        
        -- deriv[k] = (valFront - valBack) / (2 * innerStep)

        int.yMin[paramPlane][paramWin] = param + innerStep
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