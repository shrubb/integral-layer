require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'IntegralSmartNorm'

torch.manualSeed(666)
local h,w = 4,4

int = IntegralSmartNorm(2, 2, h, w)
pX, pY = 4,4

int.exact = true
int.smart = true
int.replicate = true
int.normalize = false
crit = nn.MSECriterion()

img = torch.ones(int.nInputPlane, h, w)
target = torch.ones(int.nInputPlane*int.nWindows, h, w)

--     reference.xMin = torch.Tensor{-200, 35.5,  99, 0.0}
--     reference.xMax = torch.Tensor{-195, 35.5, 100, 1.1}
--     reference.yMin = torch.Tensor{  95, -1.0,  99,  -10}
--     reference.yMax = torch.Tensor{ 100, -1.0, 100,   0}
-- int.xMin[1][1] = -0.7
-- int.xMax[1][1] = 0.3
-- int.yMin[1][1] = -1.15
-- int.yMax[1][1] = 0.2

-- int.xMin[1][2] = 1.5
-- int.xMax[1][2] = 1.5
-- int.yMin[1][2] = -1
-- int.yMax[1][2] = -1

img[{1,pX,pY}] = 5
int:forward(img)

loss = crit:forward(int.output, target)
gradOutput = crit:updateGradInput(int.output, target)

int:updateGradInput(img, gradOutput)

params = {}
loss = {}
deriv = {}
derivM = {}

local k = 1
local step = 0.05
local innerStep = 0.005

for param = -10,16,step do
    params[k] = param
    img[{1,pX,pY}] = param
    pred = int:forward(img)
    loss[k] = crit:forward(pred, target)
    
    int:updateGradInput(img, crit:updateGradInput(pred, target))
    derivM[k] = int.gradInput[{1,pX,pY}]
    
    img[{1,pX,pY}] = param + innerStep
    valFront = crit:forward(int:forward(img), target)
    img[{1,pX,pY}] = param - innerStep
    valBack = crit:forward(int:forward(img), target)
    
    deriv[k] = (valFront - valBack) / (2 * innerStep)
    
    k = k + 1
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