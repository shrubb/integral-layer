require 'IntegralSmartNorm'

int = IntegralSmartNorm(64, 4, 256, 128):cuda()

-- require 'cunn'
-- int = nn.SpatialConvolution(64, 64, 3,3, 1,1, 1,1):cuda()

int.normalize = true
int.exact = true
int.saveMemoryIntegralInput = false
int.saveMemoryIntegralGradOutput = false
int.saveMemoryUpdateGradInput = false

batch = torch.CudaTensor(4, 64, 256, 128):fill(0.666)

int:forward(batch)
gradOutput = int.output:clone()
int:backward(batch, gradOutput)

local nRepeats = 30

local timer = torch.Timer()
for k = 1,nRepeats do
    int:forward(batch)
end

cutorch.synchronize()
print('Time for 1 forward: ' .. (timer:time().real / nRepeats))

int:backward(batch, gradOutput)
cutorch.synchronize()

timer:reset()
for k = 1,nRepeats do
    int:updateGradInput(batch, gradOutput)
end

cutorch.synchronize()
print('Time for 1 updateGradInput: ' .. (timer:time().real / nRepeats))

timer:reset()
for k = 1,nRepeats do
    int:accGradParameters(batch, gradOutput)
end

cutorch.synchronize()
print('Time for 1 accGradParameters: ' .. (timer:time().real / nRepeats))

-- Int, 64->4 (exact, memory saving)
-- 1153 MB
-- Time for 1 forward: 0.079148634274801   
-- Time for 1 updateGradInput: 0.081282838185628   
-- Time for 1 accGradParameters: 0.1293244043986

-- Int, 64->4 (exact, fastest)
-- 1465 MB
-- Integral takes ~ 0.00572596391042
-- Time for 1 forward: 0.079643694559733
-- Time for 1 updateGradInput: 0.053880707422892
-- Time for 1 accGradParameters: 0.12368336518606

-- Conv 3x3, 64->64
-- 559 MB
-- Time for 1 forward: 0.015473898251851
-- Time for 1 updateGradInput: 0.019741233189901
-- Time for 1 accGradParameters: 0.023761534690857
