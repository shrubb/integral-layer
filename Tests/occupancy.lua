require 'IntegralSmartNorm'

int = IntegralSmartNorm(32, 8, 128, 64):cuda()

-- require 'cunn'
-- int = nn.SpatialConvolution(32, 32*8, 3,3, 1,1, 1,1):cuda()

int.normalize = false
int.exact = true
int.saveMemoryIntegralInput = false
int.saveMemoryIntegralGradOutput = false
int.saveMemoryUpdateGradInput = false
int.saveMemoryAccGradParameters = false

batch = torch.CudaTensor(16, 32, 128, 64):fill(0.666)

int:forward(batch)
gradOutput = int.output:clone()
int:backward(batch, gradOutput)

local nRepeats = 60

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

-- Batch size 16

-- Int, 32->8 (exact, memory saving)
-- 1097 MB
-- Time for 1 forward: 0.062835502624512
-- Time for 1 updateGradInput: 0.088626098632812 
-- Time for 1 accGradParameters: 0.13107203642527

-- Int, 32->8 (exact, fastest -- in progress)
-- 1397 MB
-- Time for 1 forward: 0.061824099222819
-- Time for 1 updateGradInput: 0.050251269340515
-- Time for 1 accGradParameters: 0.12137493292491

-- Conv 3x3, 32->32*8
-- 617 MB
-- Time for 1 forward: 0.016526905695597
-- Time for 1 updateGradInput: 0.019815866152445
-- Time for 1 accGradParameters: 0.020316966374715
