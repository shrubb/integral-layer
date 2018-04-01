require 'IntegralSmartNorm'

int = IntegralSmartNorm(64, 4, 256, 128):cuda()

int.normalize = true
int.exact = true
int.saveMemoryIntegralInput = true
int.saveMemoryIntegralGradOutput = true
int.saveMemoryUpdateGradInput = true

batch = torch.CudaTensor(4, 64, 256, 128):fill(0.666)

int:forward(batch)
gradOutput = int.output:clone()
int:backward(batch, gradOutput)

local nRepeats = 15

local timer = torch.Timer()
for k = 1,nRepeats do
    int:forward(batch)
end

cutorch.synchronize()
print('Time for 1 forward: ' .. (timer:time().real / nRepeats))

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
