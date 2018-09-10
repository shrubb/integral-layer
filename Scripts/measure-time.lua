require 'nngraph'
require 'cudnn'
require 'cunn'
require 'IntegralZeroPadding'

cudnn.fastest = true
cudnn.benchmark = true

local useCudnn = true
local cpu = false
local exact = false
if exact then print('WARNING: running exact ints version') end  

if cpu then assert(torch.getnumthreads() == 1) end

local tensorType = cpu and 'torch.FloatTensor' or 'torch.CudaTensor'

local nClasses = 18
local w, h = assert(tonumber(arg[1])), assert(tonumber(arg[2]))
local modelFile = assert(arg[3])

if modelFile:find('.lua') then
    net = assert(loadfile(modelFile))(w, h, nClasses)
else
    net = torch.load(modelFile)
    while not torch.type(net:get(#net)):find('FullConvolution') and 
          not torch.type(net:get(#net)):find('Bilinear') do
        print('Removing ' .. torch.type(net:get(#net)))
        net:remove()
    end
end

net:type(tensorType)

-- optimize a bit for inference
net:replace(function(module)
   if torch.typename(module):find('BatchNormalization') or 
      torch.typename(module):find('Dropout') then
      return nn.Identity()
   else
      return module
   end
end)

if not useCudnn or cpu then
  net = cudnn.convert(net, nn)
  for _,m in ipairs(net:listModules()) do
    if torch.type(m):find('cudnn') then
      print('WARNING:', m)
    end
  end
end

net:evaluate()

for _,int in ipairs(net:findModules('IntegralSmartNorm')) do
    int.exact = exact
end

sample = torch.FloatTensor(1, 3, h, w):fill(1.23):typeAs(net)
net:forward(sample)
if not cpu then cutorch.synchronize() end

-- torch.setnumthreads(1)

local nRepeats = 40

print('Starting repeats')
local timer = torch.Timer()

for _ = 1,nRepeats do
    net:forward(sample)
end

if not cpu then cutorch.synchronize() end
local time = timer:time().real

print(('%.3f FPS'):format(nRepeats / time))
print(('%.5f seconds per frame'):format(time / nRepeats))

-- 000: 0.08601
-- 00b: 0.04690
-- 