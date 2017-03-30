torch.setdefaulttensortype('torch.FloatTensor')

require 'cutorch'
torch.CudaTensor(4,4) -- warm up

for k = 0,9 do

require 'image'
local lena = image.lena():mean(1):squeeze()[{{1,7}, {1,2^k}}]:contiguous()

require 'Integral-cuda-multi'

-- compute true forward and backward results for some data
local intGold = Integral(3, lena:size(1), lena:size(2))
local params, gradParamsGold = intGold:getParameters()

local forwardGold = intGold:forward(lena)
local gradInputGold = intGold:backward(lena, forwardGold)

print('Computing gold result done')

local intTest = Integral(3, lena:size(1), lena:size(2))
intTest:cuda()
local paramsTest, gradParamsTest = intTest:getParameters()

paramsTest:copy(params)
intTest:recalculateArea()

-- compare results
function relativeError(A, B)
	local normalizer = A:norm(1)
	if normalizer < 5e-5 then
		normalizer = 1
	end

	-- real relative error would not require division by nElement, I know
	return (A-B):norm(1) / B:nElement() / normalizer
end

lena = lena:cuda()

local forwardTest = intTest:forward(lena)

local forwardErr = relativeError(forwardGold, forwardTest:float())
-- print('Output mean relative error:', forwardErr * 100 .. ' %')

local gradInputTest = intTest:backward(lena, forwardGold:cuda())
local gradInputErr = relativeError(gradInputGold, gradInputTest:float())
-- print('gradInput mean relative error:', gradInputErr * 100 .. ' %')

local gradParamsErr = relativeError(gradParamsGold, gradParamsTest:float())
-- print('gradParams mean relative error:', gradParamsErr * 100 .. ' %')

assert(forwardErr    < 1e-6)
assert(gradInputErr  < 1e-6)
assert(gradParamsErr < 7e-4)

end