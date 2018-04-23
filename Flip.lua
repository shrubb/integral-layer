require 'nn'
local _, parent = torch.class('nn.Flip', 'nn.Module')

ffi = require 'ffi'

ffi.cdef [[
void flipCuda(struct THCState *state,
    const float *input, const int h, const int w, float *output, bool accumulate);
]]

local CUDA_lib

if pcall(require, 'cutorch') then
    CUDA_lib = ffi.load('C/lib/libflip-cuda.so')
end

do
    function nn.Flip:__init()
        self.output = torch.Tensor()
        self.gradInput = torch.Tensor()
        self._accumulateGradInput = false
    end

    function nn.Flip:accumulateGradInput(yes)
        self._accumulateGradInput = yes == false and false or true
        return self
    end

    function nn.Flip:updateOutput(input)
        self.output:resize(input:size())
        assert(self.output:isContiguous())

        input = input:view(-1, input:size(input:nDimension()))
        CUDA_lib.flipCuda(cutorch.getState(), 
            input:data(), input:size(1), input:size(2),
            self.output:data(), false)

        return self.output
    end

    function nn.Flip:updateGradInput(input, gradOutput)
        self.gradInput:resize(input:size())
        assert(self.gradInput:isContiguous())

        gradOutput = gradOutput:view(-1, gradOutput:size(gradOutput:nDimension()))
        CUDA_lib.flipCuda(cutorch.getState(), 
            gradOutput:data(), gradOutput:size(1), gradOutput:size(2),
            self.gradInput:data(), self._accumulateGradInput)

        return self.gradInput
    end
end
