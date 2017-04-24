require 'nn'

ffi = require 'ffi'

local _, parent = torch.class('IntegralSmartNorm', 'nn.Module')

ffi.cdef [[

void forwardNoNorm(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr);

void forwardNoNormFrac(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride);

void backwardNoNormFrac(
    float *intData, float *gradOutData, int h, int w, float *deltas,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride);

void backwardNoNorm(
    float *intData, float *gradOutData, int h, int w, float *deltas,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr); ]]

local C_lib = ffi.load('C/lib/libintegral-c.so')

ffi.cdef [[
void forwardCudaNoNorm(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax);

void forwardCudaSingle(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr);

void backwardCudaSingle(
    float *intData, float *gradOutData, float *tmpArray, float *tmpArraySum, int h, int w, 
    float *deltas, int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr); ]]

local CUDA_lib

if pcall(require, 'cutorch') then
    local CUDA_lib = ffi.load('C/lib/libintegral-cuda.so')
end

do
    cv = require 'cv'
    require 'cv.imgproc'
    require 'cv.highgui'

    require 'nn'

    -- to be defined below
    local updateOutputCPU, accGradParametersCPU
    local updateOutputGPU, accGradParametersGPU

    function IntegralSmartNorm:__init(nWindows, h, w)
        parent.__init(self)
        self.nWindows, self.h, self.w = nWindows, h, w
        
        self.outputNonNorm = torch.FloatTensor(nWindows, h, w)
        
        self.integralDouble = torch.DoubleTensor()
        self.integral = torch.FloatTensor()

        -- the only parameters of the module: box filter anchor and size
        self.xMin = torch.FloatTensor(self.nWindows)
        self.yMin = torch.FloatTensor(self.nWindows)
        self.xMax = torch.FloatTensor(self.nWindows)
        self.yMax = torch.FloatTensor(self.nWindows)

        -- loss gradients wrt module's parameters
        self.gradXMin = torch.FloatTensor(self.nWindows):zero()
        self.gradYMin = torch.FloatTensor(self.nWindows):zero()
        self.gradXMax = torch.FloatTensor(self.nWindows):zero()
        self.gradYMax = torch.FloatTensor(self.nWindows):zero()

        -- for smart normalization
        -- TODO efficient memory usage
        self.onesIntegral = cv.integral{torch.ones(h, w)}:float()
        self.outputOnes = torch.FloatTensor(nWindows, h, w)
        self.cdiv = nn.CDivTable()
        
        self:float() -- set self.updateOutput, self.accGradParameters and self._type
        self:reset()

        if self.gradInput then
            self.gradInput = self.gradInput:float()
        end

        -- dirty fix (see dirtyFixWindows()) parameters
        self.maxX, self.maxY = 64, 64 -- Stanford Background settings
    end

    -- define custom way of transferring the module to GPU
    function IntegralSmartNorm:type(type, tensorCache)
        if not type then
            return self._type
        end

        if type == 'torch.DoubleTensor' then
            error(
                'Sorry, Integral() in double precision is not yet fully implemented. ' ..
                'Maybe you can help? https://github.com/shrubb/integral-layer')
        end

        if type == 'torch.CudaTensor' then
            -- io.stdout:write('warm '); io.stdout:flush()
            torch.CudaTensor(4,4) -- warm up
            -- print('up')

            self.updateOutput = updateOutputGPU
            self.accGradParameters = accGradParametersGPU
            self.tmpArrayGPU = torch.CudaTensor(self.h, self.w)
            self.tmpArraySumGPU = torch.CudaTensor(self.h, self.w)
            self.integralCuda = torch.CudaTensor()
        else
            self.updateOutput = updateOutputCPU
            self.accGradParameters = accGradParametersCPU
            self.tmpArrayGPU = nil
            self.tmpArraySumGPU = nil
            self.integralCuda = nil
        end

        tensorCache = tensorCache or {}

        -- convert only specified tensors
        -- maybe finally replace this with `self:type(type, tensorCache)`
        -- remaining:
        -- `grad...`, `integral`, `integralCuda`, `integralDouble`, `tmpArrayGPU`, `tmpArraySumGPU`

        -- io.stdout:write('warm '); io.stdout:flush()
        for _,param in ipairs{
                'outputNonNorm', 'gradInput', 'xMin', 'xMax', 'yMin', 'yMax', 'areaCoeff',
                'gradXMin', 'gradXMax', 'gradYMin', 'gradYMax', 'onesIntegral',
                'outputOnes', 'cdiv'} do
            self[param] = nn.utils.recursiveType(self[param], type, tensorCache)
        end
        -- print('up')

        if self.backpropHelper then
            self.backpropHelper:type(type, tensorCache)
        end
        
        self._type = type
        return self
    end

    -- overload
    function IntegralSmartNorm:write(file)
        file:writeObject(self.nWindows)
        file:writeObject(self.h)
        file:writeObject(self.w)
        file:writeObject(self.xMin)
        file:writeObject(self.xMax)
        file:writeObject(self.yMin)
        file:writeObject(self.yMax)
    end

    -- overload
    function IntegralSmartNorm:read(file)
        local nWindows = file:readObject()
        local h = file:readObject()
        local w = file:readObject()
        
        self:__init(nWindows, h, w)
        self.xMin = file:readObject()
        self.xMax = file:readObject()
        self.yMin = file:readObject()
        self.yMax = file:readObject()

        self:type(self.xMin:type())
    end

    function IntegralSmartNorm:reset()
        -- the only parameters of the module. Randomly initialize them
        self.xMin:rand(self.nWindows):add(-0.64):mul(2 * self.h * 0.14)
        self.yMin:rand(self.nWindows):add(-0.64):mul(2 * self.w * 0.14)
        
        for i = 1,self.nWindows do
            self.xMax[i] = torch.uniform(self.xMin[i] + self.h * 0.05, self.xMin[i] + self.h * 0.25)
            self.yMax[i] = torch.uniform(self.yMin[i] + self.w * 0.05, self.yMin[i] + self.w * 0.25)
        end
        
        -- loss gradients wrt module's parameters
        self.gradXMin:zero()
        self.gradYMin:zero()
        self.gradXMax:zero()
        self.gradYMax:zero()
    end

    function IntegralSmartNorm:parameters()
        local params = {self.xMin, self.xMax, self.yMin, self.yMax}
        local gradParams = {self.gradXMin, self.gradXMax, self.gradYMin, self.gradYMax}
        return params, gradParams
    end

    local function round_down(x)
        local rounded = math.floor(x)
        return rounded, x-rounded -- return integer and fractional parts
    end

    local function round_up(x)
        local rounded = math.ceil(x)
        return rounded, rounded-x -- return integer and fractional parts
    end

    local function dirtyFixWindows(self)
        -- dirty fix: don't let windows explode
        for _, borderKind in pairs{'Min', 'Max'} do
            self['x' .. borderKind]:clamp(-self.maxX, self.maxX)
            self['y' .. borderKind]:clamp(-self.maxY, self.maxY)
        end

        -- dirty fix: don't let windows become thinner than 1px
        for i = 1,self.nWindows do
            if self.xMin[i] > self.xMax[i] then
                if math.abs(self.xMin[i]) > math.abs(self.xMax[i]) then
                    self.xMax[i] = self.xMin[i] + 0.5
                else
                    self.xMin[i] = self.xMax[i] - 0.5
                end
            end

            if self.yMin[i] > self.yMax[i] then
                if math.abs(self.yMin[i]) > math.abs(self.yMax[i]) then
                    self.yMax[i] = self.yMin[i] + 0.5
                else
                    self.yMin[i] = self.yMax[i] - 0.5
                end
            end
        end
    end

    function updateOutputCPU(self, input)
        dirtyFixWindows(self)

        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):type(self._type):forward(input)
        end
        
        assert(input:size(2) == self.h and input:size(3) == self.w)

        self.outputNonNorm:resize(input:size(1)*self.nWindows, input:size(2), input:size(3))
        
        self.integralDouble:resize(input:size(1), input:size(2)+1, input:size(3)+1)
        self.integral:resize(self.integralDouble:size())

        -- first, compute non-normalized box filter map (into self.outputOnes) of 1-s
        do
            -- we put the result in the first plane
            local outputOnesSingle = self.outputOnes[{{1, self.nWindows}, {}, {}}]

            self.outputOnes:resize(input:size(1)*self.nWindows, input:size(2), input:size(3))

            -- do it just for one "input window"
            for nWindow = 1,self.nWindows do    
                -- Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
                -- `integral()` behavior. Namely, I(x,0) and I(0,y) are
                -- always 0 (so it's a C-style array sum).

                -- However, when computing sums, we subtract values at points 
                -- like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
                -- and yMin, and thus finally they are not affected.
                
                local xMinCurr, xMinCurrFrac = round_up  (self.xMin[nWindow])
                local xMaxCurr, xMaxCurrFrac = round_down(self.xMax[nWindow]+1)
                local yMinCurr, yMinCurrFrac = round_up  (self.yMin[nWindow])
                local yMaxCurr, yMaxCurrFrac = round_down(self.yMax[nWindow]+1)
                
                local outPlaneIdx = nWindow
                
                assert(self.outputOnes:stride(2) == self.w)
                local outData = torch.data(self.outputOnes[outPlaneIdx])
                local intData = torch.data(self.onesIntegral)

                if self.exact then
                    local ones = torch.ones(self.h, self.w)
                    C_lib.forwardNoNormFrac(
                        intData, self.h, self.w, outData, 
                        xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                        xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac,
                        torch.data(ones), ones:stride(1))
                else
                    C_lib.forwardNoNorm(
                        intData, self.h, self.w, outData, 
                        xMinCurr, xMaxCurr, yMinCurr, yMaxCurr)
                end
            end

            -- replace zeros with ones to avoid division-by-zero errors
            outputOnesSingle[outputOnesSingle:eq(0)] = 1

            -- then copy this result to all other output planes
            for inPlaneIdx = 2,input:size(1) do
                local outWindows = {self.nWindows*(inPlaneIdx-1) + 1, self.nWindows*inPlaneIdx}
                self.outputOnes[{outWindows, {}, {}}]:copy(outputOnesSingle)
            end
        end

        -- next, compute non-normalized box filter map (into self.outputNonNorm) from input
        do
            for inPlaneIdx = 1,input:size(1) do
                cv.integral{input[inPlaneIdx], self.integralDouble[inPlaneIdx]}
                self.integral[inPlaneIdx]:copy(self.integralDouble[inPlaneIdx]) -- cast
            
                for nWindow = 1,self.nWindows do
                    
                    -- Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
                    -- `integral()` behavior. Namely, I(x,0) and I(0,y) are
                    -- always 0 (so it's a C-style array sum).

                    -- However, when computing sums, we subtract values at points 
                    -- like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
                    -- and yMin, and thus finally they are not affected.
                    
                    local xMinCurr, xMinCurrFrac = round_up  (self.xMin[nWindow])
                    local xMaxCurr, xMaxCurrFrac = round_down(self.xMax[nWindow]+1)
                    local yMinCurr, yMinCurrFrac = round_up  (self.yMin[nWindow])
                    local yMaxCurr, yMaxCurrFrac = round_down(self.yMax[nWindow]+1)
                    
                    local outPlaneIdx = self.nWindows*(inPlaneIdx-1) + nWindow
                    
                    assert(self.outputNonNorm:stride(2) == self.w)
                    local outData = torch.data(self.outputNonNorm[outPlaneIdx])
                    local intData = torch.data(self.integral[inPlaneIdx])

                    if self.exact then
                        local inData = torch.data(input[inPlaneIdx])
                        C_lib.forwardNoNormFrac(
                            intData, self.h, self.w, outData, 
                            xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                            xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac, 
                            inData, input:stride(2))
                    else
                        C_lib.forwardNoNorm(
                            intData, self.h, self.w, outData, 
                            xMinCurr, xMaxCurr, yMinCurr, yMaxCurr)
                    end
                end
            end
        end

        -- divide elementwise to get normalized box filter maps
        self.output = self.cdiv:forward {self.outputNonNorm, self.outputOnes}
        
        return self.output
    end

    function updateOutputGPU(self, input)
        dirtyFixWindows(self)

        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):type(self._type):forward(input)
        end
        
        assert(input:size(2) == self.h and input:size(3) == self.w)

        self.outputNonNorm:resize(input:size(1)*self.nWindows, input:size(2), input:size(3))
        
        self.integralDouble:resize(input:size(1), input:size(2)+1, input:size(3)+1)
        self.integral:resize(self.integralDouble:size()) -- not used here
        self.integralCuda:resize(self.integralDouble:size())

        -- first, compute non-normalized box filter map (into self.outputOnes) of 1-s        
        do
            -- we put the result in the first plane
            local outputOnesSingle = self.outputOnes[{{1, self.nWindows}, {}, {}}]

            self.outputOnes:resize(input:size(1)*self.nWindows, input:size(2), input:size(3))

            local outData = torch.data(self.outputOnes)
            local intData = torch.data(self.onesIntegral)
            
            CUDA_lib.forwardCudaNoNorm(
                intData, self.h, self.w, self.nWindows, outData, 
                torch.data(self.xMin), torch.data(self.xMax),
                torch.data(self.yMin), torch.data(self.yMax))

            -- replace zeros with ones to avoid division-by-zero errors
            outputOnesSingle[outputOnesSingle:eq(0)] = 1

            -- copy this result to all other output planes
            for inPlaneIdx = 2,input:size(1) do
                local outWindows = {self.nWindows*(inPlaneIdx-1) + 1, self.nWindows*inPlaneIdx}
                self.outputOnes[{outWindows, {}, {}}]:copy(outputOnesSingle)
            end
        end

        -- next, compute non-normalized box filter map (into self.outputNonNorm) from input
        do
            for inPlaneIdx = 1,input:size(1) do

                cv.integral{input[inPlaneIdx]:float(), self.integralDouble[inPlaneIdx]}
                self.integralCuda[inPlaneIdx]:copy(self.integralDouble[inPlaneIdx]) -- cast and copy to GPU
                
                local outPlaneIdx = 1 + self.nWindows*(inPlaneIdx-1)

                local intData = torch.data(self.integralCuda[inPlaneIdx])
                local outData = torch.data(self.outputNonNorm[outPlaneIdx])
                
                CUDA_lib.forwardCudaNoNorm(
                    intData, self.h, self.w, self.nWindows, outData, 
                    torch.data(self.xMin), torch.data(self.xMax),
                    torch.data(self.yMin), torch.data(self.yMax))
            end
        end

        -- divide elementwise to get normalized box filter maps
        self.output = self.cdiv:forward {self.outputNonNorm, self.outputOnes}
        
        return self.output
    end

    -- that rare case when it's useful to redefine :backward()
    function IntegralSmartNorm:backward(input, gradOutput, scale)
        scale = scale or 1

        -- recover the gradient from division
        gradOutput = self.cdiv:updateGradInput({self.outputNonNorm, self.outputOnes}, gradOutput)[1]

        -- substitute incoming gradOutput with it
        self:updateGradInput(input, gradOutput)
        self:accGradParameters(input, gradOutput, scale)
        return self.gradInput
    end

    function IntegralSmartNorm:updateGradInput(input, gradOutput)
        if self.gradInput then
            
            if input:nDimension() == 2 then
                input = nn.Unsqueeze(1):type(self._type):forward(input)
            end

            -- never call :backward() on backpropHelper!
            -- Otherwise you'll get into infinite recursion
            self.backpropHelper = self.backpropHelper or IntegralSmartNorm(1, self.h, self.w):type(self._type)
            self.backpropHelper.exact = self.exact
        
            self.gradInput:resize(input:size()):zero()
            
            for inPlaneIdx = 1,input:size(1) do
                for nWindow = 1,self.nWindows do
                    self.backpropHelper.xMin[1] = -self.xMax[nWindow]
                    self.backpropHelper.xMax[1] = -self.xMin[nWindow]
                    self.backpropHelper.yMin[1] = -self.yMax[nWindow]
                    self.backpropHelper.yMax[1] = -self.yMin[nWindow]
                    
                    local outPlaneIdx = self.nWindows*(inPlaneIdx-1) + nWindow

                    self.gradInput[inPlaneIdx]:add(
                        self.backpropHelper:forward(gradOutput[outPlaneIdx]):squeeze())
                end
            end
            
            return self.gradInput
        end
    end

    function accGradParametersCPU(self, input, gradOutput, scale)

        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):type(self._type):forward(input)
        end
        
        for inPlaneIdx = 1,input:size(1) do
            for nWindow = 1,self.nWindows do
                local outPlaneIdx = self.nWindows*(inPlaneIdx-1) + nWindow
                -- local outputDot = torch.dot(self.outputNonNorm[outPlaneIdx], gradOutput[outPlaneIdx])
                
                local xMinCurr, xMinCurrFrac = round_up  (self.xMin[nWindow]-1)
                local xMaxCurr, xMaxCurrFrac = round_down(self.xMax[nWindow]+1)
                local yMinCurr, yMinCurrFrac = round_up  (self.yMin[nWindow]-1)
                local yMaxCurr, yMaxCurrFrac = round_down(self.yMax[nWindow]+1)

                assert(gradOutput:stride(2) == self.w)
                local gradOutData = torch.data(gradOutput[outPlaneIdx])
                local intData = torch.data(self.integral[inPlaneIdx])
                
                -- deltas of dOut(x,y) (sum over one window)
                local deltas = ffi.new('float[4]')
                
                if self.exact then
                    local inData = torch.data(input[inPlaneIdx])
                    C_lib.backwardNoNormFrac(
                        intData, gradOutData, self.h, self.w, deltas,
                        xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                        xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac,
                        inData, input:stride(2))
                else
                    C_lib.backwardNoNorm(
                        intData, gradOutData, self.h, self.w, deltas,
                        xMinCurr, xMaxCurr, yMinCurr, yMaxCurr)
                end

                local xMinDelta, xMaxDelta = deltas[0], deltas[1]
                local yMinDelta, yMaxDelta = deltas[2], deltas[3]
                
                self.gradXMax[nWindow] = self.gradXMax[nWindow] + scale * xMaxDelta
                self.gradXMin[nWindow] = self.gradXMin[nWindow] + scale * xMinDelta
                self.gradYMax[nWindow] = self.gradYMax[nWindow] + scale * yMaxDelta
                self.gradYMin[nWindow] = self.gradYMin[nWindow] + scale * yMinDelta
            end
        end
    end

    function accGradParametersGPU(self, input, gradOutput, scale)

        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):type(self._type):forward(input)
        end

        -- we have `self.integralCuda`
        -- self.integral:copy(self.integralDouble) -- cast; TEMPORARY

        scale = scale or 1
        
        for inPlaneIdx = 1,input:size(1) do
            for nWindow = 1,self.nWindows do
                local outPlaneIdx = self.nWindows*(inPlaneIdx-1) + nWindow
                local outputDot = torch.dot(self.outputNonNorm[outPlaneIdx], gradOutput[outPlaneIdx])
                
                -- round towards zero (?)
                -- and +1 because OpenCV's integral adds extra row and col
                local xMinCurr = round_down(self.xMin[nWindow])
                local xMaxCurr = round_down(self.xMax[nWindow])
                local yMinCurr = round_down(self.yMin[nWindow])
                local yMaxCurr = round_down(self.yMax[nWindow])
                
                -- deltas of dOut(x,y) (sum over one window)
                local deltas = torch.FloatTensor(4)
                
                CUDA_lib.backwardCudaSingle(
                    torch.data(self.integralCuda[inPlaneIdx]), 
                    torch.data(gradOutput[outPlaneIdx]), 
                    torch.data(self.tmpArrayGPU),
                    torch.data(self.tmpArraySumGPU),
                    self.h, self.w, torch.data(deltas),
                    xMinCurr, xMaxCurr, yMinCurr, yMaxCurr)

                local xMinDelta, xMaxDelta = deltas[1], deltas[2]
                local yMinDelta, yMaxDelta = deltas[3], deltas[4]
                
                self.gradXMax[nWindow] = self.gradXMax[nWindow] + scale * xMaxDelta
                self.gradXMin[nWindow] = self.gradXMin[nWindow] + scale * xMinDelta
                self.gradYMax[nWindow] = self.gradYMax[nWindow] + scale * yMaxDelta
                self.gradYMin[nWindow] = self.gradYMin[nWindow] + scale * yMinDelta
            end
        end
    end

    function IntegralSmartNorm:zeroGradParameters()
        self.gradXMin:zero()
        self.gradYMin:zero()
        self.gradXMax:zero()
        self.gradYMax:zero()
    end

    function IntegralSmartNorm:updateParameters(lr)
        self.xMin:add(lr, self.gradXMin)
        self.yMin:add(lr, self.gradYMin)
        self.xMax:add(lr, self.gradXMax)
        self.yMax:add(lr, self.gradYMax)
    end
end