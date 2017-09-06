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

void forwardNoNormReplicate(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr);

void forwardNoNormReplicateFrac(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride);

void updateGradInput(
    float *gradOutputInt, int channels, int h, int w,float *gradInput,
    int *xMin, int *xMax, int *yMin, int *yMax,
    float *gradOutput, int gradOutputStride);

void updateGradInputFrac(
    float *gradOutputInt, int channels, int h, int w, float *gradInput,
    int *xMin, int *xMax, int *yMin, int *yMax,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *gradOutput, int gradOutputStride);

void backwardNoNorm(
    float *intData, float *gradOutData, float scale, int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt);

void backwardNoNormFrac(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *inData, int inStrideRow); ]]

local C_lib = ffi.load('C/lib/libintegral-c.so')

ffi.cdef [[
void forwardCudaNoNorm(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax);

void forwardCudaNoNormFrac(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStride);

void backwardCudaSingle(
    float *intData, float *gradOutData, float *tmpArray, float *tmpArraySum, int h, int w, 
    float *deltas, int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr);

void backwardCudaSingleFrac(
    float *intData, float *gradOutData, float *tmpArray, float *tmpArraySum, int h, int w, 
    float *deltas, int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride);

void integralImageCuda(float *input, float *output, int channels, int h, int w, float *tmp);
void integralImageInplaceCuda(float *input, float *output, int channels, int h, int w);

void _initCublasHandle(); ]]

local CUDA_lib

if pcall(require, 'cutorch') then
    CUDA_lib = ffi.load('C/lib/libintegral-cuda.so')
    CUDA_lib._initCublasHandle();
end

do
    cv = require 'cv'
    require 'cv.imgproc'
    require 'cv.highgui'

    require 'nn'

    -- to be defined below
    local updateOutputCPU, accGradParametersCPU
    local updateOutputGPU, accGradParametersGPU

    function IntegralSmartNorm:__init(nInputPlane, nWindows, h, w)
        -- nWindows is the number of box filters per channel
        parent.__init(self)
        self.nInputPlane, self.nWindows, self.h, self.w = nInputPlane, nWindows, h, w
        
        self.outputNonNorm = torch.FloatTensor()
        
        self.integralDouble = torch.DoubleTensor()
        self.integral = torch.FloatTensor()
        self.integralGradOutput = torch.FloatTensor() -- (nInputPlane) x (h+1) x (w+1)

        -- the only parameters of the module: box filter anchor and size
        self.xMin = torch.FloatTensor(nInputPlane, nWindows)
        self.yMin = torch.FloatTensor(nInputPlane, nWindows)
        self.xMax = torch.FloatTensor(nInputPlane, nWindows)
        self.yMax = torch.FloatTensor(nInputPlane, nWindows)

        -- when computing exact forward/backward passes, will need 
        -- these tmp arrays for int and frac parts of xMin/xMax/yMin/yMax
        self.xMinInt, self.xMaxInt = self.xMin:int(), self.xMin:int()
        self.yMinInt, self.yMaxInt = self.xMin:int(), self.xMin:int()
        self.xMinFrac, self.xMaxFrac = self.xMin:clone(), self.xMin:clone()
        self.yMinFrac, self.yMaxFrac = self.xMin:clone(), self.xMin:clone()

        -- loss gradients wrt module's parameters
        self.gradXMin = torch.FloatTensor(nInputPlane, nWindows):zero()
        self.gradYMin = torch.FloatTensor(nInputPlane, nWindows):zero()
        self.gradXMax = torch.FloatTensor(nInputPlane, nWindows):zero()
        self.gradYMax = torch.FloatTensor(nInputPlane, nWindows):zero()

        -- for smart normalization
        self.ones = torch.ones(h, w)
        self.onesIntegral = cv.integral{torch.ones(h, w)}:float() --torch.ones(h, w):float()
        self.outputOnes = torch.FloatTensor(nInputPlane*nWindows, h, w)
        self.cdiv = nn.CDivTable()
        
        self:float() -- set self.updateOutput, self.accGradParameters and self._type
        self:reset()

        self.gradInput = torch.FloatTensor()

        -- if false, rounds window borders and thus ignores fractional box parts
        -- (this is obviously faster, set this to false in production)
        self.exact = true
        -- if true, additionally divides the output the box sum by its area
        self.normalize = true
        -- if false, simply divides the box sum by its area
        -- if true, divides it by the "valid" area (= inside the image) under the box
        self.smart = true -- hard switch for now
        -- specifies how to treat pixels outside of the image
        -- if true, acts like BORDER_MODE_REPLICATE in OpenCV
        -- if false, treats those pixels as zeros (BORDER_MODE_CONSTANT with a value of 0)
        self.replicate = true

        -- has `self.cdiv:updateGradInput()` already been done for current input?
        self._backwardDone = false
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
            self.updateOutput = updateOutputGPU
            self.accGradParameters = accGradParametersGPU
            self.tmpArrayGPU = torch.CudaTensor(self.h, self.w) -- (nInputPlane) x (h+1) x (w+1)
            self.tmpArraySumGPU = torch.CudaTensor(self.h, self.w)
            self.integralCuda = torch.CudaTensor() -- (nInputPlane) x (h+1) x (w+1)

            for _, param in ipairs{'xMinInt', 'xMaxInt', 'yMinInt', 'yMaxInt'} do
                self[param] = self[param]:cudaInt()
            end
        else
            self.updateOutput = updateOutputCPU
            self.accGradParameters = accGradParametersCPU
            self.tmpArrayGPU = nil
            self.tmpArraySumGPU = nil
            self.integralCuda = nil
            
            for _, param in ipairs{'xMinInt', 'xMaxInt', 'yMinInt', 'yMaxInt'} do
                self[param] = self[param]:int()
            end
        end

        tensorCache = tensorCache or {}

        -- convert only specified tensors
        -- maybe finally replace this with `self:type(type, tensorCache)`
        -- remaining:
        -- `integral`, `integralCuda`, `integralDouble`, `tmpArrayGPU`, `tmpArraySumGPU`,
        -- `xMinInt`, `xMaxInt`, `yMinInt`, `yMaxInt`

        for _, param in ipairs{
                'outputNonNorm', 'gradInput', 'xMin', 'xMax', 'yMin', 'yMax', 'areaCoeff',
                'gradXMin', 'gradXMax', 'gradYMin', 'gradYMax', 'onesIntegral', 'ones',
                'outputOnes', 'cdiv', 'xMinFrac', 'xMaxFrac', 'yMinFrac', 'yMaxFrac',
                'integralGradOutput'} do
            self[param] = nn.utils.recursiveType(self[param], type, tensorCache)
        end

        if self.backpropHelper then
            self.backpropHelper:type(type, tensorCache)
        end
        
        self._type = type
        return self
    end

    -- overload
    function IntegralSmartNorm:write(file)
        file:writeObject(self.nInputPlane)
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
        local nInputPlane = file:readObject()
        local nWindows    = file:readObject()
        local h           = file:readObject()
        local w           = file:readObject()
        
        self:__init(nInputPlane, nWindows, h, w)
        self.xMin = file:readObject()
        self.xMax = file:readObject()
        self.yMin = file:readObject()
        self.yMax = file:readObject()

        self:type(self.xMin:type())
    end

    function IntegralSmartNorm:reset()
        -- the only parameters of the module. Randomly initialize them
        self.xMin:rand(self.nInputPlane, self.nWindows):add(-0.64):mul(2 * self.h * 0.14)
        self.yMin:rand(self.nInputPlane, self.nWindows):add(-0.64):mul(2 * self.w * 0.14)
        
        do
            local xMin, xMax = self.xMin:view(-1), self.xMax:view(-1)
            local yMin, yMax = self.yMin:view(-1), self.yMax:view(-1)

            for i = 1,xMax:nElement() do
                xMax[i] = torch.uniform(
                    xMin[i] + self.h * 0.05,
                    xMin[i] + self.h * 0.25)
                yMax[i] = torch.uniform(
                    yMin[i] + self.w * 0.05,
                    yMin[i] + self.w * 0.25)
            end
        end -- do
        
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
        -- dirty fix 1: don't let windows go outside the image, otherwise gradParams will vanish
        self.xMin:clamp(-self.h+1, self.h-1)
        self.xMax:clamp(-self.h+1, self.h-1)
        self.yMin:clamp(-self.w+1, self.w-1)
        self.yMax:clamp(-self.w+1, self.w-1)

        -- dirty fix 2: don't let windows become thinner than 1px (or 2.01 px, if in non-exact mode)
        local xMin, xMax = self.xMin:view(-1), self.xMax:view(-1)
        local yMin, yMax = self.yMin:view(-1), self.yMax:view(-1)

        -- if self.exact then
        --     for i = 1,xMax:nElement() do
        --         if xMin[i] > xMax[i] then
        --             if math.abs(xMin[i]) > math.abs(xMax[i]) then
        --                 xMax[i] = xMin[i] + 0.5
        --             else
        --                 xMin[i] = xMax[i] - 0.5
        --             end
        --         end

        --         if yMin[i] > yMax[i] then
        --             if math.abs(yMin[i]) > math.abs(yMax[i]) then
        --                 yMax[i] = yMin[i] + 0.5
        --             else
        --                 yMin[i] = yMax[i] - 0.5
        --             end
        --         end
        --     end
        -- else
            for i = 1,xMax:nElement() do
                local minWidth = 1

                if xMin[i] + minWidth - 0.99 > xMax[i] then
                    local mean = 0.5 * (xMin[i] + xMax[i])
                    xMin[i] = mean - (minWidth - 0.9) / 2
                    xMax[i] = mean + (minWidth - 0.9) / 2
                end

                if yMin[i] + minWidth - 0.99 > yMax[i] then
                    local mean = 0.5 * (yMin[i] + yMax[i])
                    yMin[i] = mean - (minWidth - 0.9) / 2
                    yMax[i] = mean + (minWidth - 0.9) / 2
                end
            end
        -- end
    end

    function updateOutputCPU(self, input)
        dirtyFixWindows(self)

        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):type(self._type):forward(input)
        end

        assert(input:size(1) == self.nInputPlane)
        assert(input:size(2) == self.h and input:size(3) == self.w)
        
        self.integralDouble:resize(self.nInputPlane, self.h+1, self.w+1)
        self.integral:resize(self.integralDouble:size())

        -- first, compute non-normalized box filter map (into self.outputOnes) of 1-s
        if self.normalize then
            assert(self.outputOnes:stride(2) == self.w) -- for C function safety

            local xMin, xMax = self.xMin:view(-1), self.xMax:view(-1)
            local yMin, yMax = self.yMin:view(-1), self.yMax:view(-1)

            for globalWindowIdx = 1,self.nInputPlane*self.nWindows do
                -- Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
                -- `integral()` behavior. Namely, I(x,0) and I(0,y) are
                -- always 0 (so it's a C-style array sum).

                -- However, when computing sums, we subtract values at points 
                -- like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
                -- and yMin, and thus finally they are not affected.
                local xMinCurr, xMinCurrFrac = round_up  (xMin[globalWindowIdx])
                local xMaxCurr, xMaxCurrFrac = round_down(xMax[globalWindowIdx]+1)
                local yMinCurr, yMinCurrFrac = round_up  (yMin[globalWindowIdx])
                local yMaxCurr, yMaxCurrFrac = round_down(yMax[globalWindowIdx]+1)

                -- TODO: efficient memory usage
                local outData = torch.data(self.outputOnes[globalWindowIdx])
                local intData = torch.data(self.onesIntegral)

                -- TODO: multi-window C_lib.forwardNoNorm[Frac] for speed
                local forwardCFunction

                if self.exact then
                    if self.replicate then
                        forwardCFunction = C_lib.forwardNoNormReplicateFrac
                    else
                        forwardCFunction = C_lib.forwardNoNormFrac
                    end

                    forwardCFunction(
                        intData, self.h, self.w, outData, 
                        xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                        xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac,
                        torch.data(self.ones), self.ones:stride(1))
                else
                    if self.replicate then
                        forwardCFunction = C_lib.forwardNoNormReplicate 
                    else
                        forwardCFunction = C_lib.forwardNoNorm
                    end

                    forwardCFunction(
                        intData, self.h, self.w, outData, 
                        xMinCurr, xMaxCurr, yMinCurr, yMaxCurr)
                end
            end

            -- replace zeros with ones to avoid division-by-zero errors
            -- (no need to do it anymore, I hope?)

            -- then copy this result to all other output planes
            -- (we don't need this as we're doing expansion later)
            -- for inPlaneIdx = 2,input:size(1) do
            --     local outWindows = {self.nWindows*(inPlaneIdx-1) + 1, self.nWindows*inPlaneIdx}
            --     self.outputOnes[{outWindows, {}, {}}]:copy(outputOnesSingle)
            -- end
        end

        -- next, compute non-normalized box filter map of `input` into self.outputNonNorm
        do
            self.outputNonNorm:resize(self.nInputPlane, self.nWindows, self.h, self.w)
            assert(self.outputNonNorm:stride(3) == self.w) -- for C function safety

            for inPlaneIdx = 1,input:size(1) do
                cv.integral{input[inPlaneIdx], self.integralDouble[inPlaneIdx]}
                self.integral[inPlaneIdx]:copy(self.integralDouble[inPlaneIdx]) -- cast

                local xMin, xMax = self.xMin[inPlaneIdx], self.xMax[inPlaneIdx]
                local yMin, yMax = self.yMin[inPlaneIdx], self.yMax[inPlaneIdx]
                local outputNonNorm = self.outputNonNorm[inPlaneIdx]

                for windowIdx = 1,self.nWindows do
                    
                    -- Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
                    -- `integral()` behavior. Namely, I(x,0) and I(0,y) are
                    -- always 0 (so it's a C-style array sum).

                    -- However, when computing sums, we subtract values at points 
                    -- like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
                    -- and yMin, and thus finally they are not affected.
                    
                    local xMinCurr, xMinCurrFrac = round_up  (xMin[windowIdx])
                    local xMaxCurr, xMaxCurrFrac = round_down(xMax[windowIdx]+1)
                    local yMinCurr, yMinCurrFrac = round_up  (yMin[windowIdx])
                    local yMaxCurr, yMaxCurrFrac = round_down(yMax[windowIdx]+1)
                    
                    local outData = torch.data(outputNonNorm[windowIdx])
                    local intData = torch.data(self.integral[inPlaneIdx])

                    local forwardCFunction

                    if self.exact then
                        if self.replicate then
                            forwardCFunction = C_lib.forwardNoNormReplicateFrac
                        else
                            forwardCFunction = C_lib.forwardNoNormFrac
                        end

                        local inData = torch.data(input[inPlaneIdx])

                        forwardCFunction(
                            intData, self.h, self.w, outData, 
                            xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                            xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac, 
                            inData, input:stride(2))
                    else
                        if self.replicate then
                            forwardCFunction = C_lib.forwardNoNormReplicate
                        else
                            forwardCFunction = C_lib.forwardNoNorm
                        end

                        forwardCFunction(
                            intData, self.h, self.w, outData, 
                            xMinCurr, xMaxCurr, yMinCurr, yMaxCurr)
                    end
                end
            end

            self.outputNonNorm = 
                self.outputNonNorm:view(self.nInputPlane*self.nWindows, self.h, self.w)
        end

        if self.normalize then
            -- divide elementwise to get normalized box filter maps
            self.output = self.cdiv:forward {self.outputNonNorm, self.outputOnes}
        else
            self.output = self.outputNonNorm
        end
        
        self._backwardDone = false

        return self.output
    end

    function updateOutputGPU(self, input)
        error('NYI')

        dirtyFixWindows(self)

        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):type(self._type):forward(input)
        end
        
        assert(input:size(2) == self.h and input:size(3) == self.w)
        self.integralCuda:resize(input:size(1), input:size(2)+1, input:size(3)+1)

        -- first, compute non-normalized box filter map (into self.outputOnes) of 1-s        
        if self.normalize then
            -- we put the result in the first plane
            -- local outputOnesSingle = self.outputOnes[{{1, self.nWindows}, {}, {}}]
            assert(self.outputOnes:stride(2) == self.w) -- for C function safety

            local outData = torch.data(self.outputOnes)
            local intData = torch.data(self.onesIntegral)
            
            if self.exact then
                -- TODO get rid of `ones`
                local ones = torch.ones(self.h, self.w):cuda()
                CUDA_lib.forwardCudaNoNormFrac(
                    intData, self.h, self.w, self.nWindows, outData, 
                    torch.data(self.xMin), torch.data(self.xMax),
                    torch.data(self.yMin), torch.data(self.yMax),
                    torch.data(ones), ones:stride(1))
            else
                CUDA_lib.forwardCudaNoNorm(
                    intData, self.h, self.w, self.nWindows, outData, 
                    torch.data(self.xMin), torch.data(self.xMax),
                    torch.data(self.yMin), torch.data(self.yMax))
            end

            -- replace zeros with ones to avoid division-by-zero errors
            -- (no need to do it anymore, I hope?)

            -- copy this result to all other output planes
            -- for inPlaneIdx = 2,input:size(1) do
            --     local outWindows = {self.nWindows*(inPlaneIdx-1) + 1, self.nWindows*inPlaneIdx}
            --     self.outputOnes[{outWindows, {}, {}}]:copy(outputOnesSingle)
            -- end
        end

        -- next, compute non-normalized box filter map (into self.outputNonNorm) from input
        do
            if self.tmpArrayGPU:nElement() < self.integralCuda:nElement() then
                self.tmpArrayGPU:resize(self.integralCuda:nElement())
            end

            CUDA_lib.integralImageCuda(
                torch.data(input), torch.data(self.integralCuda),
                input:size(1), input:size(2), input:size(3),
                torch.data(self.tmpArrayGPU))

            for inPlaneIdx = 1,input:size(1) do
                self.outputNonNorm:resize(input:size(1)*self.nWindows, input:size(2), input:size(3))

                assert(self.outputNonNorm:stride(1) == self.w * self.h) -- for C function safety
                assert(self.outputNonNorm:stride(2) == self.w) -- for C function safety

                local outPlaneIdx = 1 + self.nWindows*(inPlaneIdx-1)

                local intData = torch.data(self.integralCuda[inPlaneIdx])
                local outData = torch.data(self.outputNonNorm[outPlaneIdx])
                
                local forwardCudaFunction

                if self.exact then
                    if self.replicate then
                        forwardCudaFunction = CUDA_lib.forwardCudaNoNormReplicateFrac
                    else
                        forwardCudaFunction = CUDA_lib.forwardCudaNoNormFrac
                    end

                    forwardCudaFunction(
                        intData, self.h, self.w, self.nWindows, outData, 
                        torch.data(self.xMin), torch.data(self.xMax),
                        torch.data(self.yMin), torch.data(self.yMax),
                        torch.data(input[inPlaneIdx]), input:stride(2))
                else
                    if self.replicate then
                        forwardCudaFunction = CUDA_lib.forwardCudaNoNormReplicate
                    else
                        forwardCudaFunction = CUDA_lib.forwardCudaNoNorm
                    end

                    forwardCudaFunction(
                        intData, self.h, self.w, self.nWindows, outData, 
                        torch.data(self.xMin), torch.data(self.xMax),
                        torch.data(self.yMin), torch.data(self.yMax))
                end
            end
        end

        if self.normalize then
            local outputNonNorm4D = splitFirstDim(self, self.outputNonNorm)
            local outputOnes4D = 
                nn.utils.addSingletonDimension(self.outputOnes):expandAs(outputNonNorm4D)
            
            -- divide elementwise to get normalized box filter maps
            self.output = self.cdiv:forward {outputNonNorm4D, outputOnes4D}
            self.output = flattenFirstDims(self, self.output)
        else
            self.output = self.outputNonNorm
        end

        self._backwardDone = false
        
        return self.output
    end

    function IntegralSmartNorm:updateGradInput(input, gradOutput)
        if self.gradInput then

            if self.normalize then
                if not self._backwardDone then
                    self.cdiv:updateGradInput({self.outputNonNorm, self.outputOnes}, gradOutput)
                    self._backwardDone = true
                end
                gradOutput = self.cdiv.gradInput[1]
            end
            
            if input:nDimension() == 2 then
                input = nn.Unsqueeze(1):type(self._type):forward(input)
            end

            self.gradInput:resize(input:size()):zero()
            gradOutput = gradOutput:view(self.nInputPlane, self.nWindows, self.h, self.w)

            if self._type == 'torch.CudaTensor' then
                for inPlaneIdx = 1,self.nInputPlane do
                    local xMin, xMax = self.xMin[inPlaneIdx], self.xMax[inPlaneIdx]
                    local yMin, yMax = self.yMin[inPlaneIdx], self.yMax[inPlaneIdx]

                    local xMinInt, xMinFrac = self.xMinInt[inPlaneIdx], self.xMinFrac[inPlaneIdx]
                    local xMaxInt, xMaxFrac = self.xMaxInt[inPlaneIdx], self.xMaxFrac[inPlaneIdx]
                    local yMinInt, yMinFrac = self.yMinInt[inPlaneIdx], self.yMinFrac[inPlaneIdx]
                    local yMaxInt, yMaxFrac = self.yMaxInt[inPlaneIdx], self.yMaxFrac[inPlaneIdx]

                    -- TODO pack this in CUDA code
                    for windowIdx = 1,self.nWindows do
                        xMinInt[windowIdx], xMinFrac[windowIdx] = round_up  (-xMax[windowIdx])
                        xMaxInt[windowIdx], xMaxFrac[windowIdx] = round_down(-xMin[windowIdx]+1)
                        yMinInt[windowIdx], yMinFrac[windowIdx] = round_up  (-yMax[windowIdx])
                        yMaxInt[windowIdx], yMaxFrac[windowIdx] = round_down(-yMin[windowIdx]+1)
                    end

                    -- integralCuda is (nInputPlane) x (h+1) x (w+1)
                    if self.tmpArrayGPU:nElement() < self.integralCuda:nElement() then
                        self.tmpArrayGPU:resize(self.integralCuda:nElement())
                    end

                    if self.integralGradOutput:nElement() < self.integralCuda:nElement() then
                        self.integralGradOutput:resize(self.integralCuda:nElement())
                    end

                    CUDA_lib.integralImageCuda(
                        torch.data(gradOutput[inPlaneIdx]), torch.data(self.integralGradOutput),
                        self.nWindows, self.h, self.w, self.tmpArrayGPU)

                    error('NYI')

                    if self.exact then
                        CUDA_lib.updateGradInputFrac()
                    else
                        CUDA_lib.updateGradInput()
                    end
                end
            else
                for inPlaneIdx = 1,self.nInputPlane do
                    -- gradInput of a conv is the conv of gradOutput with flipped kernels
                    -- so let's negate the parameters
                    local xMin, xMax = self.xMin[inPlaneIdx], self.xMax[inPlaneIdx]
                    local yMin, yMax = self.yMin[inPlaneIdx], self.yMax[inPlaneIdx]

                    local xMinInt, xMinFrac = self.xMinInt[inPlaneIdx], self.xMinFrac[inPlaneIdx]
                    local xMaxInt, xMaxFrac = self.xMaxInt[inPlaneIdx], self.xMaxFrac[inPlaneIdx]
                    local yMinInt, yMinFrac = self.yMinInt[inPlaneIdx], self.yMinFrac[inPlaneIdx]
                    local yMaxInt, yMaxFrac = self.yMaxInt[inPlaneIdx], self.yMaxFrac[inPlaneIdx]

                    for windowIdx = 1,self.nWindows do
                        xMinInt[windowIdx], xMinFrac[windowIdx] = round_up  (-xMax[windowIdx])
                        xMaxInt[windowIdx], xMaxFrac[windowIdx] = round_down(-xMin[windowIdx]+1)
                        yMinInt[windowIdx], yMinFrac[windowIdx] = round_up  (-yMax[windowIdx])
                        yMaxInt[windowIdx], yMaxFrac[windowIdx] = round_down(-yMin[windowIdx]+1)
                    end

                    -- compute integral image of gradOutput's planes corresponding to inPlaneIdx
                    -- into self.integralGradOutput
                    self.integralGradOutput:resize(self.nWindows, self.h+1, self.w+1)

                    for windowIdx = 1,self.nWindows do
                        cv.integral{gradOutput[inPlaneIdx][windowIdx], self.integralDouble[1]}
                        self.integralGradOutput[windowIdx]:copy(self.integralDouble[1]) -- cast
                    end

                    local updateGradInputCFunction

                    -- compute the needed integral sums
                    if self.exact then
                        if self.replicate then
                            updateGradInputCFunction = C_lib.updateGradInputFrac
                        else
                            error('NYI')
                        end

                        updateGradInputCFunction(
                            torch.data(self.integralGradOutput), self.nWindows,
                            self.h, self.w, torch.data(self.gradInput[inPlaneIdx]),
                            torch.data(xMinInt), torch.data(xMaxInt),
                            torch.data(yMinInt), torch.data(yMaxInt),
                            torch.data(xMinFrac), torch.data(xMaxFrac),
                            torch.data(yMinFrac), torch.data(yMaxFrac),
                            torch.data(gradOutput[inPlaneIdx]), gradOutput:stride(3))
                    else
                        if self.replicate then
                            updateGradInputCFunction = C_lib.updateGradInput
                        else
                            error('NYI')
                        end

                        updateGradInputCFunction(
                            torch.data(self.integralGradOutput), self.nWindows,
                            self.h, self.w, torch.data(self.gradInput[inPlaneIdx]),
                            torch.data(xMinInt), torch.data(xMaxInt),
                            torch.data(yMinInt), torch.data(yMaxInt),
                            torch.data(gradOutput[inPlaneIdx]), gradOutput:stride(3))
                    end
                end
            end

            return self.gradInput
        end
    end

    function accGradParametersCPU(self, input, gradOutput, scale)
        if self.normalize then
            if not self._backwardDone then
                self.cdiv:updateGradInput({self.outputNonNorm, self.outputOnes}, gradOutput)
                self._backwardDone = true
            end
            gradOutput = self.cdiv.gradInput[1]
        end

        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):type(self._type):forward(input)
        end

        for k = 1,(self.normalize and 2 or 1) do
            -- iteration 1: gradient by outputNonNorm
            -- iteration 2: gradient by outputOnes
            local gradOutput = self.normalize and self.cdiv.gradInput[k] or gradOutput
            gradOutput = gradOutput:view(self.nInputPlane, self.nWindows, self.h, self.w)

            assert(gradOutput:stride(gradOutput:nDimension()-1) == self.w) -- for C function

            for inPlaneIdx = 1,self.nInputPlane do

                local xMin, xMax = self.xMin[inPlaneIdx], self.xMax[inPlaneIdx]
                local yMin, yMax = self.yMin[inPlaneIdx], self.yMax[inPlaneIdx]

                local gradXMin, gradXMax = self.gradXMin[inPlaneIdx], self.gradXMax[inPlaneIdx]
                local gradYMin, gradYMax = self.gradYMin[inPlaneIdx], self.gradYMax[inPlaneIdx]

                local xMinInt, xMinFrac = self.xMinInt[inPlaneIdx], self.xMinFrac[inPlaneIdx]
                local xMaxInt, xMaxFrac = self.xMaxInt[inPlaneIdx], self.xMaxFrac[inPlaneIdx]
                local yMinInt, yMinFrac = self.yMinInt[inPlaneIdx], self.yMinFrac[inPlaneIdx]
                local yMaxInt, yMaxFrac = self.yMaxInt[inPlaneIdx], self.yMaxFrac[inPlaneIdx]

                for windowIdx = 1,self.nWindows do
                    xMinInt[windowIdx], xMinFrac[windowIdx] = round_up  (xMin[windowIdx]-1)
                    xMaxInt[windowIdx], xMaxFrac[windowIdx] = round_down(xMax[windowIdx]  )
                    yMinInt[windowIdx], yMinFrac[windowIdx] = round_up  (yMin[windowIdx]-1)
                    yMaxInt[windowIdx], yMaxFrac[windowIdx] = round_down(yMax[windowIdx]  )
                end

                local gradOutData = torch.data(gradOutput[inPlaneIdx])
                local intData
                if k == 1 then
                    intData = torch.data(self.integral[inPlaneIdx])
                else
                    intData = torch.data(self.onesIntegral)
                end
                
                if self.exact then
                    local inData, inStrideRow
                    if k == 1 then
                        inData = torch.data(input[inPlaneIdx])
                        inStrideRow = input:stride(input:nDimension()-1)
                    else -- k == 2
                        inData = torch.data(self.ones)
                        inStrideRow = self.ones:stride(1)
                    end

                    C_lib.backwardNoNormFrac(
                        intData, gradOutData, scale,
                        self.nWindows, self.h, self.w,
                        torch.data(gradXMin), torch.data(gradXMax),
                        torch.data(gradYMin), torch.data(gradYMax),
                        torch.data(xMinInt), torch.data(xMaxInt),
                        torch.data(yMinInt), torch.data(yMaxInt),
                        torch.data(xMinFrac), torch.data(xMaxFrac),
                        torch.data(yMinFrac), torch.data(yMaxFrac),
                        inData, inStrideRow)
                else
                    C_lib.backwardNoNorm(
                        intData, gradOutData, scale,
                        self.nWindows, self.h, self.w,
                        torch.data(gradXMin), torch.data(gradXMax),
                        torch.data(gradYMin), torch.data(gradYMax),
                        torch.data(xMinInt), torch.data(xMaxInt),
                        torch.data(yMinInt), torch.data(yMaxInt))
                end
            end
        end
    end

    function accGradParametersGPU(self, input, gradOutput, scale)
        error('NYI')
        
        if self.normalize then
            if not self._backwardDone then
                self.cdiv:updateGradInput({self.outputNonNorm, self.outputOnes}, gradOutput)
                self._backwardDone = true
            end
            gradOutput = self.cdiv.gradInput[1]
        end

        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):type(self._type):forward(input)
        end

        for k = 1,(self.normalize and 2 or 1) do
            -- iteration 1: gradient by outputNonNorm
            -- iteration 2: gradient by outputOnes
            local gradOutput = self.normalize and flattenFirstDims(self, self.cdiv.gradInput[k]) or gradOutput
            assert(gradOutput:stride(2) == self.w) -- for C function safety

            for inPlaneIdx = 1,input:size(1) do
                for nWindow = 1,self.nWindows do
                    local outPlaneIdx = self.nWindows*(inPlaneIdx-1) + nWindow
                    
                    local xMinCurr, xMinCurrFrac = round_up  (self.xMin[nWindow]-1)
                    local xMaxCurr, xMaxCurrFrac = round_down(self.xMax[nWindow]+1)
                    local yMinCurr, yMinCurrFrac = round_up  (self.yMin[nWindow]-1)
                    local yMaxCurr, yMaxCurrFrac = round_down(self.yMax[nWindow]+1)

                    local gradOutData = torch.data(gradOutput[outPlaneIdx])
                    local intData
                    if k == 1 then
                        intData = torch.data(self.integralCuda[inPlaneIdx])
                    else
                        intData = torch.data(self.onesIntegral)
                    end

                    -- deltas of dOut(x,y) (sum over one window)
                    local deltas = torch.CudaTensor(4)
                    
                    if self.exact then
                        local inData, inStride
                        if k == 1 then
                            inData = torch.data(input[inPlaneIdx])
                            inStride = input:stride(2)
                        else -- k == 2
                            inData = torch.data(self.ones)
                            inStride = self.ones:stride(1)
                        end

                        CUDA_lib.backwardCudaSingleFrac(
                            intData, gradOutData,
                            torch.data(self.tmpArrayGPU),
                            torch.data(self.tmpArraySumGPU),
                            self.h, self.w, torch.data(deltas),
                            xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                            xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac,
                            inData, inStride)
                    else
                        CUDA_lib.backwardCudaSingle(
                            intData, gradOutData,
                            torch.data(self.tmpArrayGPU),
                            torch.data(self.tmpArraySumGPU),
                            self.h, self.w, torch.data(deltas),
                            xMinCurr, xMaxCurr, yMinCurr, yMaxCurr)
                    end

                    local xMinDelta, xMaxDelta = deltas[1], deltas[2]
                    local yMinDelta, yMaxDelta = deltas[3], deltas[4]
                    
                    self.gradXMax[nWindow] = self.gradXMax[nWindow] + scale * xMaxDelta
                    self.gradXMin[nWindow] = self.gradXMin[nWindow] + scale * xMinDelta
                    self.gradYMax[nWindow] = self.gradYMax[nWindow] + scale * yMaxDelta
                    self.gradYMin[nWindow] = self.gradYMin[nWindow] + scale * yMinDelta
                end
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