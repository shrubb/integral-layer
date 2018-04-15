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
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const int strideH, const int strideW);

void forwardNoNormReplicateFrac(
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const float xMinCurrFrac, const float xMaxCurrFrac,
    const float yMinCurrFrac, const float yMaxCurrFrac,
    const float *inData, const int inDataStride, const int strideH, const int strideW);

void updateGradInput(
    float *gradOutputInt, int nWindows, int h, int w, float *gradInput,
    int *xMin, int *xMax, int *yMin, int *yMax,
    float *gradOutput, int gradOutputStride, int strideH, int strideW);

void updateGradInputFrac(
    float *gradOutputInt, int channels, int h, int w, float *gradInput,
    int *xMin, int *xMax, int *yMin, int *yMax,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *gradOutput, int gradOutputStride, int strideH, int strideW);

void backwardNoNorm(
    float *intData, float *gradOutData, float scale, int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    const int strideH, const int strideW);

void backwardNoNormFrac(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *inData, int inStrideRow,
    const int strideH, const int strideW); ]]

local C_lib = ffi.load('C/lib/libintegral-c.so')

ffi.cdef [[
void forwardNoNormReplicateCuda(struct THCState *state,
    float *intData, int intDataStrideChannel, float *outData,
    int batchSize, int nInputPlane, int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW);

void forwardNoNormReplicateFracCuda(struct THCState *state,
    float *intData, int intDataStrideChannel, float *outData,
    int batchSize, int nInputPlane, int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow, int inDataStrideChannel,
    const int strideH, const int strideW);

void updateGradInputPlanewiseCuda(struct THCState *state,
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    int strideH, int strideW);

void updateGradInputPlanewiseFracCuda(struct THCState *state,
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *gradOutputData, int gradOutputStrideRow, int gradOutputStrideChannel,
    int strideH, int strideW);

void updateGradInputFracCuda(struct THCState *state,
    const float *gradOutputIntData, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax,
    const float *gradOutputData,
    const int gradOutputStrideRow, const int gradOutputStrideChannel,
    const int strideH, const int strideW);

void updateGradInputCuda(struct THCState *state,
    const float *gradOutputIntData, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax,
    const int strideH, const int strideW);

void backwardPlanewiseCuda(struct THCState *state,
    float *intData, float *tmpArray,
    int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    int strideH, int strideW);

void backwardPlanewiseFracCuda(struct THCState *state,
    float *intData, float *tmpArray,
    int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow,
    int strideH, int strideW);

void backwardFracCuda(struct THCState *state, const int paramId, const bool inputIsOnes,
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const float *inData, int inDataStrideRow, const int inDataStrideChannel,
    const int strideH, const int strideW);

void backwardCuda(struct THCState *state, const int paramId,
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const int strideH, const int strideW);

void integralImageCuda(struct THCState *state,
    float *input, float *output, int channels, int h, int w, float *tmp);

// void integralImageInplaceCuda(float *input, float *output, int channels, int h, int w);

void dirtyFixWindows(struct THCState *state,
    float *xMin, float *xMax, float *yMin, float *yMax,
    int size, int h, int w, float minWidth);
]]

local CUDA_lib

if pcall(require, 'cutorch') then
    CUDA_lib = ffi.load('C/lib/libintegral-cuda.so')
end

do
    cv = require 'cv'
    require 'cv.imgproc'
    require 'cv.highgui'

    require 'nn'

    -- if needed, enlarge `array` to fit a tensor of size `size`
    -- return its first bytes viewed as a tensor of size `size`
    local function reserveBuffer(array, ...)
        local size = {...}
        if torch.isTypeOf(size[1], 'torch.LongStorage') then
            size = size[1]
        end

        array = array:view(-1)

        local totalElements = 1
        for k = 1,#size do
            totalElements = totalElements * size[k]
        end

        if array:nElement() < totalElements then
            array:resize(totalElements)
        end

        return array[{{1, totalElements}}]:view(...)
    end

    -- to be defined below
    local updateOutputCPU, accGradParametersCPU
    local updateOutputGPU, accGradParametersGPU

    function IntegralSmartNorm:__init(nInputPlane, nWindows, h, w, strideH, strideW)
        strideH = strideH or 1
        strideW = strideW or 1

        -- nWindows is the number of box filters per channel
        parent.__init(self)
        self.nInputPlane, self.nWindows = nInputPlane, nWindows
        self.h, self.w, self.strideH, self.strideW = h, w, strideH, strideW
        self.reparametrization = self.w / 0.64 --583

        local function applyStride(k, stride)
            return math.ceil(k / stride)
        end
        self.hOut, self.wOut = applyStride(h, strideH), applyStride(w, strideW)
        
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
        self.outputOnes = torch.FloatTensor(nInputPlane*nWindows, self.hOut, self.wOut)
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
        -- if true, acts like `BORDER_MODE_REPLICATE` in OpenCV
        -- if false, treats those pixels as zeros (`BORDER_MODE_CONSTANT` with a value of 0)
        self.replicate = true
        -- if true, compute integral images for each batch sample one by one (a bit less memory usage but slower)
        -- if false, compute integral images in `:forward()` for all batch samples at once AND reuse them in `accGradParameters`
        self.saveMemoryIntegralInput = false
        -- if true, compute `batchSize*nInputPlane` integral images at `:updateGradInput()` one by one (slower)
        -- if false, compute all integral images at `:updateGradInput()` at once
        -- has no effect if `self.saveMemoryUpdateGradInput == true` (see below)
        self.saveMemoryIntegralGradOutput = false
        -- if true, accumulate gradInput straight into `self.gradInput`, but use kernel launches for that
        -- if false, use just two kernels, but increase extra memory usage to `self.output:nElement()` floats
        self.saveMemoryUpdateGradInput = false
        -- totally same as above but for `accGradParameters`. If the above is true, does not really use more memory
        self.saveMemoryAccGradParameters = false

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
            self.tmpArraySumGPU = torch.CudaTensor(4, self.nInputPlane)
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
            if self[param] then
                if self[param].storageOffset then
                    -- print('self.' .. param .. '.storageOffset = ' .. self[param]:storageOffset())
                end
                -- print('self.' .. param .. ' is ' .. self[param]:type() .. ', converting to ' .. type)
                self[param] = nn.utils.recursiveType(self[param], type, tensorCache)
                if self[param].storageOffset then
                    -- print('self.' .. param .. '.storageOffset = ' .. self[param]:storageOffset() .. '\n')
                end
            end
        end

        if self.backpropHelper then
            self.backpropHelper:type(type, tensorCache)
        end
        
        self._type = type
        return self
    end

    -- overload
    function IntegralSmartNorm:write(file)
        for _, param in ipairs{
            'nInputPlane', 'nWindows', 'h', 'w',
            'xMin', 'xMax', 'yMin', 'yMax',
            'gradXMin', 'gradXMax', 'gradYMin', 'gradYMax'} do

            file:writeObject(self[param])
        end
    end

    -- overload
    function IntegralSmartNorm:read(file)
        local nInputPlane = file:readObject()
        local nWindows    = file:readObject()
        local h           = file:readObject()
        local w           = file:readObject()
        
        self:__init(nInputPlane, nWindows, h, w)
        for _, param in ipairs{
            'xMin', 'xMax', 'yMin', 'yMax',
            'gradXMin', 'gradXMax', 'gradYMin', 'gradYMax'} do

            self[param] = file:readObject()
        end

        self:type(self.xMin:type())
    end

    -- reparametrization
    function IntegralSmartNorm:_reparametrize(mul)
        for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
            self[param]:mul(mul and self.reparametrization or (1 / self.reparametrization))
        end
    end

    function IntegralSmartNorm:reset()
        -- The only parameters of the module. Randomly initialize them
        
        -- put boxes in an overlapping uniform grid 
        -- of size ~ (ceil k/2) x (k) with the (largest possible k)-1;
        -- then place the remaining ones uniformly
        local k = 0
        while (k+1) * math.ceil((k+1)/2) <= self.nWindows do
            k = k + 1
        end

        -- add rows then cols as possible
        local gridSizeW, gridSizeH = k, math.ceil(k/2)
        -- while gridSizeW * (gridSizeH+1) <= self.nWindows do gridSizeH = gridSizeH + 1 end
        -- while (gridSizeW+1) * gridSizeH <= self.nWindows do gridSizeW = gridSizeW + 1 end

        local windowCounter = 1
        local winH, winW = (2*self.h-2) / gridSizeH, (2*self.w-2) / gridSizeW
        -- loop the overlapping window grid
        for x = 1,gridSizeH do
            local xCenter = -self.h+1 - winH*0.5 + winH*x
            for y = 1,gridSizeW do
                local yCenter = -self.w+1 - winW*0.5 + winW*y

                self.xMin[{{}, windowCounter}] = xCenter - winH*0.585
                self.xMax[{{}, windowCounter}] = xCenter + winH*0.585
                self.yMin[{{}, windowCounter}] = yCenter - winW*0.585
                self.yMax[{{}, windowCounter}] = yCenter + winW*0.585

                windowCounter = windowCounter + 1
            end
        end

        -- uniform init
        local minHeight, minWidth = winH, winW
        for inPlaneIdx = 1,self.nInputPlane do
            for windowIdx = windowCounter,self.nWindows do
                local centerX = torch.uniform(-self.h+1+minHeight/2, self.h-1-minHeight/2)
                local centerY = torch.uniform(-self.w+1+minWidth /2, self.w-1-minWidth /2)
                local height = 2 * torch.uniform(minHeight/2, 
                    math.min((self.h-1)-centerX, centerX-(-self.h+1)))
                local width  = 2 * torch.uniform(minWidth /2, 
                    math.min((self.w-1)-centerY, centerY-(-self.w+1)))

                self.xMin[{inPlaneIdx, windowIdx}] = (centerX - height/2)
                self.xMax[{inPlaneIdx, windowIdx}] = (centerX + height/2)
                self.yMin[{inPlaneIdx, windowIdx}] = (centerY - width /2)
                self.yMax[{inPlaneIdx, windowIdx}] = (centerY + width /2)
            end
        end

        self:_reparametrize(false)

        -- loss gradients wrt module's parameters
        self.gradXMin:zero()
        self.gradYMin:zero()
        self.gradXMax:zero()
        self.gradYMax:zero()
    end

    function IntegralSmartNorm:resetSingleWindow(idx)
        local minHeight, minWidth = 2,2--self.h / 12, self.w / 12
        local centerX = torch.uniform(-self.h*(2/4.8)+1+minHeight/2, self.h*(2/4.8)-1-minHeight/2)
        local centerY = torch.uniform(-self.w*(2/4.8)+1+minWidth /2, self.w*(2/4.8)-1-minWidth /2)
        local height = 2 * torch.uniform(minHeight/2, 
            math.min((self.h*(2/4.8)-1)-centerX, centerX-(-self.h*(2/4.8)+1)))
        local width  = 2 * torch.uniform(minWidth /2, 
            math.min((self.w*(2/4.8)-1)-centerY, centerY-(-self.w*(2/4.8)+1)))

        self.xMin:view(-1)[idx] = (centerX - height/2) / self.reparametrization
        self.xMax:view(-1)[idx] = (centerX + height/2) / self.reparametrization
        self.yMin:view(-1)[idx] = (centerY - width /2) / self.reparametrization
        self.yMax:view(-1)[idx] = (centerY + width /2) / self.reparametrization
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
        -- dirty fix 1: don't let windows go outside the image, otherwise gradients may vanish
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
                local minWidth = self.exact and 1 or 2

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
        self:_reparametrize(true)

        dirtyFixWindows(self)

        local hasChannelDim, hasBatchDim = true, true

        if input:nDimension() == 2 then
            input = nn.utils.addSingletonDimension(input)
            hasChannelDim = false
            hasBatchDim = false
        end

        -- force batch
        if input:nDimension() == 3 then
            input = nn.utils.addSingletonDimension(input)
            hasBatchDim = false
        end

        local batchSize = input:size(1)

        assert(input:size(2) == self.nInputPlane)
        assert(input:size(3) == self.h and input:size(4) == self.w)
        
        self.integralDouble:resize(self.nInputPlane, self.h+1, self.w+1)
        self.integral:resize(self.integralDouble:size())

        -- first, compute non-normalized box filter map (into self.outputOnes) of 1-s
        if self.normalize then
            self.outputOnes:resize(
                batchSize, self.nInputPlane*self.nWindows, self.hOut, self.wOut)
            assert(self.outputOnes:stride(3) == self.wOut) -- for C function safety

            local xMin, xMax = self.xMin:view(-1), self.xMax:view(-1)
            local yMin, yMax = self.yMin:view(-1), self.yMax:view(-1)

            for batchIdx = 1,batchSize do

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
                    local outData = torch.data(self.outputOnes[{batchIdx, globalWindowIdx}])
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
                            torch.data(self.ones), self.ones:stride(1),
                            self.strideH, self.strideW)
                    else
                        if self.replicate then
                            forwardCFunction = C_lib.forwardNoNormReplicate 
                        else
                            forwardCFunction = C_lib.forwardNoNorm
                        end

                        forwardCFunction(
                            intData, self.h, self.w, outData, 
                            xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                            self.strideH, self.strideW)
                    end
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
            self.outputNonNorm:resize(
                batchSize, self.nInputPlane, self.nWindows, self.hOut, self.wOut)
            assert(self.outputNonNorm:stride(4) == self.wOut) -- for C function safety

            for batchIdx = 1,batchSize do

                for inPlaneIdx = 1,self.nInputPlane do
                    cv.integral{input[{batchIdx, inPlaneIdx}], self.integralDouble[inPlaneIdx]}
                    self.integral[inPlaneIdx]:copy(self.integralDouble[inPlaneIdx]) -- cast

                    local xMin, xMax = self.xMin[inPlaneIdx], self.xMax[inPlaneIdx]
                    local yMin, yMax = self.yMin[inPlaneIdx], self.yMax[inPlaneIdx]
                    local outputNonNorm = self.outputNonNorm[{batchIdx, inPlaneIdx}]

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
                                error('NYI')
                                forwardCFunction = C_lib.forwardNoNormFrac
                            end

                            local inData = torch.data(input[{batchIdx, inPlaneIdx}])

                            forwardCFunction(
                                intData, self.h, self.w, outData, 
                                xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                                xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac, 
                                inData, input:stride(3),
                                self.strideH, self.strideW)
                        else
                            if self.replicate then
                                forwardCFunction = C_lib.forwardNoNormReplicate
                            else
                                error('NYI')
                                forwardCFunction = C_lib.forwardNoNorm
                            end

                            forwardCFunction(
                                intData, self.h, self.w, outData, 
                                xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                                self.strideH, self.strideW)
                        end
                    end
                end
            end

            self.outputNonNorm = 
                self.outputNonNorm:view(batchSize, self.nInputPlane*self.nWindows, self.hOut, self.wOut)
        end

        if self.normalize then
            -- divide elementwise to get normalized box filter maps
            self.output = self.cdiv:forward {self.outputNonNorm, self.outputOnes}
        else
            self.output = self.outputNonNorm
        end
        
        self._backwardDone = false

        if not hasBatchDim   and self.output:size(1) == 1 then self.output = self.output[1] end
        if not hasChannelDim and self.output:size(1) == 1 then self.output = self.output[1] end

        self:_reparametrize(false)

        return self.output
    end

    function updateOutputGPU(self, input)
        self:_reparametrize(true)

        CUDA_lib.dirtyFixWindows(cutorch.getState(),
            torch.data(self.xMin), torch.data(self.xMax),
            torch.data(self.yMin), torch.data(self.yMax),
            self.xMin:nElement(), self.h, self.w, self.exact and 1 or 2)

        local hasChannelDim, hasBatchDim = true, true

        if input:nDimension() == 2 then
            input = nn.utils.addSingletonDimension(input)
            hasChannelDim = false
            hasBatchDim = false
        end

        -- force batch
        if input:nDimension() == 3 then
           input = nn.utils.addSingletonDimension(input)
           hasBatchDim = false
        end

        local batchSize = input:size(1)

        assert(input:size(2) == self.nInputPlane)
        assert(input:size(3) == self.h and input:size(4) == self.w)
        assert(self.xMin:stride(1) == self.xMin:size(2))

        -- first, compute non-normalized box filter map (into self.outputOnes) of 1-s        
        if self.normalize then
            self.outputOnes:resize(batchSize, self.nInputPlane*self.nWindows, self.hOut, self.wOut)
            assert(self.outputOnes:stride(3) == self.wOut) -- for C function safety

            local forwardCFunction

            if self.exact then
                if self.replicate then
                    forwardCFunction = CUDA_lib.forwardNoNormReplicateFracCuda
                else
                    error('NYI')
                    forwardCFunction = C_lib.forwardNoNormFrac
                end

                forwardCFunction(cutorch.getState(),
                    self.onesIntegral:data(), 0, self.outputOnes:data(),
                    batchSize, self.nInputPlane, self.nWindows, self.h, self.w,
                    torch.data(self.xMin), torch.data(self.xMax),
                    torch.data(self.yMin), torch.data(self.yMax),
                    torch.data(self.ones), self.ones:stride(1), 0,
                    self.strideH, self.strideW)
            else
                if self.replicate then
                    forwardCFunction = CUDA_lib.forwardNoNormReplicateCuda
                else
                    error('NYI')
                    forwardCFunction = C_lib.forwardNoNorm
                end

                forwardCFunction(cutorch.getState(),
                    self.onesIntegral:data(), 0, self.outputOnes:data(),
                    batchSize, self.nInputPlane, self.nWindows, self.h, self.w,
                    torch.data(self.xMin), torch.data(self.xMax),
                    torch.data(self.yMin), torch.data(self.yMax),
                    self.strideH, self.strideW)
            end
        end

        -- next, compute non-normalized box filter map (into self.outputNonNorm) from input
        do
            self.outputNonNorm:resize(batchSize, self.nInputPlane*self.nWindows, self.hOut, self.wOut)
            assert(self.outputNonNorm:stride(3) == self.wOut) -- for C function safety
            assert(self.outputNonNorm:stride(2) == self.wOut*self.hOut) -- for C function safety

            if self.saveMemoryIntegralInput then
                -- compute integral images for each batch sample separately
                self.integralCuda:resize(self.nInputPlane, self.h+1, self.w+1)
            else
                -- compute integral images for all batch samples at once
                -- plus reuse them in `accGradParameters`
                self.integralCuda:resize(batchSize, self.nInputPlane, self.h+1, self.w+1)
            end
            assert(self.integralCuda:isContiguous()) -- TODO: use actual strides

            reserveBuffer(self.tmpArrayGPU, self.integralCuda:size())

            if self.saveMemoryIntegralInput then
                -- slow way, process samples 1 by 1
                for batchIdx = 1,batchSize do

                    CUDA_lib.integralImageCuda(cutorch.getState(),
                        input[batchIdx]:data(), self.integralCuda:data(),
                        self.nInputPlane, self.h, self.w,
                        self.tmpArrayGPU:data())
                
                    local forwardCudaFunction

                    if self.exact then
                        if self.replicate then
                            forwardCudaFunction = CUDA_lib.forwardNoNormReplicateFracCuda
                        else
                            error('NYI')
                            forwardCudaFunction = CUDA_lib.forwardNoNormFracCuda
                        end

                        forwardCudaFunction(cutorch.getState(),
                            self.integralCuda:data(), self.integralCuda:stride(1), 
                            self.outputNonNorm[batchIdx]:data(),
                            1, self.nInputPlane, self.nWindows, self.h, self.w,
                            torch.data(self.xMin), torch.data(self.xMax),
                            torch.data(self.yMin), torch.data(self.yMax),
                            torch.data(input[batchIdx]), input:stride(3), input:stride(2),
                            self.strideH, self.strideW)
                    else
                        if self.replicate then
                            forwardCudaFunction = CUDA_lib.forwardNoNormReplicateCuda
                        else
                            error('NYI')
                            forwardCudaFunction = CUDA_lib.forwardNoNormCuda
                        end

                        forwardCudaFunction(cutorch.getState(),
                            self.integralCuda:data(), self.integralCuda:stride(1), 
                            self.outputNonNorm[batchIdx]:data(),
                            1, self.nInputPlane, self.nWindows, self.h, self.w,
                            torch.data(self.xMin), torch.data(self.xMax),
                            torch.data(self.yMin), torch.data(self.yMax),
                            self.strideH, self.strideW)
                    end
                end -- for batchIdx
            else -- NOT self.saveMemoryIntegralInput
                -- faster option, process whole batch at once
                CUDA_lib.integralImageCuda(cutorch.getState(),
                    input:data(), self.integralCuda:data(),
                    batchSize*self.nInputPlane, self.h, self.w,
                    self.tmpArrayGPU:data())
            
                local forwardCudaFunction

                if self.exact then
                    if self.replicate then
                        forwardCudaFunction = CUDA_lib.forwardNoNormReplicateFracCuda
                    else
                        error('NYI')
                        forwardCudaFunction = CUDA_lib.forwardNoNormFracCuda
                    end

                    forwardCudaFunction(cutorch.getState(),
                        self.integralCuda:data(), self.integralCuda:stride(2), self.outputNonNorm:data(),
                        batchSize, self.nInputPlane, self.nWindows, self.h, self.w,
                        torch.data(self.xMin), torch.data(self.xMax),
                        torch.data(self.yMin), torch.data(self.yMax),
                        torch.data(input), input:stride(3), input:stride(2),
                        self.strideH, self.strideW)
                else
                    if self.replicate then
                        forwardCudaFunction = CUDA_lib.forwardNoNormReplicateCuda
                    else
                        error('NYI')
                        forwardCudaFunction = CUDA_lib.forwardNoNormCuda
                    end

                    forwardCudaFunction(cutorch.getState(),
                        self.integralCuda:data(), self.integralCuda:stride(2), self.outputNonNorm:data(),
                        batchSize, self.nInputPlane, self.nWindows, self.h, self.w,
                        torch.data(self.xMin), torch.data(self.xMax),
                        torch.data(self.yMin), torch.data(self.yMax),
                        self.strideH, self.strideW)
                end
            end
        end -- do

        if self.normalize then
            -- divide elementwise to get normalized box filter maps
            self.output = self.cdiv:forward {self.outputNonNorm, self.outputOnes}
        else
            self.output = self.outputNonNorm
        end

        self._backwardDone = false

        if not hasBatchDim   and self.output:size(1) == 1 then self.output = self.output[1] end
        if not hasChannelDim and self.output:size(1) == 1 then self.output = self.output[1] end

        self:_reparametrize(false)

        return self.output
    end

    function IntegralSmartNorm:updateGradInput(input, gradOutput)
        if self.gradInput then

            self:_reparametrize(true)

            if self.normalize then
                if not self._backwardDone then
                    self.cdiv:updateGradInput({self.outputNonNorm, self.outputOnes}, gradOutput)
                    self._backwardDone = true
                end
                gradOutput = self.cdiv.gradInput[1]
            end
            
            local hasChannelDim, hasBatchDim = true, true

            if input:nDimension() == 2 then
                input = nn.utils.addSingletonDimension(input)
                hasChannelDim = false
                hasBatchDim = false
            end

            -- force batch
            if input:nDimension() == 3 then
               input = nn.utils.addSingletonDimension(input)
               hasBatchDim = false
            end

            local batchSize = input:size(1)

            self.gradInput:resize(input:size())
            gradOutput = 
                gradOutput:view(batchSize, self.nInputPlane, self.nWindows, self.hOut, self.wOut)

            if self._type == 'torch.CudaTensor' then -- GPU

                if self.saveMemoryUpdateGradInput then
                    if self.saveMemoryIntegralGradOutput then
                        self.integralGradOutput:resize(self.nWindows, self.hOut+1, self.wOut+1)
                    else
                        self.integralGradOutput:resize(batchSize, self.nInputPlane, self.nWindows, self.hOut+1, self.wOut+1)
                    end

                    reserveBuffer(self.tmpArrayGPU, self.integralGradOutput:size())

                    if not self.saveMemoryIntegralGradOutput then
                        CUDA_lib.integralImageCuda(cutorch.getState(),
                            torch.data(gradOutput),
                            torch.data(self.integralGradOutput),
                            batchSize*self.nInputPlane*self.nWindows, self.hOut, self.wOut,
                            torch.data(self.tmpArrayGPU))
                    end

                    for batchIdx = 1,batchSize do
                        for inPlaneIdx = 1,self.nInputPlane do

                            if self.saveMemoryIntegralGradOutput then
                                CUDA_lib.integralImageCuda(cutorch.getState(),
                                    torch.data(gradOutput[{batchIdx, inPlaneIdx}]),
                                    torch.data(self.integralGradOutput),
                                    self.nWindows, self.hOut, self.wOut,
                                    torch.data(self.tmpArrayGPU))
                            end

                            -- compute the needed integral sums
                            local updateGradInputCFunction

                            if self.exact then
                                if self.replicate then
                                    updateGradInputCFunction = CUDA_lib.updateGradInputPlanewiseFracCuda
                                else
                                    error('NYI')
                                end

                                updateGradInputCFunction(cutorch.getState(),
                                    torch.data(self.integralGradOutput[self.saveMemoryIntegralGradOutput and 1 or {batchIdx, inPlaneIdx}]),
                                    torch.data(self.gradInput[{batchIdx, inPlaneIdx}]),
                                    self.h, self.w, self.nWindows,
                                    torch.data(self.xMin[inPlaneIdx]),
                                    torch.data(self.xMax[inPlaneIdx]),
                                    torch.data(self.yMin[inPlaneIdx]),
                                    torch.data(self.yMax[inPlaneIdx]),
                                    torch.data(gradOutput[{batchIdx, inPlaneIdx}]),
                                    gradOutput:stride(4), gradOutput:stride(3),
                                    self.strideH, self.strideW)
                            else
                                if self.replicate then
                                    updateGradInputCFunction = CUDA_lib.updateGradInputPlanewiseCuda
                                else
                                    error('NYI')
                                end

                                updateGradInputCFunction(cutorch.getState(),
                                    torch.data(self.integralGradOutput[self.saveMemoryIntegralGradOutput and 1 or {batchIdx, inPlaneIdx}]),
                                    torch.data(self.gradInput[{batchIdx, inPlaneIdx}]),
                                    self.h, self.w, self.nWindows,
                                    torch.data(self.xMin[inPlaneIdx]),
                                    torch.data(self.xMax[inPlaneIdx]),
                                    torch.data(self.yMin[inPlaneIdx]),
                                    torch.data(self.yMax[inPlaneIdx]),
                                    self.strideH, self.strideW)
                            end
                        end
                    end
                else -- if NOT self.saveMemoryUpdateGradInput:

                    self.integralGradOutput:resize(batchSize, self.nInputPlane, self.nWindows, self.hOut+1, self.wOut+1)
                    
                    reserveBuffer(self.tmpArrayGPU, self.integralGradOutput:size())
                    
                    CUDA_lib.integralImageCuda(cutorch.getState(),
                        torch.data(gradOutput),
                        torch.data(self.integralGradOutput),
                        batchSize*self.nInputPlane*self.nWindows, self.hOut, self.wOut,
                        torch.data(self.tmpArrayGPU))

                    local tmpArrayGPUBegin = reserveBuffer(self.tmpArrayGPU, batchSize, self.nInputPlane, self.nWindows, self.h, self.w)
                    assert(self.xMin:stride(1) == self.xMin:size(2))

                    if self.exact then
                        if self.replicate then
                            updateGradInputCFunction = CUDA_lib.updateGradInputFracCuda
                        else
                            error('NYI')
                        end

                        updateGradInputCFunction(cutorch.getState(),
                            torch.data(self.integralGradOutput),
                            torch.data(self.tmpArrayGPU),
                            batchSize, self.nInputPlane, self.nWindows, self.h, self.w,
                            torch.data(self.xMin), torch.data(self.xMax),
                            torch.data(self.yMin), torch.data(self.yMax),
                            torch.data(gradOutput),
                            gradOutput:stride(4), gradOutput:stride(3),
                            self.strideH, self.strideW)
                    else
                        if self.replicate then
                            updateGradInputCFunction = CUDA_lib.updateGradInputCuda
                        else
                            error('NYI')
                        end

                        updateGradInputCFunction(cutorch.getState(),
                            torch.data(self.integralGradOutput),
                            torch.data(self.tmpArrayGPU),
                            batchSize, self.nInputPlane, self.nWindows, self.h, self.w,
                            torch.data(self.xMin), torch.data(self.xMax),
                            torch.data(self.yMin), torch.data(self.yMax),
                            self.strideH, self.strideW)
                    end

                    torch.sum(self.gradInput, tmpArrayGPUBegin, 3)
                    self.gradInput = self.gradInput[{{}, {}, 1, {}, {}}]
                end
            else -- CPU

                self.gradInput:zero()
                self.integralGradOutput:resize(self.nWindows, self.hOut+1, self.wOut+1)
                self.integralDouble:resize(1, self.hOut+1, self.wOut+1) -- temporary buffer for OpenCV

                for batchIdx = 1,batchSize do
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

                        for windowIdx = 1,self.nWindows do
                            cv.integral{
                                gradOutput[{batchIdx, inPlaneIdx, windowIdx}],
                                self.integralDouble[1]}
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
                                torch.data(self.integralGradOutput),
                                self.nWindows, self.h, self.w,
                                torch.data(self.gradInput[{batchIdx, inPlaneIdx}]),
                                torch.data(xMinInt), torch.data(xMaxInt),
                                torch.data(yMinInt), torch.data(yMaxInt),
                                torch.data(xMinFrac), torch.data(xMaxFrac),
                                torch.data(yMinFrac), torch.data(yMaxFrac),
                                torch.data(gradOutput[{batchIdx, inPlaneIdx}]),
                                gradOutput:stride(4), self.strideH, self.strideW)
                        else
                            error('NYI')
                            if self.replicate then
                                updateGradInputCFunction = C_lib.updateGradInput
                            else
                                error('NYI')
                            end

                            updateGradInputCFunction(
                                torch.data(self.integralGradOutput),
                                self.nWindows, self.h, self.w,
                                torch.data(self.gradInput[{batchIdx, inPlaneIdx}]),
                                torch.data(xMinInt), torch.data(xMaxInt),
                                torch.data(yMinInt), torch.data(yMaxInt),
                                torch.data(gradOutput[{batchIdx, inPlaneIdx}]),
                                gradOutput:stride(4), self.strideH, self.strideW)
                        end
                    end
                end
            end

            if not hasBatchDim   then self.gradInput = self.gradInput[1] end
            if not hasChannelDim then self.gradInput = self.gradInput[1] end

            self:_reparametrize(false)

            return self.gradInput
        end
    end

    function accGradParametersCPU(self, input, gradOutput, scale)
        scale = scale or 1

        self:_reparametrize(true)

        if self.normalize then
            if not self._backwardDone then
                self.cdiv:updateGradInput({self.outputNonNorm, self.outputOnes}, gradOutput)
                self._backwardDone = true
            end
            gradOutput = self.cdiv.gradInput[1]
        end

        local hasChannelDim, hasBatchDim = true, true

        if input:nDimension() == 2 then
            input = nn.utils.addSingletonDimension(input)
            hasChannelDim = false
            hasBatchDim = false
        end

        -- force batch
        if input:nDimension() == 3 then
           input = nn.utils.addSingletonDimension(input)
           hasBatchDim = false
        end

        local batchSize = input:size(1)

        for k = 1,(self.normalize and 2 or 1) do
            -- iteration 1: gradient by outputNonNorm
            -- iteration 2: gradient by outputOnes
            local gradOutput = self.normalize and self.cdiv.gradInput[k] or gradOutput
            gradOutput = 
                gradOutput:view(batchSize, self.nInputPlane, self.nWindows, self.hOut, self.wOut)

            assert(gradOutput:stride(gradOutput:nDimension()-1) == self.wOut) -- for C function

            for batchIdx = 1,batchSize do
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

                    local gradOutData = torch.data(gradOutput[{batchIdx, inPlaneIdx}])
                    local intData
                    if k == 1 then
                        intData = torch.data(self.integral[inPlaneIdx])
                    else
                        intData = torch.data(self.onesIntegral)
                    end
                    
                    if self.exact then
                        local inData, inStrideRow
                        if k == 1 then
                            inData = torch.data(input[{batchIdx, inPlaneIdx}])
                            inStrideRow = input:stride(input:nDimension()-1)
                        else -- k == 2
                            inData = torch.data(self.ones)
                            inStrideRow = self.ones:stride(1)
                        end

                        C_lib.backwardNoNormFrac(
                            intData, gradOutData, scale * self.reparametrization,
                            self.nWindows, self.h, self.w,
                            torch.data(gradXMin), torch.data(gradXMax),
                            torch.data(gradYMin), torch.data(gradYMax),
                            torch.data(xMinInt), torch.data(xMaxInt),
                            torch.data(yMinInt), torch.data(yMaxInt),
                            torch.data(xMinFrac), torch.data(xMaxFrac),
                            torch.data(yMinFrac), torch.data(yMaxFrac),
                            inData, inStrideRow, self.strideH, self.strideW)
                    else
                        C_lib.backwardNoNorm(
                            intData, gradOutData, scale * self.reparametrization,
                            self.nWindows, self.h, self.w,
                            torch.data(gradXMin), torch.data(gradXMax),
                            torch.data(gradYMin), torch.data(gradYMax),
                            torch.data(xMinInt), torch.data(xMaxInt),
                            torch.data(yMinInt), torch.data(yMaxInt),
                            self.strideH, self.strideW)
                    end
                end
            end
        end

        self:_reparametrize(false)
    end

    function accGradParametersGPU(self, input, gradOutput, scale)
        scale = scale or 1

        self:_reparametrize(true)
        
        if self.normalize then
            if not self._backwardDone then
                self.cdiv:updateGradInput({self.outputNonNorm, self.outputOnes}, gradOutput)
                self._backwardDone = true
            end
            gradOutput = self.cdiv.gradInput[1]
        end

        local hasChannelDim, hasBatchDim = true, true

        if input:nDimension() == 2 then
            input = nn.utils.addSingletonDimension(input)
            hasChannelDim = false
            hasBatchDim = false
        end

        -- force batch
        if input:nDimension() == 3 then
           input = nn.utils.addSingletonDimension(input)
           hasBatchDim = false
        end

        local batchSize = input:size(1)

        if self.saveMemoryIntegralInput then
            assert(self.integralCuda:stride(2) == self.w+1)
        else
            assert(self.integralCuda:size(1) == batchSize)
            assert(self.integralCuda:stride(3) == self.w+1)
        end

        for k = 1,(self.normalize and 2 or 1) do
            -- iteration 1: gradient by outputNonNorm
            -- iteration 2: gradient by outputOnes
            local gradOutput = self.normalize and self.cdiv.gradInput[k] or gradOutput
            gradOutput = 
                gradOutput:view(batchSize, self.nInputPlane, self.nWindows, self.hOut * self.wOut)

            if self.saveMemoryAccGradParameters then
                -- significantly slower (many kernel launches) but sometimes lower memory usage
                -- not recommended to use
                local tmpArraySumGPUBegin = reserveBuffer(self.tmpArraySumGPU, 4, self.nWindows)

                for batchIdx = 1,batchSize do
                    reserveBuffer(self.tmpArrayGPU, self.nInputPlane, self.h+1, self.w+1)

                    if self.saveMemoryIntegralInput and k == 1 then
                        CUDA_lib.integralImageCuda(cutorch.getState(),
                            input[batchIdx]:data(), self.integralCuda:data(),
                            self.nInputPlane, self.h, self.w,
                            self.tmpArrayGPU:data())
                    end

                    local tmpArrayGPUBegin =reserveBuffer(self.tmpArrayGPU, 4, self.nWindows, self.hOut*self.wOut)

                    local accGradParametersCFunction

                    if self.exact then
                        if self.replicate then
                            accGradParametersCFunction = CUDA_lib.backwardPlanewiseFracCuda
                        else
                            error('NYI')
                        end

                        for inPlaneIdx = 1,self.nInputPlane do
                            local intData = k == 1 and 
                                self.integralCuda[self.saveMemoryIntegralInput and inPlaneIdx or {batchIdx,inPlaneIdx}]:data() or
                                self.onesIntegral:data()

                            local inData, inStrideRow, inStrideChannel
                            if k == 1 then
                                inData = torch.data(input[{batchIdx, inPlaneIdx}])
                                inStrideRow = input:stride(input:nDimension()-1)
                                -- not using it to save memory
                                inStrideChannel = input:stride(input:nDimension()-2)
                            else -- k == 2
                                -- TODO write a separate faster kernel for k == 1
                                inData = torch.data(self.ones)
                                inStrideRow = self.ones:stride(1)
                                inStrideChannel = 0
                            end

                            tmpArrayGPUBegin:copy(
                                gradOutput[{batchIdx, {inPlaneIdx, inPlaneIdx}}]
                                    :expand(4, self.nWindows, self.hOut * self.wOut))

                            -- multiplies `self.tmpArrayGPU` by parameter deltas
                            accGradParametersCFunction(cutorch.getState(),
                                intData, torch.data(self.tmpArrayGPU),
                                self.nWindows, self.h, self.w,
                                torch.data(self.xMin[inPlaneIdx]), torch.data(self.xMax[inPlaneIdx]),
                                torch.data(self.yMin[inPlaneIdx]), torch.data(self.yMax[inPlaneIdx]),
                                inData, inStrideRow, self.strideH, self.strideW) --, inStrideChannel)

                            torch.sum(tmpArraySumGPUBegin, tmpArrayGPUBegin, 3)

                            torch.add(self.gradXMax[inPlaneIdx], self.gradXMax[inPlaneIdx], scale * self.reparametrization, tmpArraySumGPUBegin[1])
                            torch.add(self.gradXMin[inPlaneIdx], self.gradXMin[inPlaneIdx], scale * self.reparametrization, tmpArraySumGPUBegin[2])
                            torch.add(self.gradYMax[inPlaneIdx], self.gradYMax[inPlaneIdx], scale * self.reparametrization, tmpArraySumGPUBegin[3])
                            torch.add(self.gradYMin[inPlaneIdx], self.gradYMin[inPlaneIdx], scale * self.reparametrization, tmpArraySumGPUBegin[4])
                        end
                    else
                        if self.replicate then
                            accGradParametersCFunction = CUDA_lib.backwardPlanewiseCuda
                        else
                            error('NYI')
                        end

                        for inPlaneIdx = 1,self.nInputPlane do
                            local intData = k == 1 and 
                                self.integralCuda[self.saveMemoryIntegralInput and inPlaneIdx or {batchIdx,inPlaneIdx}]:data() or
                                self.onesIntegral:data()

                            tmpArrayGPUBegin:copy(
                                gradOutput[{batchIdx, {inPlaneIdx, inPlaneIdx}}]
                                    :expand(4, self.nWindows, self.hOut * self.wOut))

                            -- multiplies `self.tmpArrayGPU` by parameter deltas
                            accGradParametersCFunction(cutorch.getState(),
                                intData, torch.data(self.tmpArrayGPU),
                                self.nWindows, self.h, self.w,
                                torch.data(self.xMin[inPlaneIdx]), torch.data(self.xMax[inPlaneIdx]),
                                torch.data(self.yMin[inPlaneIdx]), torch.data(self.yMax[inPlaneIdx]),
                                self.strideH, self.strideW)

                            torch.sum(tmpArraySumGPUBegin, tmpArrayGPUBegin, 3)

                            torch.add(self.gradXMax[inPlaneIdx], self.gradXMax[inPlaneIdx], scale * self.reparametrization, tmpArraySumGPUBegin[1])
                            torch.add(self.gradXMin[inPlaneIdx], self.gradXMin[inPlaneIdx], scale * self.reparametrization, tmpArraySumGPUBegin[2])
                            torch.add(self.gradYMax[inPlaneIdx], self.gradYMax[inPlaneIdx], scale * self.reparametrization, tmpArraySumGPUBegin[3])
                            torch.add(self.gradYMin[inPlaneIdx], self.gradYMin[inPlaneIdx], scale * self.reparametrization, tmpArraySumGPUBegin[4])
                        end
                    end
                end
            else -- if NOT self.saveMemoryAccGradParameters
                -- faster implementation
                local tmpArrayGPUBegin = reserveBuffer(self.tmpArrayGPU, batchSize, self.nInputPlane, self.nWindows, self.h*self.w)
                assert(tmpArrayGPUBegin:isContiguous())
                -- first `batchSize` channels will hold intermediate sums, then we will sum THEM in the last channel
                local tmpArraySumGPUBegin = 
                    reserveBuffer(self.tmpArraySumGPU, batchSize+1, self.nInputPlane, self.nWindows)

                local accGradParametersCFunction

                if self.exact then
                    if self.replicate then
                        accGradParametersCFunction = CUDA_lib.backwardFracCuda
                    else
                        error('NYI')
                    end
                else
                    if self.replicate then
                        accGradParametersCFunction = CUDA_lib.backwardCuda
                    else
                        error('NYI')
                    end
                end

                local intData, intDataStrideChannel
                local inData, inStrideRow, inStrideChannel
                if k == 1 then
                    intData = self.integralCuda:data()
                    intDataStrideChannel = self.integralCuda:stride(2)

                    inData = input:data()
                    inStrideRow = input:stride(input:nDimension()-1)
                    -- not using it to save memory
                    inStrideChannel = input:stride(input:nDimension()-2)
                else -- k == 2
                    intData = self.onesIntegral:data()
                    intDataStrideChannel = 0

                    -- TODO write a separate faster kernel for k == 1
                    inData = torch.data(self.ones)
                    inStrideRow = self.ones:stride(1)
                    inStrideChannel = 0
                end

                local tmpArraySpatialSum = tmpArraySumGPUBegin[{{1, batchSize}}]
                local tmpArrayFinalSum = tmpArraySumGPUBegin[{{batchSize+1, batchSize+1}}]

                local gradParams = {self.gradXMin, self.gradXMax, self.gradYMin, self.gradYMax}

                for paramId, gradParam in ipairs(gradParams) do
                    tmpArrayGPUBegin:copy(gradOutput)

                    -- multiply `self.tmpArrayGPU` by parameter deltas
                    if self.exact then
                        accGradParametersCFunction(cutorch.getState(), paramId-1, k == 2,
                            intData, intDataStrideChannel, self.tmpArrayGPU:data(),
                            batchSize, self.nInputPlane, self.nWindows, self.h, self.w,
                            self.xMin:data(), self.xMax:data(),
                            self.yMin:data(), self.yMax:data(),
                            inData, inStrideRow, inStrideChannel, self.strideH, self.strideW)
                    else
                        accGradParametersCFunction(cutorch.getState(), paramId-1,
                            intData, intDataStrideChannel, self.tmpArrayGPU:data(),
                            batchSize, self.nInputPlane, self.nWindows, self.h, self.w,
                            self.xMin:data(), self.xMax:data(),
                            self.yMin:data(), self.yMax:data(),
                            self.strideH, self.strideW)
                    end

                    -- `tmpArrayGPUBegin` is "batchSize x nInputPlane x nWindows x (h * w)"
                    torch.sum(tmpArraySpatialSum, tmpArrayGPUBegin, 4) -- sum out both spatial dimensions
                    torch.sum(tmpArrayFinalSum, tmpArraySpatialSum, 1) -- sum out the batch dimension

                    torch.add(gradParam, gradParam, scale * self.reparametrization, tmpArrayFinalSum)
                end
            end
        end -- for k = 1,(self.normalize and 2 or 1) do

        self:_reparametrize(false)
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
