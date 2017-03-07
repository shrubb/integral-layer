require 'nn'

ffi = require 'ffi'

local _, parent = torch.class('Integral', 'nn.Module')

ffi.cdef [[

void forward(
    void *intData, int h, int w, void *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float areaCoeff);

void backward();

]]

local C = ffi.load('C/lib/libintegral-c.so')

do
    cv = require 'cv'
    require 'cv.imgproc'
    require 'cv.highgui'

    function Integral:__init(nWindows, h, w)
        parent.__init(self)
        self.nWindows, self.h, self.w = nWindows, h, w
        self.output = torch.Tensor(self.nWindows, h, w)
        self.integralDouble = torch.DoubleTensor(h+1, w+1)
        self.integral = torch.FloatTensor(h+1, w+1)
        self:reset()
        self:zeroGradParameters()
    end

    -- renew normalization coeffs
    function Integral:recalculateArea()
        for i = 1,self.nWindows do
            self.areaCoeff[i] = 
                1 / ((self.xMax[i]-self.xMin[i]+1)*(self.yMax[i]-self.yMin[i]+1))
        end
    end

    function Integral:reset()
        -- the only parameters of the module
        self.xMin = torch.round((torch.rand(self.nWindows) - 0.5) * (2 * self.h * 0.3))
        self.yMin = torch.round((torch.rand(self.nWindows) - 0.5) * (2 * self.w * 0.3))
        self.xMax = torch.Tensor(self.nWindows)
        self.yMax = torch.Tensor(self.nWindows)
        
        for i = 1,self.nWindows do
            self.xMax[i] = torch.round(torch.uniform(self.xMin[i] + self.h * 0.05, self.xMin[i] + self.h * 0.25))
            self.yMax[i] = torch.round(torch.uniform(self.yMin[i] + self.w * 0.05, self.yMin[i] + self.w * 0.25))
        end
        
        --[[do
            -- strict initialization for debugging
            self.xMin[1] = 0
            self.xMax[1] = 0
            self.yMin[1] = 0
            self.yMax[1] = 0

            for i = 2,self.nWindows do
                self.xMin[i] = self.xMin[i-1] - 3
                self.xMax[i] = self.xMax[i-1] + 3
                self.yMin[i] = self.yMin[i-1] - 3
                self.yMax[i] = self.yMax[i-1] + 3
            end
        end]]
        
        -- area to normalize over
        self.areaCoeff = torch.Tensor(self.nWindows)
        self:recalculateArea()
        
        -- loss gradients wrt module's parameters
        self.gradXMin = torch.zeros(self.xMin:size())
        self.gradYMin = torch.zeros(self.xMin:size())
        self.gradXMax = torch.zeros(self.xMin:size())
        self.gradYMax = torch.zeros(self.xMin:size())
    end

    function Integral:parameters()
        local params = {self.xMin, self.xMax, self.yMin, self.yMax}
        local gradParams = {self.gradXMin, self.gradXMax, self.gradYMin, self.gradYMax}
        return params, gradParams
    end

    local function round_towards_zero(x)
        if x >= 0 then return math.floor(x) 
        else return math.floor(x) end
    end

    function Integral:updateOutput(input)
        if input:nDimension() ~= 2 then
            error('wrong input:nDimension()')
        end
        
    --     self.output:fill(input:max())
        
        assert(input:size(1) == self.h and input:size(2) == self.w)
        
    --     local xMaxInt, xMinInt = math.floor(self.xMax), math.ceil(self.xMin)
    --     local xMaxDiff, xMinDiff = self.xMax-xMaxInt, self.xMin-xMinInt
    --     local yMaxInt, yMinInt = math.floor(self.yMax), math.ceil(self.yMin)
    --     local yMaxDiff, yMinDiff = self.yMax-yMaxInt, self.yMin-yMinInt
        
        cv.integral{input, self.integralDouble}
        self.integral:copy(self.integralDouble) -- cast
        
        for planeIdx = 1,self.nWindows do
            
            -- round towards zero (?)
            local xMinCurr = round_towards_zero(self.xMin[planeIdx])
            local xMaxCurr = round_towards_zero(self.xMax[planeIdx])+1
            local yMinCurr = round_towards_zero(self.yMin[planeIdx])
            local yMaxCurr = round_towards_zero(self.yMax[planeIdx])+1
            
            -- round down (?)
    --         local xMinCurr = torch.round(self.xMin[planeIdx] - 0.499)
    --         local xMaxCurr = torch.round(self.xMax[planeIdx] - 0.499)+1
    --         local yMinCurr = torch.round(self.yMin[planeIdx] - 0.499)
    --         local yMaxCurr = torch.round(self.yMax[planeIdx] - 0.499)+1
            
            local outPlane = self.output[planeIdx]
            
            local outData = torch.data(outPlane)
            local intData = torch.data(self.integral)
            
            -- must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
            -- `integral()` behavior. Namely, I(x,0) and I(0,y) are
            -- always 0 (so it's a C-style array sum).
            
            C.forward(
                intData, self.h, self.w, outData, 
                xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
                self.areaCoeff[planeIdx])
            
            -- outPlane:mul(self.areaCoeff[planeIdx])
        end
        
        return self.output
    end


    function Integral:updateGradInput(input, gradOutput)
        -- never call :backward() on backpropHelper!
        self.backpropHelper = self.backpropHelper or Integral(1, self.h, self.w)
        
        if self.gradInput then
            self.gradInput:resize(self.h, self.w):zero()
            
            for nWindow = 1,self.nWindows do
                self.backpropHelper.xMin[1] = -self.xMax[nWindow]
                self.backpropHelper.xMax[1] = -self.xMin[nWindow]
                self.backpropHelper.yMin[1] = -self.yMax[nWindow]
                self.backpropHelper.yMax[1] = -self.yMin[nWindow]
                self.backpropHelper:recalculateArea()
                
                self.gradInput:add(self.backpropHelper:forward(gradOutput[nWindow]):squeeze())
            end
            
            return self.gradInput
        end
    end

    function Integral:accGradParameters(input, gradOutput, scale)
        scale = scale or 1
        
        for planeIdx = 1,self.nWindows do
            local outputDot = torch.dot(self.output[planeIdx], gradOutput[planeIdx])
            
            -- round towards zero (?)
            -- and +1 because OpenCV's integral adds extra row and col
            local xMinCurr = round_towards_zero(self.xMin[planeIdx])+1
            local xMaxCurr = round_towards_zero(self.xMax[planeIdx])+1
            local yMinCurr = round_towards_zero(self.yMin[planeIdx])+1
            local yMaxCurr = round_towards_zero(self.yMax[planeIdx])+1
            
            -- round down (?)
    --         local xMinCurr = torch.round(self.xMin[planeIdx] - 0.499)+1
    --         local xMaxCurr = torch.round(self.xMax[planeIdx] - 0.499)+1
    --         local yMinCurr = torch.round(self.yMin[planeIdx] - 0.499)+1
    --         local yMaxCurr = torch.round(self.yMax[planeIdx] - 0.499)+1
            
            local gradOutData = torch.data(gradOutput[planeIdx])
            local intData = torch.data(self.integral)
            
            -- deltas of dOut(x,y) (sum over one window)
            local xMaxDelta, xMinDelta = 0, 0
            local yMaxDelta, yMinDelta = 0, 0
            
            for x = 1,self.h do
                for y = 1,self.w do
                    xMaxDelta = xMaxDelta
                        +(intData[math.max(0,math.min(x+xMaxCurr+1,self.h))*(self.w+1) 
                            + math.max(0,math.min(y+yMaxCurr,  self.w))]
                        - intData[math.max(0,math.min(x+xMaxCurr  ,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMaxCurr,  self.w))]
                        - intData[math.max(0,math.min(x+xMaxCurr+1,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMinCurr-1,self.w))]
                        + intData[math.max(0,math.min(x+xMaxCurr  ,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMinCurr-1,self.w))] )
                        * gradOutData[(x-1)*self.w + (y-1)]
                    
                    xMinDelta = xMinDelta
                        +(intData[math.max(0,math.min(x+xMinCurr-1,self.h))*(self.w+1) 
                            + math.max(0,math.min(y+yMaxCurr,  self.w))]
                        - intData[math.max(0,math.min(x+xMinCurr  ,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMaxCurr,  self.w))]
                        - intData[math.max(0,math.min(x+xMinCurr-1,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMinCurr-1,self.w))]
                        + intData[math.max(0,math.min(x+xMinCurr  ,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMinCurr-1,self.w))] )
                        * gradOutData[(x-1)*self.w + (y-1)]
                    
                    yMaxDelta = yMaxDelta
                        +(intData[math.max(0,math.min(x+xMaxCurr,  self.h))*(self.w+1) 
                            + math.max(0,math.min(y+yMaxCurr+1,self.w))]
                        - intData[math.max(0,math.min(x+xMaxCurr,  self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMaxCurr  ,self.w))]
                        - intData[math.max(0,math.min(x+xMinCurr-1,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMaxCurr+1,self.w))]
                        + intData[math.max(0,math.min(x+xMinCurr-1,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMaxCurr,  self.w))] )
                        * gradOutData[(x-1)*self.w + (y-1)]
                    
                    yMinDelta = yMinDelta
                        +(intData[math.max(0,math.min(x+xMaxCurr,  self.h))*(self.w+1) 
                            + math.max(0,math.min(y+yMinCurr-1,self.w))]
                        - intData[math.max(0,math.min(x+xMaxCurr,  self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMinCurr  ,self.w))]
                        - intData[math.max(0,math.min(x+xMinCurr-1,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMinCurr-1,self.w))]
                        + intData[math.max(0,math.min(x+xMinCurr-1,self.h))*(self.w+1)
                            + math.max(0,math.min(y+yMinCurr,  self.w))] )
                        * gradOutData[(x-1)*self.w + (y-1)]
                end
            end
            
            self.gradXMax[planeIdx] = self.gradXMax[planeIdx] + scale * (
                xMaxDelta * self.areaCoeff[planeIdx] -
                outputDot / (self.xMax[planeIdx] - self.xMin[planeIdx] + 1))
            self.gradXMin[planeIdx] = self.gradXMin[planeIdx] + scale * (
                xMinDelta * self.areaCoeff[planeIdx] +
                outputDot / (self.xMax[planeIdx] - self.xMin[planeIdx] + 1))
            self.gradYMax[planeIdx] = self.gradYMax[planeIdx] + scale * (
                yMaxDelta * self.areaCoeff[planeIdx] -
                outputDot / (self.yMax[planeIdx] - self.yMin[planeIdx] + 1))
            self.gradYMin[planeIdx] = self.gradYMin[planeIdx] + scale * (
                yMinDelta * self.areaCoeff[planeIdx] +
                outputDot / (self.yMax[planeIdx] - self.yMin[planeIdx] + 1))
        end
    end

    function Integral:zeroGradParameters()
        self.gradXMin:zero()
        self.gradYMin:zero()
        self.gradXMax:zero()
        self.gradYMax:zero()
    end

    function Integral:updateParameters(lr)
        self.xMin:add(lr, self.gradXMin)
        self.yMin:add(lr, self.gradYMin)
        self.xMax:add(lr, self.gradXMax)
        self.yMax:add(lr, self.gradYMax)
        
        self:recalculateArea()
    end
end