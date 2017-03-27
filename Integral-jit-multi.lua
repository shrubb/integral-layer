require 'nn'

local _, parent = torch.class('Integral', 'nn.Module')

do
    cv = require 'cv'
    require 'cv.imgproc'
    require 'cv.highgui'

    function Integral:__init(nWindows, h, w)
        parent.__init(self)
        self.nWindows, self.h, self.w = nWindows, h, w
        self.output = torch.Tensor(self.nWindows, h, w)
        self.integralDouble = torch.DoubleTensor()
        self.integral = torch.FloatTensor()
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
        -- the only parameters of the module. Randomly initialize them
        self.xMin = torch.round((torch.rand(self.nWindows) - 0.5) * (2 * self.h * 0.3))
        self.yMin = torch.round((torch.rand(self.nWindows) - 0.5) * (2 * self.w * 0.3))
        self.xMax = torch.Tensor(self.nWindows)
        self.yMax = torch.Tensor(self.nWindows)
        
        for i = 1,self.nWindows do
            self.xMax[i] = torch.round(torch.uniform(self.xMin[i] + self.h * 0.05, self.xMin[i] + self.h * 0.25))
            self.yMax[i] = torch.round(torch.uniform(self.yMin[i] + self.w * 0.05, self.yMin[i] + self.w * 0.25))
        end
        
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

    local function round_down(x)
        local rounded = math.floor(x)
        return rounded, x-rounded -- return integer and fractional parts
    end

    local function round_up(x)
        local rounded = math.ceil(x)
        return rounded, rounded-x -- return integer and fractional parts
    end

    function Integral:updateOutput(input)
        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):forward(input)
        end
        
        assert(input:size(2) == self.h and input:size(3) == self.w)

        self.output:resize(input:size(1)*self.nWindows, input:size(2), input:size(3))
        
        self.integralDouble:resize(input:size(1), input:size(2)+1, input:size(3)+1)
        self.integral:resize(self.integralDouble:size())

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
                local outPlane = self.output[outPlaneIdx]
                
                local outData = torch.data(outPlane)
                local intData = torch.data(self.integral[inPlaneIdx])
                
                for x = 0,self.h-1 do
                    for y = 0,self.w-1 do
                        outData[x*self.w + y] = 0
                            + intData[math.max(0,math.min(x+xMaxCurr,self.h))*(self.w+1) 
                                + math.max(0,math.min(y+yMaxCurr,self.w))]
                            - intData[math.max(0,math.min(x+xMinCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMaxCurr,self.w))]
                            - intData[math.max(0,math.min(x+xMaxCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr,self.w))]
                            + intData[math.max(0,math.min(x+xMinCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr,self.w))]

                        --[[ BEGIN FRACTIONAL PART
                            -- xMax border
                            +(intData[math.max(0,math.min(x+xMaxCurr+1,self.h))*(self.w+1) 
                                + math.max(0,math.min(y+yMaxCurr,self.w))]
                            - intData[math.max(0,math.min(x+xMaxCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMaxCurr,self.w))]
                            - intData[math.max(0,math.min(x+xMaxCurr+1,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr,self.w))]
                            + intData[math.max(0,math.min(x+xMaxCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr,self.w))]
                            ) * xMaxCurrFrac

                            -- yMax border
                            +(intData[math.max(0,math.min(x+xMaxCurr,self.h))*(self.w+1) 
                                + math.max(0,math.min(y+yMaxCurr+1,self.w))]
                            - intData[math.max(0,math.min(x+xMaxCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMaxCurr,self.w))]
                            - intData[math.max(0,math.min(x+xMinCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMaxCurr+1,self.w))]
                            + intData[math.max(0,math.min(x+xMinCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMaxCurr,self.w))]
                            ) * yMaxCurrFrac

                            -- xMin border
                            +(intData[math.max(0,math.min(x+xMinCurr,self.h))*(self.w+1) 
                                + math.max(0,math.min(y+yMaxCurr,self.w))]
                            - intData[math.max(0,math.min(x+xMinCurr-1,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMaxCurr,self.w))]
                            - intData[math.max(0,math.min(x+xMinCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr,self.w))]
                            + intData[math.max(0,math.min(x+xMinCurr-1,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr,self.w))]
                            ) * xMinCurrFrac

                            -- yMin border
                            +(intData[math.max(0,math.min(x+xMaxCurr,self.h))*(self.w+1) 
                                + math.max(0,math.min(y+yMinCurr,self.w))]
                            - intData[math.max(0,math.min(x+xMaxCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr-1,self.w))]
                            - intData[math.max(0,math.min(x+xMinCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr,self.w))]
                            + intData[math.max(0,math.min(x+xMinCurr,self.h))*(self.w+1)
                                + math.max(0,math.min(y+yMinCurr-1,self.w))]
                            ) * yMinCurrFrac
                    end
                end

                local inData = torch.data(input[inPlaneIdx])

                for x = 0,self.h-1 do
                    for y = 0,self.w-1 do
                        -- corner pixels
                        outData[x*self.w + y] = outData[x*self.w + y]

                            + xMaxCurrFrac*yMaxCurrFrac * (
                                   (x+xMaxCurr > self.w-1 or
                                    y+yMaxCurr > self.h-1 or
                                    x+xMaxCurr < 0        or
                                    y+yMaxCurr < 0) and 0
                            or inData[(x+xMaxCurr)*self.w + (y+yMaxCurr)])

                            + xMinCurrFrac*yMaxCurrFrac * (
                                   (x+xMinCurr-1 > self.w-1 or
                                    y+yMaxCurr   > self.h-1 or
                                    x+xMinCurr-1 < 0        or
                                    y+yMaxCurr   < 0) and 0
                            or inData[(x+xMinCurr-1)*self.w + (y+yMaxCurr)])

                            + xMaxCurrFrac*yMinCurrFrac * (
                                   (x+xMaxCurr   > self.w-1 or
                                    y+yMinCurr-1 > self.h-1 or
                                    x+xMaxCurr   < 0        or
                                    y+yMinCurr-1 < 0) and 0
                            or inData[(x+xMaxCurr)*self.w + (y+yMinCurr-1)])

                            + xMinCurrFrac*yMinCurrFrac * (
                                   (x+xMinCurr-1 > self.w-1 or
                                    y+yMinCurr-1 > self.h-1 or
                                    x+xMinCurr-1 < 0        or
                                    y+yMinCurr-1 < 0) and 0
                            or inData[(x+xMinCurr-1)*self.w + (y+yMinCurr-1)])

                        -- END FRACTIONAL PART ]]
                    end
                end
                
                outPlane:mul(self.areaCoeff[nWindow])
            end
        end
        
        return self.output
    end

    function Integral:updateGradInput(input, gradOutput)
        if self.gradInput then
            
            if input:nDimension() == 2 then
                input = nn.Unsqueeze(1):forward(input)
            end

            -- never call :backward() on backpropHelper!
            self.backpropHelper = self.backpropHelper or Integral(1, self.h, self.w)
        
            self.gradInput:resize(input:size()):zero()
            
            for inPlaneIdx = 1,input:size(1) do
                for nWindow = 1,self.nWindows do
                    self.backpropHelper.xMin[1] = -self.xMax[nWindow]
                    self.backpropHelper.xMax[1] = -self.xMin[nWindow]
                    self.backpropHelper.yMin[1] = -self.yMax[nWindow]
                    self.backpropHelper.yMax[1] = -self.yMin[nWindow]
                    self.backpropHelper:recalculateArea()
                    
                    local outPlaneIdx = self.nWindows*(inPlaneIdx-1) + nWindow

                    self.gradInput[inPlaneIdx]:add(
                        self.backpropHelper:forward(gradOutput[outPlaneIdx]):squeeze())
                end
            end
            
            return self.gradInput
        end
    end

    function Integral:accGradParameters(input, gradOutput, scale)
            
        if input:nDimension() == 2 then
            input = nn.Unsqueeze(1):forward(input)
        end

        scale = scale or 1

        for inPlaneIdx = 1,input:size(1) do
            for nWindow = 1,self.nWindows do
                local outPlaneIdx = self.nWindows*(inPlaneIdx-1) + nWindow
                local outputDot = torch.dot(self.output[outPlaneIdx], gradOutput[outPlaneIdx])
                
                -- round towards zero (?)
                -- and +1 because OpenCV's integral adds extra row and col
                local xMinCurr = round_down(self.xMin[nWindow])
                local xMaxCurr = round_down(self.xMax[nWindow])
                local yMinCurr = round_down(self.yMin[nWindow])
                local yMaxCurr = round_down(self.yMax[nWindow])
                
                local gradOutData = torch.data(gradOutput[outPlaneIdx])
                local intData = torch.data(self.integral[inPlaneIdx])
                
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
                
                self.gradXMax[nWindow] = self.gradXMax[nWindow] + scale * (
                    xMaxDelta * self.areaCoeff[nWindow] -
                    outputDot / (self.xMax[nWindow] - self.xMin[nWindow] + 1))
                self.gradXMin[nWindow] = self.gradXMin[nWindow] + scale * (
                    xMinDelta * self.areaCoeff[nWindow] +
                    outputDot / (self.xMax[nWindow] - self.xMin[nWindow] + 1))
                self.gradYMax[nWindow] = self.gradYMax[nWindow] + scale * (
                    yMaxDelta * self.areaCoeff[nWindow] -
                    outputDot / (self.yMax[nWindow] - self.yMin[nWindow] + 1))
                self.gradYMin[nWindow] = self.gradYMin[nWindow] + scale * (
                    yMinDelta * self.areaCoeff[nWindow] +
                    outputDot / (self.yMax[nWindow] - self.yMin[nWindow] + 1))
            end
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