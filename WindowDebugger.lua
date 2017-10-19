cv = require 'cv'
require 'cv.videoio'
require 'cv.imgproc'

local WindowDebugger = torch.class('WindowDebugger')

do
    local colormap = {
        {0.00000, 0.00000, 0.08333},
        {0.00000, 0.00000, 0.08333},
        {0.00000, 0.00000, 0.12500},
        {0.00000, 0.00000, 0.16667},
        {0.00000, 0.00000, 0.20833},
        {0.00000, 0.00000, 0.25000},
        {0.00000, 0.00000, 0.29167},
        {0.00000, 0.00000, 0.33333},
        {0.00000, 0.00000, 0.37500},
        {0.00000, 0.00000, 0.41667},
        {0.00000, 0.00000, 0.45833},
        {0.00000, 0.00000, 0.50000},
        {0.00000, 0.00000, 0.54167},
        {0.00000, 0.00000, 0.58333},
        {0.00000, 0.00000, 0.62500},
        {0.00000, 0.00000, 0.66667},
        {0.00000, 0.00000, 0.70833},
        {0.00000, 0.00000, 0.75000},
        {0.00000, 0.00000, 0.79167},
        {0.00000, 0.00000, 0.83333},
        {0.00000, 0.00000, 0.87500},
        {0.00000, 0.00000, 0.91667},
        {0.00000, 0.00000, 0.95833},
        {0.00000, 0.00000, 1.00000},
        {0.00000, 0.04167, 1.00000},
        {0.00000, 0.08333, 1.00000},
        {0.00000, 0.12500, 1.00000},
        {0.00000, 0.16667, 1.00000},
        {0.00000, 0.20833, 1.00000},
        {0.00000, 0.25000, 1.00000},
        {0.00000, 0.29167, 1.00000},
        {0.00000, 0.33333, 1.00000},
        {0.00000, 0.37500, 1.00000},
        {0.00000, 0.41667, 1.00000},
        {0.00000, 0.45833, 1.00000},
        {0.00000, 0.50000, 1.00000},
        {0.00000, 0.54167, 1.00000},
        {0.00000, 0.58333, 1.00000},
        {0.00000, 0.62500, 1.00000},
        {0.00000, 0.66667, 1.00000},
        {0.00000, 0.70833, 1.00000},
        {0.00000, 0.75000, 1.00000},
        {0.00000, 0.79167, 1.00000},
        {0.00000, 0.83333, 1.00000},
        {0.00000, 0.87500, 1.00000},
        {0.00000, 0.91667, 1.00000},
        {0.00000, 0.95833, 1.00000},
        {0.00000, 1.00000, 1.00000},
        {0.06250, 1.00000, 1.00000},
        {0.12500, 1.00000, 1.00000},
        {0.18750, 1.00000, 1.00000},
        {0.25000, 1.00000, 1.00000},
        {0.31250, 1.00000, 1.00000},
        {0.37500, 1.00000, 1.00000},
        {0.43750, 1.00000, 1.00000},
        {0.50000, 1.00000, 1.00000},
        {0.56250, 1.00000, 1.00000},
        {0.62500, 1.00000, 1.00000},
        {0.68750, 1.00000, 1.00000},
        {0.75000, 1.00000, 1.00000},
        {0.81250, 1.00000, 1.00000},
        {0.87500, 1.00000, 1.00000},
        {0.93750, 1.00000, 1.00000},
        {1.00000, 1.00000, 1.00000},
    }
    for _,c in ipairs(colormap) do for p = 1,3 do c[p] = c[p] * 255 end end

    -- `self.h` is just a wrapper over all useful data
    function WindowDebugger:__init(path)
        if path then
            self:load(path)
        else
            local h = {}
            for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
                h[param] = torch.FloatTensor()
            end
            h.size = 0
            h.growthFactor = 1.5

            self.h = h
        end
    end
    
    function WindowDebugger:save(path)
        torch.save(path, self.h)
    end
    
    function WindowDebugger:load(path)
        self.h = torch.load(path)
    end

    function WindowDebugger:add(intModule)
        local numWindowsToDisplay = math.min(intModule.xMin:nElement(), 150)

        -- check if we need to grow Tensors
        if self.h.size == 0 or self.h.size == self.h.xMin:size(1) then
            local newCapacity = self.h.size == 0 and 10 or (self.h.size * 1.5)
            for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
                self.h[param]:resize(newCapacity, numWindowsToDisplay)
            end
        end
        self.h.size = self.h.size + 1

        for _,param in ipairs{'xMin', 'xMax', 'yMin', 'yMax'} do
            self.h[param][self.h.size]:copy(intModule[param]:view(-1)[{{1,numWindowsToDisplay}}])
            self.h[param][self.h.size]:mul(intModule.reparametrization or 1)
        end

        self.h.h = intModule.h
        self.h.w = intModule.w
    end

    function WindowDebugger.drawBoxes(module, scores)
        scores = scores:view(-1)
        scores:add(-scores:min())
        scores:div(scores:max())
        local scoresAscIdx = select(2, torch.sort(scores))

        local xMin = module.xMin:float():mul(module.reparametrization or 1):view(-1)
        local xMax = module.xMax:float():mul(module.reparametrization or 1):view(-1)
        local yMin = module.yMin:float():mul(module.reparametrization or 1):view(-1)
        local yMax = module.yMax:float():mul(module.reparametrization or 1):view(-1)

        local imH, imW = 500, 500
        local frame = torch.ByteTensor(imH*2, imW*2, 3):zero()
        frame[{{imH-1,imH+1}, {}}]:fill(80)
        frame[{{}, {imW-1,imW+1}}]:fill(80)

        for idx = 1,xMin:nElement() do
            local rect = scoresAscIdx[idx]
            
            if scores[rect] > 2/63 then
                local thickness = (xMin[rect] > xMax[rect] or yMin[rect] > yMax[rect]) and -1 or 2

                local cmapIdx = math.ceil(scores[rect] * 63 + 0.5)
                assert(1 <= cmapIdx and cmapIdx <= #colormap)    
                local color = colormap[cmapIdx]

                cv.rectangle{frame,
                    {xMin[rect]/module.h*imH + imH, yMin[rect]/module.w*imW + imW},
                    {xMax[rect]/module.h*imH + imH, yMax[rect]/module.w*imW + imW},
                    color,
                    thickness
                }
            end
        end

        cv.cvtColor{frame, frame, cv.COLOR_BGR2RGB}
        return frame:permute(3,1,2):clone()
    end
    
    -- refRects: {{xMin, yMin, xMax, yMax}, {xMin, yMin, xMax, yMax}, ...}
    function WindowDebugger:exportVideo(path, refRects)
        local vw = cv.VideoWriter{
                path, cv.VideoWriter.fourcc{'H','2','6','4'}, 
                12, {100*2*1.5, 100*2*1.5}}
        assert(vw:isOpened())
        
        local colors = {
            {255,   0,   0},
            {  0, 255,   0},
            {  0,   0, 255},
            {255, 255, 255},
            {255, 255,   0},
            {255,   0, 255},
            {  0, 255, 255},
            {130, 130, 130},
            {255,  60, 160},
            { 60, 170, 255}
        }
        
        local imH, imW = 100, 100
        local frame = torch.ByteTensor(imH*2, imW*2, 3)
        local nRects = self.h.xMin:size(2)
        
        for i = 1,self.h.size do
            frame:zero()
            frame[frame:size(1) / 2]:fill(80)
            frame[{{}, frame:size(2) / 2}]:fill(80)
            
            for rect = 1,nRects do
                local thickness = 1
                
                if 
                    self.h.xMin[{i,rect}] > self.h.xMax[{i,rect}] or
                    self.h.yMin[{i,rect}] > self.h.yMax[{i,rect}]
                then
                    thickness = -1
                end
                    
                cv.rectangle{frame, 
                    {self.h.xMin[{i,rect}]/self.h.h*imH + imH, self.h.yMin[{i,rect}]/self.h.w*imW + imW},
                    {self.h.xMax[{i,rect}]/self.h.h*imH + imH, self.h.yMax[{i,rect}]/self.h.w*imW + imW},
                    colors[(rect-1) % #colors + 1],
                    thickness
                }
            end
            
            if refRects then
                for i,rect in ipairs(refRects) do
                    cv.rectangle{frame, 
                        {rect[1]/self.h.h*imH + imH, rect[2]/self.h.w*imW + imW},
                        {rect[3]/self.h.h*imH + imH, rect[4]/self.h.w*imW + imW},
                        {100, 100, 0},
                        1
                    }
                end
            end
            
            vw:write{cv.resize{frame, fx=1.5, fy=1.5}}
        end
        
        vw:release{}
    end
end
