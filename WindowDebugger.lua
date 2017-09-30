cv = require 'cv'
require 'cv.videoio'
require 'cv.imgproc'

local WindowDebugger = torch.class('WindowDebugger')

do
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
