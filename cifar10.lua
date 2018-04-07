require 'paths'
require 'image'

local dataset = {
    mean = {125.3069, 122.9504, 113.8654},
    std = {62.9932, 62.0887, 66.7049},
    h = 32,
    w = 32,
    nClasses = 10
}

function dataset.loadNames(kind)
    --[[
        `kind`: 'train' or 'val'
        
        returns:
        {1, 2, 3, ..., 50000} or {50001, ..., 60000}
    --]]
    
    if not dataset.data then
        local file = torch.load(dataset.relative)

        if file.trainData.data:type() == 'torch.ByteTensor' then
            -- no whitening: mean-std normalize
            for _, field in ipairs{'trainData', 'testData'} do
                file[field].data = file[field].data:float()
                for k = 1,3 do
                    file[field].data[{{}, k}]:add(-dataset.mean[k])
                    file[field].data[{{}, k}]:div(dataset.std[k])
                end
            end
        else
            assert(file.trainData.data:type() == 'torch.FloatTensor')
        end

        dataset.data = {
            train = file.trainData.data,
            test  = file.testData.data}
        dataset.labels = {
            train = file.trainData.labels,
            test  = file.testData.labels}
    end

    local retval = {}
    if kind == 'train' then
        for k = 1,50000 do table.insert(retval, k) end
    elseif kind == 'val' then
        for k = 50001,60000 do table.insert(retval, k) end
    else
        error('kind should be train or val')
    end

    return retval
end

function dataset.loadSample(idx)
    if idx > 50000 then
        idx = idx - 50000
        return dataset.data.test[idx], dataset.labels.test[idx]
    else
        return dataset.data.train[idx], dataset.labels.train[idx]
    end
end

ffi = require 'ffi'

local C_lib = ffi.load('C/lib/libcityscapes-c.so')

ffi.cdef [[
void updateConfusionMatrix(
    long *confMatrix, long *predictedLabels,
    long *labels, int numPixels,
    int nClasses);
]]

function dataset.updateConfusionMatrix(confMatrix, predictedLabels, labels)
    -- confMatrix:      long, 10x10
    -- predictedLabels: long, 128
    -- labels:          byte, 128
    assert(predictedLabels:nElement() == labels:nElement())
    assert(predictedLabels:isContiguous() and labels:isContiguous())
    assert(confMatrix:type() == 'torch.LongTensor')
    assert(confMatrix:size(1) == confMatrix:size(2) and confMatrix:size(1) == dataset.nClasses)
    assert(predictedLabels:type() == 'torch.LongTensor')
    
    assert(labels:type() == 'torch.LongTensor')
    C_lib.updateConfusionMatrix(
        torch.data(confMatrix), torch.data(predictedLabels),
        torch.data(labels), predictedLabels:nElement(),
        dataset.nClasses)
end

function dataset.accuracy(confMatrix)
    return confMatrix:diag():sum() / confMatrix:sum()
end

return dataset
