require 'paths'
require 'image'

cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local cityscapes = {}

cityscapes.mean = torch.FloatTensor{0.3621072769165, 0.38127624988556, 0.45571011304855} * 255
cityscapes.std  = torch.FloatTensor{0.28512728214264, 0.27228525280952, 0.27148124575615} * 255
cityscapes.dsize = {560, 424}
cityscapes.nClasses = 40
-- precomputed class frequencies
cityscapes.classProbs = torch.FloatTensor {
    0.23865289986134,   
    0.12035804986954,   
    0.081658005714417,  
    0.045853577554226,  
    0.0488794259727,    
    0.03106333129108,   
    0.027580136433244,  
    0.02241432480514,   
    0.023642545565963,  
    0.022577280178666,  
    0.021521495655179,  
    0.017949098721147,  
    0.017618739977479,  
    0.013188065961003,  
    0.015089756809175,  
    0.010573507286608,  
    0.011819037608802,  
    0.011056656949222,  
    0.011324374005198,  
    0.0083226189017296, 
    0.010846924968064,  
    0.00954554323107,   
    0.0068231713958085, 
    0.006331654265523,  
    0.0070201237685978, 
    0.0049646939150989, 
    0.0042245327495039, 
    0.0044671418145299, 
    0.0053962399251759, 
    0.0032902958337218, 
    0.0037424962501973, 
    0.0033978160936385, 
    0.0038722236640751, 
    0.0032962430268526, 
    0.0034833224490285, 
    0.0028762461151928, 
    0.0027942787855864, 
    0.026443000882864,  
    0.024739176034927,  
    0.061301950365305,  
}
-- cityscapes.classWeights = cityscapes.classProbs:clone():pow(-1/2.5)
cityscapes.classWeights = cityscapes.classProbs:clone():add(1.10):log():pow(-1)
-- add "unlabeled" class with zero weight
cityscapes.classWeights = torch.cat(cityscapes.classWeights, torch.FloatTensor{0})

function cityscapes.loadNames(kind)
    --[[
        `kind`: 'train' or 'val'
        
        returns:
        {5001, 5004, 5005, ..., 6440}
    --]]
    
    if kind == 'train' then
        return {
            5003, 5004, 5005, 5006, 5007, 5008, 5010, 5011, 5012, 5013, 5019, 5020, 5022, 
            5023, 5024, 5025, 5026, 5027, 5044, 5045, 5048, 5049, 5050, 5051, 5052, 5053, 
            5054, 5055, 5058, 5064, 5065, 5066, 5067, 5068, 5069, 5070, 5071, 5072, 5073, 
            5074, 5075, 5080, 5081, 5082, 5083, 5092, 5093, 5094, 5095, 5096, 5097, 5098, 
            5099, 5100, 5101, 5102, 5103, 5104, 5105, 5106, 5107, 5108, 5109, 5110, 5111, 
            5112, 5113, 5114, 5115, 5116, 5120, 5121, 5122, 5123, 5124, 5130, 5135, 5136, 
            5138, 5139, 5140, 5141, 5142, 5143, 5144, 5145, 5146, 5147, 5148, 5149, 5150, 
            5151, 5152, 5156, 5157, 5158, 5159, 5160, 5161, 5162, 5163, 5164, 5165, 5166, 
            5170, 5177, 5178, 5179, 5203, 5204, 5205, 5206, 5213, 5214, 5215, 5216, 5217, 
            5218, 5219, 5223, 5224, 5225, 5226, 5227, 5228, 5229, 5230, 5231, 5232, 5233, 
            5234, 5235, 5236, 5237, 5238, 5239, 5240, 5241, 5242, 5243, 5244, 5245, 5246, 
            5247, 5248, 5249, 5251, 5252, 5253, 5254, 5255, 5256, 5257, 5258, 5259, 5260, 
            5261, 5262, 5263, 5265, 5266, 5267, 5268, 5269, 5270, 5274, 5275, 5276, 5277, 
            5278, 5286, 5287, 5288, 5289, 5290, 5291, 5292, 5293, 5294, 5295, 5303, 5304, 
            5305, 5306, 5307, 5308, 5309, 5313, 5314, 5318, 5319, 5320, 5321, 5322, 5323, 
            5324, 5336, 5337, 5338, 5339, 5340, 5341, 5342, 5343, 5344, 5345, 5346, 5347, 
            5348, 5349, 5350, 5353, 5354, 5365, 5366, 5367, 5368, 5369, 5370, 5371, 5372, 
            5373, 5374, 5375, 5376, 5377, 5378, 5379, 5380, 5381, 5382, 5383, 5391, 5392, 
            5393, 5394, 5398, 5399, 5400, 5401, 5402, 5403, 5404, 5405, 5406, 5407, 5408, 
            5409, 5410, 5415, 5416, 5417, 5418, 5419, 5420, 5421, 5422, 5423, 5424, 5425, 
            5426, 5427, 5428, 5429, 5436, 5437, 5438, 5439, 5440, 5449, 5450, 5451, 5452, 
            5453, 5454, 5455, 5456, 5457, 5458, 5459, 5460, 5461, 5467, 5468, 5478, 5479, 
            5480, 5481, 5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489, 5490, 5491, 5492, 
            5493, 5494, 5495, 5496, 5497, 5498, 5499, 5500, 5501, 5502, 5503, 5504, 5505, 
            5506, 5507, 5514, 5527, 5528, 5529, 5530, 5534, 5535, 5536, 5540, 5541, 5542, 
            5543, 5544, 5545, 5546, 5547, 5548, 5552, 5553, 5554, 5572, 5573, 5574, 5575, 
            5576, 5577, 5578, 5584, 5585, 5586, 5587, 5588, 5589, 5590, 5595, 5596, 5597, 
            5598, 5599, 5600, 5601, 5602, 5608, 5609, 5610, 5611, 5614, 5615, 5616, 5622, 
            5623, 5624, 5625, 5626, 5627, 5628, 5629, 5630, 5631, 5632, 5639, 5640, 5641, 
            5642, 5643, 5646, 5647, 5648, 5649, 5652, 5653, 5654, 5655, 5659, 5660, 5661, 
            5662, 5665, 5666, 5667, 5674, 5675, 5682, 5683, 5684, 5685, 5691, 5692, 5695, 
            5696, 5700, 5701, 5702, 5703, 5704, 5705, 5714, 5715, 5716, 5719, 5720, 5721, 
            5722, 5723, 5729, 5730, 5735, 5736, 5737, 5738, 5739, 5740, 5741, 5742, 5745, 
            5746, 5747, 5748, 5749, 5750, 5751, 5752, 5753, 5754, 5755, 5756, 5757, 5758, 
            5788, 5789, 5790, 5791, 5792, 5793, 5794, 5795, 5796, 5797, 5798, 5799, 5805, 
            5806, 5807, 5808, 5809, 5815, 5816, 5817, 5818, 5819, 5820, 5824, 5825, 5826, 
            5827, 5828, 5829, 5830, 5831, 5832, 5847, 5848, 5849, 5853, 5854, 5855, 5856, 
            5863, 5864, 5865, 5866, 5867, 5868, 5872, 5873, 5874, 5875, 5876, 5877, 5878, 
            5879, 5880, 5881, 5882, 5883, 5884, 5885, 5886, 5887, 5888, 5889, 5890, 5891, 
            5892, 5893, 5894, 5895, 5896, 5897, 5898, 5899, 5900, 5901, 5902, 5903, 5904, 
            5905, 5909, 5910, 5911, 5912, 5913, 5914, 5915, 5916, 5920, 5921, 5922, 5923, 
            5924, 5925, 5929, 5930, 5931, 5936, 5937, 5938, 5939, 5940, 5941, 5942, 5943, 
            5944, 5948, 5949, 5950, 5951, 5952, 5953, 5954, 5955, 5956, 5957, 5958, 5963, 
            5964, 5968, 5969, 5978, 5979, 5980, 5981, 5982, 5983, 5984, 5985, 5986, 5987, 
            5988, 5989, 5990, 5996, 5997, 5998, 5999, 6000, 6005, 6006, 6007, 6008, 6009, 
            6013, 6014, 6015, 6016, 6017, 6018, 6019, 6020, 6024, 6025, 6026, 6027, 6028, 
            6029, 6030, 6031, 6035, 6036, 6037, 6040, 6041, 6042, 6043, 6044, 6045, 6046, 
            6047, 6050, 6051, 6054, 6055, 6056, 6059, 6060, 6061, 6062, 6063, 6064, 6065, 
            6066, 6067, 6068, 6069, 6070, 6071, 6072, 6073, 6074, 6085, 6086, 6087, 6097, 
            6105, 6110, 6111, 6112, 6113, 6114, 6115, 6116, 6120, 6121, 6122, 6132, 6133, 
            6134, 6137, 6138, 6139, 6140, 6141, 6142, 6143, 6159, 6160, 6161, 6168, 6169, 
            6172, 6173, 6177, 6178, 6185, 6186, 6187, 6188, 6189, 6190, 6191, 6197, 6198, 
            6199, 6200, 6213, 6214, 6215, 6221, 6222, 6223, 6224, 6225, 6231, 6232, 6236, 
            6237, 6238, 6239, 6240, 6241, 6242, 6243, 6244, 6245, 6246, 6251, 6252, 6253, 
            6266, 6267, 6268, 6269, 6270, 6271, 6272, 6273, 6274, 6281, 6282, 6283, 6284, 
            6296, 6300, 6301, 6309, 6310, 6311, 6312, 6313, 6316, 6317, 6318, 6319, 6320, 
            6321, 6322, 6323, 6324, 6325, 6326, 6327, 6328, 6333, 6334, 6341, 6342, 6343, 
            6344, 6345, 6346, 6350, 6351, 6352, 6357, 6358, 6359, 6360, 6361, 6362, 6363, 
            6366, 6367, 6370, 6371, 6372, 6373, 6374, 6375, 6376, 6377, 6378, 6379, 6380, 
            6381, 6382, 6383, 6392, 6393, 6402, 6403, 6404, 6405, 6406, 6415, 6416, 6417, 
            6418, 6419, 6420, 6425, 6426, 6427, 6428, 6429, 6434, 6435, 6436, 6437, 6438, 
            6439, 6440,}
    elseif kind == 'val' then
        return {
            5001, 5002, 5009, 5014, 5015, 5016, 5017, 5018, 5021, 5028, 5029, 5030, 5031, 
            5032, 5033, 5034, 5035, 5036, 5037, 5038, 5039, 5040, 5041, 5042, 5043, 5046, 
            5047, 5056, 5057, 5059, 5060, 5061, 5062, 5063, 5076, 5077, 5078, 5079, 5084, 
            5085, 5086, 5087, 5088, 5089, 5090, 5091, 5117, 5118, 5119, 5125, 5126, 5127, 
            5128, 5129, 5131, 5132, 5133, 5134, 5137, 5153, 5154, 5155, 5167, 5168, 5169, 
            5171, 5172, 5173, 5174, 5175, 5176, 5180, 5181, 5182, 5183, 5184, 5185, 5186, 
            5187, 5188, 5189, 5190, 5191, 5192, 5193, 5194, 5195, 5196, 5197, 5198, 5199, 
            5200, 5201, 5202, 5207, 5208, 5209, 5210, 5211, 5212, 5220, 5221, 5222, 5250, 
            5264, 5271, 5272, 5273, 5279, 5280, 5281, 5282, 5283, 5284, 5285, 5296, 5297, 
            5298, 5299, 5300, 5301, 5302, 5310, 5311, 5312, 5315, 5316, 5317, 5325, 5326, 
            5327, 5328, 5329, 5330, 5331, 5332, 5333, 5334, 5335, 5351, 5352, 5355, 5356, 
            5357, 5358, 5359, 5360, 5361, 5362, 5363, 5364, 5384, 5385, 5386, 5387, 5388, 
            5389, 5390, 5395, 5396, 5397, 5411, 5412, 5413, 5414, 5430, 5431, 5432, 5433, 
            5434, 5435, 5441, 5442, 5443, 5444, 5445, 5446, 5447, 5448, 5462, 5463, 5464, 
            5465, 5466, 5469, 5470, 5471, 5472, 5473, 5474, 5475, 5476, 5477, 5508, 5509, 
            5510, 5511, 5512, 5513, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 
            5524, 5525, 5526, 5531, 5532, 5533, 5537, 5538, 5539, 5549, 5550, 5551, 5555, 
            5556, 5557, 5558, 5559, 5560, 5561, 5562, 5563, 5564, 5565, 5566, 5567, 5568, 
            5569, 5570, 5571, 5579, 5580, 5581, 5582, 5583, 5591, 5592, 5593, 5594, 5603, 
            5604, 5605, 5606, 5607, 5612, 5613, 5617, 5618, 5619, 5620, 5621, 5633, 5634, 
            5635, 5636, 5637, 5638, 5644, 5645, 5650, 5651, 5656, 5657, 5658, 5663, 5664, 
            5668, 5669, 5670, 5671, 5672, 5673, 5676, 5677, 5678, 5679, 5680, 5681, 5686, 
            5687, 5688, 5689, 5690, 5693, 5694, 5697, 5698, 5699, 5706, 5707, 5708, 5709, 
            5710, 5711, 5712, 5713, 5717, 5718, 5724, 5725, 5726, 5727, 5728, 5731, 5732, 
            5733, 5734, 5743, 5744, 5759, 5760, 5761, 5762, 5763, 5764, 5765, 5766, 5767, 
            5768, 5769, 5770, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778, 5779, 5780, 
            5781, 5782, 5783, 5784, 5785, 5786, 5787, 5800, 5801, 5802, 5803, 5804, 5810, 
            5811, 5812, 5813, 5814, 5821, 5822, 5823, 5833, 5834, 5835, 5836, 5837, 5838, 
            5839, 5840, 5841, 5842, 5843, 5844, 5845, 5846, 5850, 5851, 5852, 5857, 5858, 
            5859, 5860, 5861, 5862, 5869, 5870, 5871, 5906, 5907, 5908, 5917, 5918, 5919, 
            5926, 5927, 5928, 5932, 5933, 5934, 5935, 5945, 5946, 5947, 5959, 5960, 5961, 
            5962, 5965, 5966, 5967, 5970, 5971, 5972, 5973, 5974, 5975, 5976, 5977, 5991, 
            5992, 5993, 5994, 5995, 6001, 6002, 6003, 6004, 6010, 6011, 6012, 6021, 6022, 
            6023, 6032, 6033, 6034, 6038, 6039, 6048, 6049, 6052, 6053, 6057, 6058, 6075, 
            6076, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6084, 6088, 6089, 6090, 6091, 
            6092, 6093, 6094, 6095, 6096, 6098, 6099, 6100, 6101, 6102, 6103, 6104, 6106, 
            6107, 6108, 6109, 6117, 6118, 6119, 6123, 6124, 6125, 6126, 6127, 6128, 6129, 
            6130, 6131, 6135, 6136, 6144, 6145, 6146, 6147, 6148, 6149, 6150, 6151, 6152, 
            6153, 6154, 6155, 6156, 6157, 6158, 6162, 6163, 6164, 6165, 6166, 6167, 6170, 
            6171, 6174, 6175, 6176, 6179, 6180, 6181, 6182, 6183, 6184, 6192, 6193, 6194, 
            6195, 6196, 6201, 6202, 6203, 6204, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 
            6212, 6216, 6217, 6218, 6219, 6220, 6226, 6227, 6228, 6229, 6230, 6233, 6234, 
            6235, 6247, 6248, 6249, 6250, 6254, 6255, 6256, 6257, 6258, 6259, 6260, 6261, 
            6262, 6263, 6264, 6265, 6275, 6276, 6277, 6278, 6279, 6280, 6285, 6286, 6287, 
            6288, 6289, 6290, 6291, 6292, 6293, 6294, 6295, 6297, 6298, 6299, 6302, 6303, 
            6304, 6305, 6306, 6307, 6308, 6314, 6315, 6329, 6330, 6331, 6332, 6335, 6336, 
            6337, 6338, 6339, 6340, 6347, 6348, 6349, 6353, 6354, 6355, 6356, 6364, 6365, 
            6368, 6369, 6384, 6385, 6386, 6387, 6388, 6389, 6390, 6391, 6394, 6395, 6396, 
            6397, 6398, 6399, 6400, 6401, 6407, 6408, 6409, 6410, 6411, 6412, 6413, 6414, 
            6421, 6422, 6423, 6424, 6430, 6431, 6432, 6433, 6441, 6442, 6443, 6444, 6445, 
            6446, 6447, 6448, 6449,}
    end
end

function cityscapes.calcMean(files)
    local retval = torch.FloatTensor{0, 0, 0}
    
    for i,idx in ipairs(files) do
        local imgFile = cityscapes.relative .. ('colorImage/img_%04d.png'):format(idx)
        retval:add(cv.imread{imgFile, cv.IMREAD_COLOR}:view(-1, 3):float():div(255):mean(1):squeeze())

        if i % 100 == 0 then print(i); collectgarbage() end
    end
    
    return retval:div(#files)
end

function cityscapes.calcStd(files, mean)
    local retval = torch.FloatTensor{0, 0, 0}
    
    for i,idx in ipairs(files) do
        local imgFile = cityscapes.relative .. ('colorImage/img_%04d.png'):format(idx)
        local img = cv.imread{imgFile, cv.IMREAD_COLOR}:view(-1, 3):float():div(255)
        local squareDiff = img:add(-cityscapes.mean:view(1,3):expandAs(img)):pow(2):mean(1):squeeze()
        retval:add(squareDiff)

        if i % 100 == 0 then print(i); collectgarbage() end
    end
    
    return retval:div(#files):sqrt()
end

function cityscapes.loadSample(idx, _, augment)
    local imagePath  = cityscapes.relative .. ('colorImage/img_%04d.png'):format(idx)
    local labelsPath = cityscapes.relative .. ('groundTruthPng/labels_%04d.png'):format(idx)

    -- load image
    local img = cv.imread{imagePath, cv.IMREAD_COLOR}[{{1,-2}}]
    if img:size(1) ~= cityscapes.dsize[2] or 
       img:size(2) ~= cityscapes.dsize[1] then
        img = cv.resize{img, cityscapes.dsize, interpolation=cv.INTER_CUBIC}
    end

    -- load labels
    local labels = cv.imread{labelsPath, cv.IMREAD_ANYCOLOR}[{{1,-2}}]
    assert(labels:nDimension() == 2)
    if labels:size(1) ~= cityscapes.dsize[2] or
       labels:size(2) ~= cityscapes.dsize[1] then
        labels = cv.resize{labels, cityscapes.dsize, interpolation=cv.INTER_NEAREST}
    end
    labels[labels:eq(255)] = nClasses+1

    if augment then
        local flip = torch.random(2) == 1
        local maybeFlipMatrix = torch.eye(3):double()
        if flip then
            maybeFlipMatrix[{1,1}] = -1
            maybeFlipMatrix[{1,3}] = labels:size(2)
        end
        
        local angle = (math.random() * 2 - 1) * 8
        local scaleFactor = math.random() * 1.5 + 0.5
        local imageCenter = {labels:size(2) / 2, labels:size(1) / 2}
        local rotationMatrix = torch.eye(3):double()
        rotationMatrix[{{1,2}}]:copy(cv.getRotationMatrix2D{imageCenter, angle, scaleFactor})
        
        local transformationMatrix = torch.mm(maybeFlipMatrix, rotationMatrix)[{{1,2}}]
        
        img = cv.warpAffine{
            img, transformationMatrix, flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REFLECT}
        labels = cv.warpAffine{
            labels, transformationMatrix, flags=cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT, borderValue={nClasses+1}}
        
        if torch.random(2) == 1 then
            local blurSigma = math.random() * 2.3 + 0.7
            cv.GaussianBlur{img, {5, 5}, blurSigma, dst=img, borderType=cv.BORDER_REFLECT}
        end
    end

    cv.cvtColor{img, img, cv.COLOR_BGR2RGB}
    img = img:permute(3,1,2):float()
    -- normalize image globally
    for ch = 1,3 do
        img[ch]:add(-cityscapes.mean[ch])
        img[ch]:mul(1/cityscapes.std[ch])
    end

    return img, labels
end

local labelToColor = {
    {166,83,94},
    {206,182,242},
    {0,238,255},
    {170,255,0},
    {229,145,115},
    {242,0,97},
    {65,0,242},
    {38,153,145},
    {124,166,41},
    {77,60,57},
    {230,172,195},
    {0,27,102},
    {191,255,251},
    {242,206,61},
    {255,0,0},
    {102,0,54},
    {102,129,204},
    {0,230,153},
    {217,206,163},
    {191,48,48},
    {51,26,39},
    {0,92,230},
    {29,115,52},
    {102,77,26},
    {76,19,19},
    {242,61,182},
    {13,28,51},
    {40,51,38},
    {242,129,0},
    {86,45,89},
    {0,85,128},
    {115,140,105},
    {51,27,0},
    {143,48,191},
    {115,191,230},
    {195,230,172},
    {217,166,108},
    {129,105,140},
    {38,64,77},
    {41,64,16},
    {127,51,0}
}

for label, color in ipairs(labelToColor) do
    -- color[3], color[1] = color[1], color[3]
    for k = 1,3 do
        color[k] = color[k] / 255
    end
end

require 'nn'

function cityscapes.renderLabels(labels, img, blendCoeff)
    
    local retval = torch.FloatTensor(3, cityscapes.dsize[2], cityscapes.dsize[1]):zero()
    for label, color in ipairs(labelToColor) do
        local mask = nn.utils.addSingletonDimension(labels:eq(label))
        for ch = 1,3 do
            retval[ch][mask] = color[ch]
        end
    end
    
    if img then
        local labelsBlendCoeff = blendCoeff or 0.62
        retval:mul(labelsBlendCoeff)
        
        local MIN = img:min()
        local MAX = img:max() - MIN
        retval:add((1 - labelsBlendCoeff) / MAX, img)
        retval:add(- MIN * (1 - labelsBlendCoeff) / MAX)
    end
    
    return retval
end

function cityscapes.calcClassProbs(trainFiles)
    local counts = torch.DoubleTensor(cityscapes.nClasses):zero()

    for i,idx in ipairs(trainFiles) do
        local labelsPath = cityscapes.relative .. ('groundTruthPng/labels_%04d.png'):format(idx)
        local labels = cv.imread{labelsPath, cv.IMREAD_GRAYSCALE}
        for class = 1,40 do
            counts[class] = counts[class] + labels:eq(class):sum()
        end
        if i % 100 == 0 then print(i); collectgarbage() end
    end

    return counts:div(counts:sum()):float()
end

function cityscapes.labelsToEval(labels)
    return labels
end

ffi = require 'ffi'

local C_lib = ffi.load('C/lib/libcityscapes-c.so')

ffi.cdef [[
void updateConfusionMatrix(
    long *confMatrix, long *predictedLabels,
    long *labels, int numPixels,
    int nClasses);
]]

function cityscapes.updateConfusionMatrix(confMatrix, predictedLabels, labels)
    -- confMatrix:      long, 19x19
    -- predictedLabels: long, 128x256
    -- labels:          byte, 128x256
    assert(confMatrix:type() == 'torch.LongTensor')
    assert(confMatrix:size(1) == confMatrix:size(2) and confMatrix:size(1) == dataset.nClasses)

    assert(predictedLabels:type() == 'torch.LongTensor')
    assert(labels:type() == 'torch.LongTensor')
    assert(predictedLabels:nElement() == labels:nElement())
    assert(predictedLabels:isContiguous() and labels:isContiguous())

    C_lib.updateConfusionMatrix(
        torch.data(confMatrix), torch.data(predictedLabels),
        torch.data(labels), predictedLabels:nElement(),
        cityscapes.nClasses)
end

local function validAverage(t)
    local sum, count = 0, 0
    t:apply(
        function(x)
            if x == x then -- if x is not nan
                sum = sum + x
                count = count + 1
            end
        end
    )

    return count > 0 and (sum / count) or 0
end

function cityscapes.calcIoU(confMatrix)
	-- returns: mean IoU, pixel acc, IoU by class
    local IoUclass = torch.FloatTensor(cityscapes.nClasses)
    local pixelAcc = 0

    for classIdx = 1,IoUclass:nElement() do
        local TP = confMatrix[{classIdx, classIdx}]
        local FN = confMatrix[{classIdx, {}}]:sum() - TP
        local FP = confMatrix[{{}, classIdx}]:sum() - TP
        
        IoUclass[classIdx] = TP / (TP + FP + FN)
        pixelAcc = pixelAcc + TP
    end

    pixelAcc = pixelAcc / confMatrix:sum()

    return validAverage(IoUclass), pixelAcc, IoUclass
end

return cityscapes
