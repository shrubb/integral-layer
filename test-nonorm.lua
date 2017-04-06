torch.setdefaulttensortype('torch.FloatTensor')

require 'Integral-cuda-multi'

int = Integral(2, 10, 10):cuda()

int.xMin[1], int.xMax[1] = 0,0
int.yMin[1], int.yMax[1] = -1,1

int.xMin[2], int.xMax[2] = -3,3
int.yMin[2], int.yMax[2] = -5,-5

input = torch.ones(2, 10, 10):cuda()
input[2]:mul(-2)

print(int:forward(input))

print('\nOther Integal:\n')

require 'IntegralSmartNorm'

int = IntegralSmartNorm(2, 10, 10):cuda()

int.xMin[1], int.xMax[1] = 0,0
int.yMin[1], int.yMax[1] = -1,1

int.xMin[2], int.xMax[2] = -3,3
int.yMin[2], int.yMax[2] = -5,-5

input = torch.ones(2, 10, 10):cuda()
input[2]:mul(-2)

print(int:forward(input))