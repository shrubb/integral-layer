torch.setdefaulttensortype('torch.FloatTensor')

require 'IntegralNoNorm'

int = IntegralNoNorm(2, 10, 10)

int.xMin[1], int.xMax[1] = 0,0
int.yMin[1], int.yMax[1] = -1,1

int.xMin[2], int.xMax[2] = -3,3
int.yMin[2], int.yMax[2] = -5,-5

input = torch.ones(2, 10, 10)
input[2]:mul(-1)

print(int:forward(input))