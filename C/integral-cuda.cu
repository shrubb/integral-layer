#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>

struct Matrix {
    int width;
    int height;
    int stride;
    float* elements;
};

#define BLOCK_SIZE 32

__device__ float getElement(const Matrix M, int row, int col) {
    return (row < M.height and col < M.width ? M.elements[row * M.stride + col] : 0);
}

__device__ void setElement(Matrix M, int row, int col, float value) {
    M.elements[row * M.stride + col] = value;
}

__device__ Matrix getSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void TransposeKernel(Matrix A, Matrix B) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];

    Matrix matFrom = getSubMatrix(A, blockIdx.x, blockIdx.y);
    Matrix matTo   = getSubMatrix(B, blockIdx.y, blockIdx.x);

    // transpose tile on-the-fly
    tile[threadIdx.x][threadIdx.y] = getElement(matFrom, threadIdx.x, threadIdx.y);

    __syncthreads();

    if (BLOCK_SIZE * blockIdx.x + threadIdx.x < A.height and BLOCK_SIZE * blockIdx.y + threadIdx.y < A.width) {
        setElement(matTo, threadIdx.y, threadIdx.x, tile[threadIdx.x][threadIdx.y]);
    }
}

__host__ void Transpose(const Matrix A, Matrix B) {
    assert(A.width == B.height and A.height == B.width);

    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    cudaMalloc(&d_B.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((A.height + dimBlock.x - 1) / dimBlock.x, (A.width + dimBlock.y - 1) / dimBlock.y);
    TransposeKernel <<<dimGrid, dimBlock>>> (d_A, d_B);

    cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
}

int main() {
    // Код для проверки
    const int N = 1024, M = 1024;

    static float A[N][M], B[M][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5);
        }
    }

    Matrix Am; Am.elements = A[0]; Am.height = N; Am.stride = Am.width = M;
    Matrix Bm; Bm.elements = B[0]; Bm.height = M; Bm.stride = Bm.width = N;

    Transpose(Am, Bm);

    const float EPS = 0.00001;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(A[j][i] - B[i][j]) > EPS) {
                std::cout << A[j][i] << " " << B[i][j] << "; " << i << ", " << j << std::endl;
                return 0;
            }
        }
    }

    return 0;
}

using std::max;
using std::min;

extern "C" {

void forward(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr, float areaCoeff) {

    int t, b, l, r;

    for (int x = 0; x < h; ++x) {
        for (int y = 0; y < w; ++y) {

            t = max(0, min(x+xMinCurr, h) );
            b = max(0, min(x+xMaxCurr, h) );
            l = max(0, min(y+yMinCurr, w) );
            r = max(0, min(y+yMaxCurr, w) );

            outData[x*w + y] = areaCoeff *
                ( intData[b*(w+1) + r]
                - intData[t*(w+1) + r]
                - intData[b*(w+1) + l]
                + intData[t*(w+1) + l]);
        }
    }
}

void backward(
    float *intData, float *gradOutData, int h, int w, float *deltas,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    float & xMaxDelta = deltas[1];
    float & xMinDelta = deltas[0];
    float & yMaxDelta = deltas[3];
    float & yMinDelta = deltas[2];

    for (int x = 1; x <= h; ++x) {
        for (int y = 1; y <= w; ++y) {
            
            int tClip = max(x+xMinCurr, 0);
            int bClip = min(x+xMaxCurr, h);
            int lClip = max(y+yMinCurr, 0);
            int rClip = min(y+yMaxCurr, w);

            xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,min(x+xMaxCurr+1,h))*(w+1) + max(0,rClip)]
                - intData[max(0,bClip)*(w+1) + max(0,rClip)]
                - intData[max(0,min(x+xMaxCurr+1,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                + intData[max(0,bClip)*(w+1)
                    + max(0,min(y+yMinCurr-1,w))] );
            
            xMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,min(x+xMinCurr-1,h))*(w+1) 
                    + max(0,rClip)]
                - intData[min(h,tClip)*(w+1)
                    + max(0,rClip)]
                - intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                + intData[min(h,tClip)*(w+1)
                    + max(0,min(y+yMinCurr-1,w))] );
            
            yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,bClip)*(w+1) 
                    + max(0,min(y+yMaxCurr+1,w))]
                - intData[max(0,bClip)*(w+1)
                    + max(0,rClip)]
                - intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMaxCurr+1,w))]
                + intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,rClip)] );
            
            yMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,bClip)*(w+1) 
                    + max(0,min(y+yMinCurr-1,w))]
                - intData[max(0,bClip)*(w+1)
                    + min(w,lClip)]
                - intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                + intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + min(w,lClip)] );
        }
    }
}

} // extern "C"
