#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cmath>

#define BLOCK_SIZE 32
#define BLOCK_CHANNELS (1024 / (BLOCK_SIZE * BLOCK_SIZE))

using std::max;
using std::min;
using std::floor;
using std::ceil;

__global__ void forwardKernel(
    float *intData, float *outData, int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax, float *areaCoeff) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    if (x < h and y < w and z < nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at indices 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        int t = max(0, min(x+(int)ceil (xMin[z]  ), h) );
        int b = max(0, min(x+(int)floor(xMax[z]+1), h) );
        int l = max(0, min(y+(int)ceil (yMin[z]  ), w) );
        int r = max(0, min(y+(int)floor(yMax[z]+1), w) );

        outData[z*w*h + x*w + y] = areaCoeff[z] *
            ( intData[b*(w+1) + r]
            - intData[t*(w+1) + r]
            - intData[b*(w+1) + l]
            + intData[t*(w+1) + l]);
    }
}

__global__ void forwardNoNormKernel(
    float *intData, float *outData, int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    if (x < h and y < w and z < nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        // xMinCurr = (int)ceil(xMin[z])
        int t = max(0, min(x+(int)ceil(xMin[z]), h-1) );
        // xMaxCurr = (int)floor(xMax[z])+1
        int b = max(1, min(x+(int)floor(xMax[z])+1, h)   );
        // yMinCurr = (int)ceil(yMin[z])
        int l = max(0, min(y+(int)ceil(yMin[z]), w-1) );
        // yMaxCurr = (int)floor(yMax[z])+1
        int r = max(1, min(y+(int)floor(yMax[z])+1, w)   );

        outData[z*w*h + x*w + y] = 
            ( intData[b*(w+1) + r]
            - intData[t*(w+1) + r]
            - intData[b*(w+1) + l]
            + intData[t*(w+1) + l]);
    }
}

__global__ void forwardNoNormFracKernel(
    float *intData, float *outData, int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideChannel, int inDataStrideRow) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    if (x < h and y < w and z < nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        int   xMinCurr = (int)ceil(xMin[z]);
        float xMinCurrFrac = (float)xMinCurr - xMin[z];
        int   yMinCurr = (int)ceil(yMin[z]);
        float yMinCurrFrac = (float)yMinCurr - yMin[z];

        int   xMaxCurr = (int)floor(xMax[z]);
        float xMaxCurrFrac = xMax[z] - (float)xMaxCurr;
        ++xMaxCurr;
        int   yMaxCurr = (int)floor(yMax[z]);
        float yMaxCurrFrac = yMax[z] - (float)yMaxCurr;
        ++yMaxCurr;

        int t = max(0, min(x+xMinCurr, h-1) );
        int b = max(1, min(x+xMaxCurr, h)   );
        int l = max(0, min(y+yMinCurr, w-1) );
        int r = max(1, min(y+yMaxCurr, w)   );

        outData[z*w*h + x*w + y] = 
            ( intData[b*(w+1) + r]
            - intData[t*(w+1) + r]
            - intData[b*(w+1) + l]
            + intData[t*(w+1) + l])

        // -- xMax border
        +(intData[max(0,min(x+xMaxCurr+1,h))*(w+1) 
            + max(0,min(y+yMaxCurr,w))]
        - intData[max(0,min(x+xMaxCurr,h))*(w+1)
            + max(0,min(y+yMaxCurr,w))]
        - intData[max(0,min(x+xMaxCurr+1,h))*(w+1)
            + max(0,min(y+yMinCurr,w))]
        + intData[max(0,min(x+xMaxCurr,h))*(w+1)
            + max(0,min(y+yMinCurr,w))]
        ) * xMaxCurrFrac

        // -- yMax border
        +(intData[max(0,min(x+xMaxCurr,h))*(w+1) 
            + max(0,min(y+yMaxCurr+1,w))]
        - intData[max(0,min(x+xMaxCurr,h))*(w+1)
            + max(0,min(y+yMaxCurr,w))]
        - intData[max(0,min(x+xMinCurr,h))*(w+1)
            + max(0,min(y+yMaxCurr+1,w))]
        + intData[max(0,min(x+xMinCurr,h))*(w+1)
            + max(0,min(y+yMaxCurr,w))]
        ) * yMaxCurrFrac

        // -- xMin border
        +(intData[max(0,min(x+xMinCurr,h))*(w+1) 
            + max(0,min(y+yMaxCurr,w))]
        - intData[max(0,min(x+xMinCurr-1,h))*(w+1)
            + max(0,min(y+yMaxCurr,w))]
        - intData[max(0,min(x+xMinCurr,h))*(w+1)
            + max(0,min(y+yMinCurr,w))]
        + intData[max(0,min(x+xMinCurr-1,h))*(w+1)
            + max(0,min(y+yMinCurr,w))]
        ) * xMinCurrFrac

        // -- yMin border
        +(intData[max(0,min(x+xMaxCurr,h))*(w+1) 
            + max(0,min(y+yMinCurr,w))]
        - intData[max(0,min(x+xMaxCurr,h))*(w+1)
            + max(0,min(y+yMinCurr-1,w))]
        - intData[max(0,min(x+xMinCurr,h))*(w+1)
            + max(0,min(y+yMinCurr,w))]
        + intData[max(0,min(x+xMinCurr,h))*(w+1)
            + max(0,min(y+yMinCurr-1,w))]
        ) * yMinCurrFrac

        // -- corner pixels
        + xMaxCurrFrac*yMaxCurrFrac * (
               (x+xMaxCurr > h-1 or
                y+yMaxCurr > w-1 or
                x+xMaxCurr < 0   or
                y+yMaxCurr < 0) ? 0 :
               inData[z*inDataStrideChannel + (x+xMaxCurr)*inDataStrideRow + (y+yMaxCurr)])

        + xMinCurrFrac*yMaxCurrFrac * (
               (x+xMinCurr-1 > h-1 or
                y+yMaxCurr   > w-1 or
                x+xMinCurr-1 < 0   or
                y+yMaxCurr   < 0) ? 0 :
               inData[z*inDataStrideChannel + (x+xMinCurr-1)*inDataStrideRow + (y+yMaxCurr)])

        + xMaxCurrFrac*yMinCurrFrac * (
               (x+xMaxCurr   > h-1 or
                y+yMinCurr-1 > w-1 or
                x+xMaxCurr   < 0   or
                y+yMinCurr-1 < 0) ? 0 :
               inData[z*inDataStrideChannel + (x+xMaxCurr)*inDataStrideRow + (y+yMinCurr-1)])

        + xMinCurrFrac*yMinCurrFrac * (
               (x+xMinCurr-1 > h-1 or
                y+yMinCurr-1 > w-1 or
                x+xMinCurr-1 < 0   or
                y+yMinCurr-1 < 0) ? 0 :
               inData[z*inDataStrideChannel + (x+xMinCurr-1)*inDataStrideRow + (y+yMinCurr-1)]);
    }
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator  T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

inline
bool isPow2(unsigned int x) {
    return ((x&(x-1))==0);
}

// From NVIDIA CUDA samples
template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(float *g_idata, float *g_odata, unsigned int n)
{
    float *sdata = SharedMemory<float>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];
        // printf("%d reading %f from position %d\n", tid, g_idata[i], i);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            mySum += g_idata[i+blockSize];
            // printf("%d reading additional %f from position %d\n", tid, g_idata[i+blockSize], i+blockSize);
        }

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

void reduceKernelLauncher(int size, int threads, int blocks, float *d_idata, float *d_odata) {

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduce6<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce6<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce6<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce6< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce6< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce6< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduce6<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduce6<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduce6<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduce6<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduce6<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce6<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce6<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce6< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce6< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce6< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduce6<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduce6<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduce6<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduce6<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
}

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);  

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    blocks = min(maxBlocks, blocks);
}

float reduce(float *input, float *output, int n) {

    int maxThreads = min(256, nextPow2(n) / 4);
    int maxBlocks = 64;

    int threads = 0, blocks = 0;
    getNumBlocksAndThreads(n, maxBlocks, maxThreads, blocks, threads);

    // std::cout << blocks << " blocks, " << threads << " threads" << std::endl;

    float gpu_result = 0;

    // execute the kernel
    reduceKernelLauncher(n, threads, blocks, input, output);

    // Clear input for later use as temporary buffer.
    cudaMemset(input, 0, n*sizeof(float));

    // sum partial block sums on GPU
    int s = blocks;

    while (s > 1)
    {
        threads = 0;
        blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
        cudaMemcpy(input, output, s*sizeof(float), cudaMemcpyDeviceToDevice);
        reduceKernelLauncher(s, threads, blocks, input, output);

        s = (s + (threads*2-1)) / (threads*2);
    }

    cudaMemcpy(&gpu_result, output, sizeof(float), cudaMemcpyDeviceToHost);

    return gpu_result;
}

extern "C" {

void forwardCuda(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax, float *areaCoeff) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid((h + dimBlock.x - 1) / dimBlock.x, (w + dimBlock.y - 1) / dimBlock.y, (nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardKernel <<<dimGrid, dimBlock>>> (intData, outData, h, w, nWindows, xMin, xMax, yMin, yMax, areaCoeff);
}

void forwardCudaNoNorm(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y, 
        (nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardNoNormKernel <<<dimGrid, dimBlock>>> (intData, outData, h, w, nWindows, xMin, xMax, yMin, yMax);
}

void forwardCudaNoNormFrac(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStride) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y, 
        (nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardNoNormFracKernel <<<dimGrid, dimBlock>>> (
        intData, outData, h, w, nWindows, 
        xMin, xMax, yMin, yMax, inData,
        inDataStrideChannel, inDataStrideRow);
}

// TODO templates for frac/non-frac

__global__ void xMaxDeltaIntegral(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr  , h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMaxCurr  , h))*(w+1)
                + max(0,min(y+yMinCurr, w))]
            + intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + max(0,min(y+yMinCurr, w))]);
    }
}

__global__ void xMinDeltaIntegral(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        // TODO optimize
        int tClip = max(x+xMinCurr, 0);
        int bClip = min(x+xMaxCurr, h);
        int lClip = max(y+yMinCurr, 0);
        int rClip = min(y+yMaxCurr, w);

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMinCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMinCurr-1, h))*(w+1)
                + max(0,min(y+yMinCurr  , w))]
            + intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                + max(0,min(y+yMinCurr  , w))]);
    }
}

__global__ void yMaxDeltaIntegral(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            + intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]);
    }
}

__global__ void yMinDeltaIntegral(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMinCurr-1,w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + min(w,max(y+yMinCurr, 0))]
            - intData[max(0,min(x+xMinCurr  , h))*(w+1)
                + max(0,min(y+yMinCurr-1,w))]
            + intData[max(0,min(x+xMinCurr  , h))*(w+1)
                + min(w,max(y+yMinCurr, 0))]);
    }
}

__global__ void elementWiseProduct(float *tmpArray, float *gradOutData, int len) {

    int idx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (idx < len) {
        tmpArray[idx] *= gradOutData[idx];
    }
}

void backwardCudaSingle(
    float *intData, float *gradOutData, float *tmpArray, float *tmpArraySum, int h, int w, 
    float *deltas, int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h + dimBlock.x - 1) / dimBlock.x, (w + dimBlock.y - 1) / dimBlock.y);

    dim3 dimBlock1D(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid1D((h*w + dimBlock1D.x - 1) / dimBlock1D.x);

    // xMaxDelta
    xMaxDeltaIntegral <<<dimGrid, dimBlock>>> (intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr);
    elementWiseProduct <<<dimGrid1D, dimBlock1D>>> (tmpArray, gradOutData, h*w);
    deltas[1] = reduce(tmpArray, tmpArraySum, h*w);

    // xMinDelta
    xMinDeltaIntegral <<<dimGrid, dimBlock>>> (intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr);
    elementWiseProduct <<<dimGrid1D, dimBlock1D>>> (tmpArray, gradOutData, h*w);
    deltas[0] = reduce(tmpArray, tmpArraySum, h*w);

    // yMaxDelta
    yMaxDeltaIntegral <<<dimGrid, dimBlock>>> (intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr);
    elementWiseProduct <<<dimGrid1D, dimBlock1D>>> (tmpArray, gradOutData, h*w);
    deltas[3] = reduce(tmpArray, tmpArraySum, h*w);

    // yMinDelta
    yMinDeltaIntegral <<<dimGrid, dimBlock>>> (intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr);
    elementWiseProduct <<<dimGrid1D, dimBlock1D>>> (tmpArray, gradOutData, h*w);
    deltas[2] = reduce(tmpArray, tmpArraySum, h*w);

#if 0
    // REDUCTION TESTER

    const int len = 1025;
    float hostV[len];
    std::cout << "Original array:" << std::endl;
    float sum = 0;
    for (int i = 0; i < len; ++i) {
        hostV[i] = (float)rand()/(float)RAND_MAX;
        // std::cout << hostV[i] << " ";
        sum += hostV[i];
    }
    std::cout << std::endl;
    std::cout << "Sum = " << sum << std::endl;

    float *deviceV, *deviceO;
    cudaMalloc(&deviceV, len * sizeof(float));
    cudaMalloc(&deviceO, len * sizeof(float));

    cudaMemcpy(deviceV, hostV, len * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "GPU result: " << reduce(deviceV, deviceO, len) << std::endl;
    
    std::cout << std::endl;
    cudaMemcpy(hostV, deviceO, len * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "output:" << std::endl;
    for (int i = 0 ; i < len; ++i) {
        // std::cout << hostV[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    cudaFree(deviceV);
    cudaFree(deviceO);
#endif
}

__global__ void xMaxDeltaIntegralFrac(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        float blCorner = (x+xMaxCurr > h or y+yMinCurr > w or
                          x+xMaxCurr < 1 or y+yMinCurr < 1) ? 0 :
                          inData[(x+xMaxCurr-1)*inDataStride + (y+yMinCurr-1)];
        float brCorner = (x+xMaxCurr > h or y+yMaxCurr > w or
                          x+xMaxCurr < 1 or y+yMaxCurr < 1) ? 0 :
                          inData[(x+xMaxCurr-1)*inDataStride + (y+yMaxCurr-1)];

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr  , h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMaxCurr  , h))*(w+1)
                + max(0,min(y+yMinCurr, w))]
            + intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + max(0,min(y+yMinCurr, w))]
            + brCorner * yMaxCurrFrac
            + blCorner * yMinCurrFrac);
    }
}

__global__ void xMinDeltaIntegralFrac(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        float tlCorner = (x+xMinCurr > h or y+yMinCurr > w or 
                          x+xMinCurr < 1 or y+yMinCurr < 1) ? 0 :
                          inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)];
        float trCorner = (x+xMinCurr > h or y+yMaxCurr > w or
                          x+xMinCurr < 1 or y+yMaxCurr < 1) ? 0 :
                          inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr-1)];

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMinCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMinCurr-1, h))*(w+1)
                + max(0,min(y+yMinCurr  , w))]
            + intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                + max(0,min(y+yMinCurr  , w))]
            + trCorner * yMaxCurrFrac
            + tlCorner * yMinCurrFrac);
    }
}

__global__ void yMaxDeltaIntegralFrac(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        float trCorner = (x+xMinCurr > h or y+yMaxCurr > w or
                          x+xMinCurr < 1 or y+yMaxCurr < 1) ? 0 :
                          inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr-1)];
        float brCorner = (x+xMaxCurr > h or y+yMaxCurr > w or
                          x+xMaxCurr < 1 or y+yMaxCurr < 1) ? 0 :
                          inData[(x+xMaxCurr-1)*inDataStride + (y+yMaxCurr-1)];

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            + intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            + trCorner * xMinCurrFrac
            + brCorner * xMaxCurrFrac);
    }
}

__global__ void yMinDeltaIntegralFrac(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        float tlCorner = (x+xMinCurr > h or y+yMinCurr > w or 
                          x+xMinCurr < 1 or y+yMinCurr < 1) ? 0 :
                          inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)];
        float blCorner = (x+xMaxCurr > h or y+yMinCurr > w or
                          x+xMaxCurr < 1 or y+yMinCurr < 1) ? 0 :
                          inData[(x+xMaxCurr-1)*inDataStride + (y+yMinCurr-1)];

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMinCurr-1,w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + min(w,max(y+yMinCurr, 0))]
            - intData[max(0,min(x+xMinCurr  , h))*(w+1)
                + max(0,min(y+yMinCurr-1,w))]
            + intData[max(0,min(x+xMinCurr  , h))*(w+1)
                + min(w,max(y+yMinCurr, 0))]
            + tlCorner * xMinCurrFrac
            + blCorner * xMaxCurrFrac);
    }
}

// TODO need general kernels instead of `single`
void backwardCudaSingleFrac(
    float *intData, float *gradOutData, float *tmpArray, float *tmpArraySum, int h, int w, 
    float *deltas, int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h + dimBlock.x - 1) / dimBlock.x, (w + dimBlock.y - 1) / dimBlock.y);

    dim3 dimBlock1D(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid1D((h*w + dimBlock1D.x - 1) / dimBlock1D.x);

    // xMaxDelta
    xMaxDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
        intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
        yMinCurrFrac, yMaxCurrFrac, inData, inDataStride);
    elementWiseProduct <<<dimGrid1D, dimBlock1D>>> (tmpArray, gradOutData, h*w);
    deltas[1] = reduce(tmpArray, tmpArraySum, h*w);

    // xMinDelta
    xMinDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
        intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
        yMinCurrFrac, yMaxCurrFrac, inData, inDataStride);
    elementWiseProduct <<<dimGrid1D, dimBlock1D>>> (tmpArray, gradOutData, h*w);
    deltas[0] = reduce(tmpArray, tmpArraySum, h*w);

    // yMaxDelta
    yMaxDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
        intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
        xMinCurrFrac, xMaxCurrFrac, inData, inDataStride);
    elementWiseProduct <<<dimGrid1D, dimBlock1D>>> (tmpArray, gradOutData, h*w);
    deltas[3] = reduce(tmpArray, tmpArraySum, h*w);

    // yMinDelta
    yMinDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
        intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
        xMinCurrFrac, xMaxCurrFrac, inData, inDataStride);
    elementWiseProduct <<<dimGrid1D, dimBlock1D>>> (tmpArray, gradOutData, h*w);
    deltas[2] = reduce(tmpArray, tmpArraySum, h*w);

} // extern "C"
