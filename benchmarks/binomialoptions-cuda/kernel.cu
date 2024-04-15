/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

#include "binomialOptions.h"
#include "realtype.h"
#include "approx_debug.h"
#include "approx.h"

// Overloaded shortcut functions for different precision modes
#ifndef DOUBLE_PRECISION
__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
  float d = S * __expf(vDt * (2.0f * i - NUM_STEPS)) - X;
  return (d > 0.0F) ? d : 0.0F;
}
#else
__device__ inline double expiryCallValue(double S, double X, double vDt, int i)
{
  double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
  return (d > 0.0) ? d : 0.0;
}
#endif


// GPU kernel
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif

__global__ void binomialOptionsKernel(const real *_S, const real *_X, const real *_vDt, const real *_puByDf, const real *_pdByDf, real *callValue)
{
  __shared__ real call_exchange[THREADBLOCK_SIZE + 1];

  const int     tid = threadIdx.x;
  const real      S = _S[blockIdx.x];
  const real      X = _X[blockIdx.x];
  const real    vDt = _vDt[blockIdx.x];
  const real puByDf = _pdByDf[blockIdx.x];
  const real pdByDf = _puByDf[blockIdx.x];

  real call[ELEMS_PER_THREAD + 1];
#pragma unroll
  for(int i = 0; i < ELEMS_PER_THREAD; ++i)
    call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

  if (tid == 0)
    call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

  int final_it = max(0, tid * ELEMS_PER_THREAD - 1);

#pragma unroll 16
  for(int i = NUM_STEPS; i > 0; --i)
  {
    call_exchange[tid] = call[0];
    __syncthreads();
    call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
    __syncthreads();

    if (i > final_it)
    {
#pragma unroll
      for(int j = 0; j < ELEMS_PER_THREAD; ++j)
        call[j] = puByDf * call[j + 1] + pdByDf * call[j];
    }
  }

  if (tid == 0)
  {
    callValue[blockIdx.x] = call[0];
  }
}

__global__ void preProcessKernel(
    real *d_T, real *d_R, real *d_V, 
    real *d_puByDf, real *d_pdByDf, real *d_vDt, 
    int optN) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < optN) {
        const real T = d_T[i];
        const real R = d_R[i];
        const real V = d_V[i];

        const real dt = T / (real)NUM_STEPS;
        const real vDt = V * sqrt(dt);
        const real rDt = R * dt;
        const real If = exp(rDt);
        const real Df = exp(-rDt);
        const real u = exp(vDt);
        const real d = exp(-vDt);
        const real pu = (If - d) / (u - d);
        const real pd = (real)1.0 - pu;
        const real puByDf = pu * Df;
        const real pdByDf = pd * Df;

        d_puByDf[i] = puByDf;
        d_pdByDf[i] = pdByDf;
        d_vDt[i] = vDt;
    }
}

// Host-side interface to GPU binomialOptions
extern "C" void binomialOptionsGPU(
    real *callValue,
    real *_S, real *_X, real *_R, real *_V, real *_T,
    int optN,
    int numIterations, 
    int multiplier
    )
{
    real *d_T, *d_R, *d_V;
    real *d_puByDf, *d_pdByDf, *d_vDt;

    approx::util::warmup();
    cudaMalloc((void**)&d_T, sizeof(real) * optN);
    cudaMalloc((void**)&d_R, sizeof(real) * optN);
    cudaMalloc((void**)&d_V, sizeof(real) * optN);
    cudaMalloc((void**)&d_puByDf, sizeof(real) * optN);
    cudaMalloc((void**)&d_pdByDf, sizeof(real) * optN);
    cudaMalloc((void**)&d_vDt, sizeof(real) * optN);


    real *d_S;
    cudaMalloc ((void**)&d_S, sizeof(real) * optN);
    real *d_X;
    cudaMalloc ((void**)&d_X, sizeof(real) * optN);
    real *d_CallValue;
    cudaMalloc ((void**)&d_CallValue, sizeof(real) * optN);


  auto start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < numIterations; iter++) {
    EventRecorder::CPUEvent Trial{"Trial"};
    Trial.recordStart();
    EventRecorder::GPUEvent GPUPreprocess{"GPU Preprocess"};
    GPUPreprocess.recordStart();
    cudaMemcpy(d_T, _T, sizeof(real) * optN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, _R, sizeof(real) * optN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, _V, sizeof(real) * optN, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (optN + blockSize - 1) / blockSize;

    preProcessKernel<<<numBlocks, blockSize>>>(d_T, d_R, d_V, d_puByDf, d_pdByDf, d_vDt, optN);
    GPUPreprocess.recordEnd();
    EventRecorder::LogEvent(GPUPreprocess);


    EventRecorder::GPUEvent DtoH{"Device To Host"}, HtoD{"Host to Device"}, Kernel{"Kernel"};


    cudaDeviceSynchronize();
    HtoD.recordStart();
    cudaMemcpy(d_S, _S, optN *sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, _X, optN *sizeof(real), cudaMemcpyHostToDevice);
    HtoD.recordEnd();


    #ifdef APPROX
    #pragma approx declare tensor_functor(ipt_fn: [i, 0:5] = ([i*multiplier:i*multiplier+multiplier], [i*multiplier:i*multiplier+multiplier], [i*multiplier:i*multiplier+multiplier], [i*multiplier:i*multiplier+multiplier], [i*multiplier:i*multiplier+multiplier]))
    #pragma approx declare tensor_functor(opt_fn: [i, 0:1] = ([i*multiplier:i*multiplier+multiplier]))
    int ni = optN/multiplier;
    #pragma approx declare tensor(ipt_tens: ipt_fn(d_S[0:ni], d_X[0:ni], d_vDt[0:ni], d_puByDf[0:ni], d_pdByDf[0:ni]))

    #pragma approx ml(infer) in(ipt_tens) out(opt_fn(d_CallValue[0:ni])) label("BinomialOptionsKernel")
    #endif
    {
      Kernel.recordStart();
      binomialOptionsKernel<<<optN, THREADBLOCK_SIZE>>>(d_S, d_X, d_vDt, d_puByDf, d_pdByDf, d_CallValue);
      Kernel.recordEnd();
    }

    cudaDeviceSynchronize();
    DtoH.recordStart();
    cudaMemcpy(callValue, d_CallValue, optN *sizeof(real), cudaMemcpyDeviceToHost);
    DtoH.recordEnd();
    Trial.recordEnd();

    EventRecorder::LogEvent(HtoD);
    // Somehow, this breaks the code
    // Can't imagine why, but it does.
    // EventRecorder::LogEvent(Kernel);
    EventRecorder::LogEvent(DtoH);
    EventRecorder::LogEvent(Trial);
  }

  #if CAPTURE_OUTPUT
  int a = 0;
  int *b = &a;
  #pragma approx declare tensor_functor(ipt_fn: [i, 0:1] = ([i]))
  #pragma approx declare tensor_functor(opt_fn: [i, 0:1] = ([i]))
  #pragma approx declare tensor(dummy_ipt: ipt_fn(b[0:1]))
  #pragma approx declare tensor(output: opt_fn(callValue[0:optN]))

  #pragma approx ml(offline) in(dummy_ipt) out(opt_fn(d_CallValue[0:optN])) label("BinomialOptionsOutput")
  {

  }
  #endif
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (ms)\n", time * 1e-6f / numIterations);

  cudaFree(d_CallValue);
  cudaFree(d_S);
  cudaFree(d_X);
  cudaFree(d_vDt);
  cudaFree(d_puByDf);
  cudaFree(d_pdByDf);
  cudaFree(d_T);
  cudaFree(d_R);
  cudaFree(d_V);
}
