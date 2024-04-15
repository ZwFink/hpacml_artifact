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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <approx.h>
#include <approx_debug.h>
#include <cassert>
#include <iostream>

#include "binomialOptions.h"
#include "realtype.h"

#define DOUBLE 0
#define FLOAT 1
#define INT 2
#define LONG 3

#define THREADBLOCK_SIZE 256
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)

#define NUM_TRIALS 2


static double CND(double d)
{
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}


void BlackScholesCall(
    real &callResult,
    const real S, const real X,
    const real T, const real R,
    const real V
)
{
    double sqrtT = sqrt(T);

    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;

    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);

    callResult   = (real)(S * CNDD1 - X * expRT * CNDD2);
}


void writeQualityFile(const char *fileName, void *ptr, int type, size_t numElements){
    FILE *fd = fopen(fileName, "wb");
    assert(fd && "Could Not Open File\n");
    fwrite(&numElements, sizeof(size_t), 1, fd);
    fwrite(&type, sizeof(int), 1, fd);
    if ( type == DOUBLE)
        fwrite(ptr, sizeof(double), numElements, fd);
    else if ( type == FLOAT)
        fwrite(ptr, sizeof(float), numElements, fd);
    else if ( type == INT)
        fwrite(ptr, sizeof(int), numElements, fd);
    else
        assert(0 && "Not supported data type to write\n");
    fclose(fd);
}
void readData(FILE *fd, double **data,  size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;
    double *ptr = (double*) malloc (sizeof(double)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;
    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == DOUBLE){
        fread(ptr, sizeof(double), elements, fd);
    }
    else if ( type == FLOAT){
        float *tmp = (float*) malloc (sizeof(float)*elements);
        fread(tmp, sizeof(float), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (double) tmp[i];
        }
        free (tmp);
    }
    else if( type == INT ){
        int *tmp = (int*) malloc (sizeof(int)*elements);
        fread(tmp, sizeof(int), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (double) tmp[i];
        }
        free(tmp);
    }
    return; 
}

void readData(FILE *fd, float **data,  size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;

    float *ptr = (float*) malloc (sizeof(float)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == FLOAT ){
        fread(ptr, sizeof(float), elements, fd);
    }
    else if ( type == DOUBLE){
        double *tmp = (double*) malloc (sizeof(double)*elements);
        fread(tmp, sizeof(double), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (float) tmp[i];
        }
        free (tmp);
    }
    else if ( type == INT ){
        int *tmp = (int*) malloc (sizeof(int) * elements);
        fread(tmp, sizeof(int), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (float) tmp[i];
        }
        free(tmp);
    }
    return; 
}

void readData(FILE *fd, int **data,   size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;

    int *ptr = (int*) malloc (sizeof(int)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == INT ){
        fread(ptr, sizeof(int), elements, fd);
    }
    else if ( type == DOUBLE){
        double *tmp = (double*) malloc (sizeof(double)*elements);
        fread(tmp, sizeof(double), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (int) tmp[i];
        }
        free (tmp);
    }
    else if( type == FLOAT ){
        float *tmp = (float*) malloc (sizeof(float)*elements);
        fread(tmp, sizeof(float), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (int) tmp[i];
        }
        free(tmp);
    }
    return; 
}

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////

real randData(real low, real high)
{
    real t = (real)rand() / (real)RAND_MAX;
    return ((real)1.0 - t) * low + t * high;
}

// Black-Scholes formula for binomial tree results validation
extern "C" void BlackScholesCall(
    real &callResult,
    TOptionData optionData
    );

// Process an array of OptN options on GPU
extern "C" void binomialOptionsGPU(
    real *callValue,
    real *_S, real *_X, real *_R, real *_V, real *_T,
    int optN,
    int numIterations,
    int multiplier
    );


int main(int argc, char **argv)
{
  printf("[%s] - Starting...\n", argv[0]);

  FILE *file;

  if(!(argc == 4 || argc == 6))
    {
      std::cout << "USAGE: " << argv[0] << " input_file begin_option end_option [output_file multiplier]";
      return EXIT_FAILURE;
    }

  char *inputFile = argv[1];
  int begin_option = std::atoi(argv[2]);
  int end_option = std::atoi(argv[3]);

    //Read input data from file
    file = fopen(inputFile, "rb");
    if(file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", inputFile);
        exit(1);
    }

  bool write_output = false;
  int multiplier = 1;
  std::string ofname;
  if(argc == 6)
    {
      write_output = true;
      ofname = argv[4];
      multiplier = atoi(argv[5]);
    }


  // sptprice
  real *S;
  // strike
  real *X;
  // time
  real *T;
  // rate
  real *R;
  // volatility
  real *V;
  int *otype;

  real
    sumDelta, sumRef, gpuTime, errorVal;

  printf("Reading input data...\n");
  size_t numOptions = 0;
  
  #define PAD 256
  #define LINESIZE 64
  readData(file,&otype, &numOptions);  
  readData(file,&S, &numOptions);  
  readData(file,&X, &numOptions);  
  readData(file,&R, &numOptions);  
  readData(file,&V, &numOptions);  
  readData(file,&T, &numOptions);  
  
  
  delete[] otype;
  std::fill(V, V+numOptions, 0.10f);
  
  // now we want to consider options in the range [begin_option, end_option)
  numOptions = end_option - begin_option;
  S = S + begin_option;
  X = X + begin_option;
  R = R + begin_option;
  V = V + begin_option;
  T = T + begin_option;

  real *callValue = new real[numOptions];
  real *callValueBS = new real[numOptions];

  for (int i = 0; i < numOptions; i++)
  {
    BlackScholesCall(callValueBS[i], S[i], X[i], T[i], R[i], V[i]);
  }


  approx::util::warmup();
  printf("Running GPU binomial tree...\n");

  auto start = std::chrono::high_resolution_clock::now();

  binomialOptionsGPU(callValue, S, X, R, V, T, numOptions, NUM_ITERATIONS, multiplier);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<real> elapsed_seconds = end - start;
  gpuTime = (real)elapsed_seconds.count();

  printf("Options count            : %i     \n", numOptions);
  printf("Time steps               : %i     \n", NUM_STEPS);
  printf("Total binomialOptionsGPU() time: %f msec\n", gpuTime * 1000);
  printf("Options per second       : %f     \n", numOptions / (gpuTime*NUM_ITERATIONS));

    sumDelta = 0;
  sumRef = 0;

  for (int i = 0; i < numOptions; i++)
  {
    sumDelta += fabs(callValueBS[i] - callValue[i]);
    sumRef += fabs(callValueBS[i]);
  }

  if (sumRef > 1E-5)
  {
    printf("L1 norm: %E\n", sumDelta / sumRef);
  }
  else
  {
    printf("Avg. diff: %E\n", (double)(sumDelta / (real)numOptions));
  }

  if (write_output)
  {
    writeQualityFile(ofname.c_str(), callValue, DOUBLE, numOptions);
  }

  exit(EXIT_SUCCESS);
}
