#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <iostream>
#include <cuda.h>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <iterator> 
#include <set>
#include <chrono>
#include "approx_debug.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.0f

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#include "video_sequence.h"
#include "kernel_find_index.h"
#include "kernel_likelihood.h"
#include "kernel_normalize_weights.h"
#include "kernel_sum.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define TRIALS 20

float roundFloat(float value);

typedef struct particlefilter_results {
  int num_nan;
  double offload_time;
  double average_kernel_time;
  double particle_x;
  double particle_y;
} particlefilter_results_t;

class ObjectOrigin {

protected:
  int x0 = 0;
  int y0 = 0;

public:
  ObjectOrigin(int IszY, int IszX){ 
    x0 = IszX / 2;
    y0 = IszY / 2;
  }

  int xOrigin()const  { return x0; }
  int yOrigin() const { return y0; }
  virtual void regenerate() { }
};

class UARObjectOrigin : public ObjectOrigin {
private:

  std::uniform_int_distribution<std::mt19937::result_type> dist;
  std::mt19937 rng;
public:
  UARObjectOrigin(int IszY, int IszX, int radius) : ObjectOrigin(IszY, IszX) { 
  /*get object centers*/
  std::random_device dev;
  rng = std::mt19937(42);

  dist = std::uniform_int_distribution<std::mt19937::result_type>((IszY / 2) - radius, (IszY / 2) + radius);
  x0 = dist(rng);
  y0 = dist(rng);
  }

  void regenerate() override {
    x0 = dist(rng);
    y0 = dist(rng);
  }

};

void runMockFixed(const int repeat)
{
  float* tensorGpu = nullptr;
  cudaError_t status = cudaMalloc((void**)&tensorGpu, sizeof(float) * 128*128*13000);
  float* tensorGpuOpt = nullptr;
  cudaError_t status2 = cudaMalloc((void**)&tensorGpuOpt, sizeof(float) * 2*13000);

  for(int obj = 0; obj < repeat; obj++) {
    std::cout << obj << std::endl;
    #pragma approx declare tensor_functor(fnctor: [i, j, k, 0:1] = ([i, j, k]))
    #pragma approx declare tensor_functor(opt_fnctor: [i, j, 0:1] = ([i, j]))
    #pragma approx declare tensor(tens: fnctor(tensorGpu[0:13000, 0:128, 0:128]))
    #pragma approx ml(offline) in(tens) out(opt_fnctor(tensorGpuOpt[0:13000, 0:2]))
    {
    }
  }

  cudaFree(tensorGpu);
  cudaFree(tensorGpuOpt);
  return;
}

class NonRepeatingUARObjectOrigin : public ObjectOrigin {
private:
    std::vector<std::pair<int, int>> coordinates;
    std::mt19937 rng;

    // Inner class to define the iterator
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<int, int>;
        using pointer = value_type*;
        using reference = value_type&;

        Iterator(pointer ptr) : m_ptr(ptr) {}

        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }
        Iterator& operator++() { m_ptr++; return *this; }
        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }
        friend bool operator==(const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!=(const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };

    private:
        pointer m_ptr;
    };

public:
    NonRepeatingUARObjectOrigin(int IszY, int IszX, size_t num_coords) : ObjectOrigin(IszY, IszX) {
        std::random_device dev;
        rng = std::mt19937(dev());

        std::uniform_int_distribution<int> distY(5, 120);
        std::uniform_int_distribution<int> distX(5, 120);

        std::set<std::pair<int, int>> uniqueCoords;

        while (uniqueCoords.size() < num_coords) {
            int y = distY(rng);
            int x = distX(rng);
            uniqueCoords.insert({y, x});
            if(uniqueCoords.size() % 1000 == 0)
                printf("Generated %lu coordinates\n", uniqueCoords.size());
        }

        coordinates.assign(uniqueCoords.begin(), uniqueCoords.end());
    }

    size_t size() {
        return coordinates.size();
    }

    Iterator begin() { return Iterator(&coordinates[0]); }
    Iterator end() { return Iterator(&coordinates[0] + coordinates.size()); }
    std::pair<int, int> operator[](size_t idx) { return coordinates[idx]; }
    std::vector<std::pair<int, int>> getCoordinates() { return coordinates; }
};


class NonRepeatingSquareObjectOrigin : public ObjectOrigin {
private:
    std::vector<std::pair<int, int>> coordinates;
    std::mt19937 rng;

    // Inner class to define the iterator
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<int, int>;
        using pointer = value_type*;
        using reference = value_type&;

        Iterator(pointer ptr) : m_ptr(ptr) {}

        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }
        Iterator& operator++() { m_ptr++; return *this; }
        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }
        friend bool operator==(const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!=(const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };

    private:
        pointer m_ptr;
    };

public:
    NonRepeatingSquareObjectOrigin(int IszY, int IszX, int radius) : ObjectOrigin(IszY, IszX) {
        std::random_device dev;
        rng = std::mt19937(42);

        for (int y = (IszY / 2) - radius; y <= (IszY / 2) + radius; ++y) {
            for (int x = (IszX / 2) - radius; x <= (IszX / 2) + radius; ++x) {
                coordinates.emplace_back(x, y);
            }
        }

        std::shuffle(coordinates.begin(), coordinates.end(), rng);
    }

    size_t size() {
        return coordinates.size();
    }

    Iterator begin() { return Iterator(&coordinates[0]); }
    Iterator end() { return Iterator(&coordinates[0] + coordinates.size()); }
    std::pair<int, int> operator[](size_t idx) { return coordinates[idx]; }
    std::vector<std::pair<int, int>> getCoordinates() { return coordinates; }
};


class ExhaustiveUARObjectOrigin : public ObjectOrigin {
private:

  std::uniform_int_distribution<std::mt19937::result_type> dist;
  std::mt19937 rng;
public:
  ExhaustiveUARObjectOrigin(int IszY, int IszX, int radius) : ObjectOrigin(IszY, IszX) { 
  /*get object centers*/
  std::random_device dev;
  rng = std::mt19937(42);

  dist = std::uniform_int_distribution<std::mt19937::result_type>((IszY / 2) - radius, (IszY / 2) + radius);
  x0 = dist(rng);
  y0 = dist(rng);
  }

  void regenerate() override {
    x0 = dist(rng);
    y0 = dist(rng);
  }

};

class BinaryFileObjectOrigin : public ObjectOrigin {
public:
  BinaryFileObjectOrigin(std::string fname, int entry, int IszY, int IszX) : ObjectOrigin(IszY, IszX) { 
    FILE *fp = fopen(fname.c_str(), "rb");
    if (fp == NULL) {
      printf("ERROR: Unable to open file %s\n", fname.c_str());
      exit(1);
    }
    fseek(fp, entry*sizeof(int), SEEK_SET);
    fread(&x0, sizeof(int), 1, fp);
    fread(&y0, sizeof(int), 1, fp);
    fclose(fp);
  }

};

class ConstantObjectOrigin : public ObjectOrigin {
public:
  ConstantObjectOrigin(int x, int y, int IszY, int IszX) : ObjectOrigin(IszY, IszX) { 
    x0 = x;
    y0 = y;
  }
};

enum class ObjectOriginType {
  CENTER,
  UAR,
  BINARY_FILE,
  CONSTANT
};

class ObjectOriginFactory {
public:
  static ObjectOrigin *create(ObjectOriginType type, std::string fname, int entry, int x0, int y0, int IszY, int IszX) {
    switch (type) {
      case ObjectOriginType::CENTER:
        return new ObjectOrigin(IszY, IszX);
      case ObjectOriginType::UAR:
        // return new UARObjectOrigin(IszY, IszX);
      case ObjectOriginType::BINARY_FILE:
        return new BinaryFileObjectOrigin(fname, entry, IszY, IszX);
      case ObjectOriginType::CONSTANT:
        return new ConstantObjectOrigin(x0, y0, IszY, IszX);
      default:
        return nullptr;
    }
  }
};

ObjectOrigin *OO = nullptr;

/*****************************
 * Returns a long int representing the time
 *****************************/
 using timepoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
timepoint_t get_time() {
  return std::chrono::high_resolution_clock::now();;
}

/* Returns the number of microseconds elapsed between the two specified times */
double elapsed_time(timepoint_t start_time, timepoint_t end_time) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
}

/**
 * Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
 * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a uniformly distributed number [0, 1)
 */
float randu(int * seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index] / ((float) M));
}

/**
 * Generates a normally distributed random number using the Box-Muller transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a float representing random number generated using the Box-Muller algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
 */
float randn(int * seed, int index) {
  /*Box-Muller algorithm*/
  float u = randu(seed, index);
  float v = randu(seed, index);
  float cosine = cos(2 * PI * v);
  float rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

/**
 * Takes in a float and returns an integer that approximates to that float
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
float roundFloat(float value) {
  int newValue = (int) (value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
template<typename VideoSequence>
void setIf(int testValue, int newValue, VideoSequence &Vid) {
  int x, y, z;
  int dimX = Vid.xdim();
  int dimY = Vid.ydim();
  int dimZ = Vid.zdim();

  for (x = 0; x < dimX; x++) {
    for (y = 0; y < dimY; y++) {
      for (z = 0; z < dimZ; z++) {
        if(Vid(x, y, z) == testValue)
          Vid(x, y, z) = newValue;
      }
    }
  }
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
template<typename VideoSequence>
void addNoise(VideoSequence &Vid) {
  int x, y, z;
  int dimX = Vid.xdim();
  int dimY = Vid.ydim();
  int dimZ = Vid.zdim();
  int *seed = Vid.seed_array();
  for (x = 0; x < dimX; x++) {
    for (y = 0; y < dimY; y++) {
      for (z = 0; z < dimZ; z++) {
        Vid(x, y, z) = Vid(x, y, z) + (unsigned char) (5 * randn(seed, 0));
      }
    }
  }
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int * disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      float distance = sqrt(pow((float) (x - radius + 1), 2) + pow((float) (y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
      else
        disk[x * diameter + y] = 0;
    }
  }
}

/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
template <typename VideoSequence>
void dilate_matrix(VideoSequence &matrix, int posX, int posY, int posZ, int error) {
  int dimX = matrix.xdim();
  int dimY = matrix.ydim();
  int dimZ = matrix.zdim();
  int startX = posX - error;
  while (startX < 0)
    startX++;
  int startY = posY - error;
  while (startY < 0)
    startY++;
  int endX = posX + error;
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      float distance = sqrt(pow((float) (x - posX), 2) + pow((float) (y - posY), 2));
      if (distance < error)
        matrix(x, y, posZ) = 1;
    }
  }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
  template <typename VideoSequence>
  void imdilate_disk(VideoSequence &matrix, int error, VideoSequence &newMatrix) {
  int x, y, z;
  int dimX = matrix.xdim();
  int dimY = matrix.ydim();
  int dimZ = matrix.zdim();
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if(matrix(x, y, z) == 1)
          dilate_matrix(newMatrix, x, y, z, error);
        }
      }
    }
  }

/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void getneighbors(int * se, int numOnes, int * neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int) (y - center);
        neighbors[neighY * 2 + 1] = (int) (x - center);
        neighY++;
      }
    }
  }
}

void videoSequenceGivenFrames(VideoSequence<FrameOuterAccessPolicy> &I, int IszX, int IszY, std::vector<std::pair<int, int>> Centers, int *seed) {
  using VideoSequence = VideoSequence<FrameOuterAccessPolicy>;
  int frame = 0;
  auto Nfr = Centers.size();
  int *centerX = new int[Nfr];
  int *centerY = new int[Nfr];
  for(auto [x0, y0] : Centers) {
    I(x0, y0, frame) = 1;
    centerX[frame] = x0;
    centerY[frame] = y0;
      ++frame;
  }

      /*dilate matrix*/
  VideoSequence newMatrix(IszX, IszY, Nfr, seed);
  imdilate_disk(I, 5, newMatrix);
  int x, y, z;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (z = 0; z < Nfr; z++) {
        I(x, y, z) = newMatrix(x, y, z);
      }
    }
  }
  /*define background, add noise*/
  setIf(0, 100, I);
  setIf(1, 228, I);
  /*add noise*/
  addNoise(I);

  auto *I_data = I.data();

  //#pragma approx declare tensor_functor(ipt_functor: [i, j, k, 0:1] = ([i, j, k]))
  //#pragma approx declare tensor_functor(opt_functor: [i, 0:2] = ([i], [i]))
  //#pragma approx declare tensor(ipt_tensor: ipt_functor(I_data[0:Nfr, 0:IszX, 0:IszY]))

  //#pragma approx ml(offline) in(ipt_tensor) out(opt_functor(centerX[0:Nfr], centerY[0:Nfr])) label("particlefilter")
  //{

  //}


}

/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the backgrounf intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itsef
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
template<typename VideoSequence>
std::pair<int, int> videoSequence(VideoSequence &I, int IszX, int IszY, int xCenter, int yCenter, int Nfr, int * seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  /*get object centers*/
  int x0 = xCenter;
  int y0 = yCenter;
  I(x0, y0, 0) = 1;

  /*move point*/
  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    printf("At frame %d, particle is at (%d, %d)\n", k, xk, yk);
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      continue;
    I(xk, yk, k) = 1;
  }
  std::pair<int, int> center(xk, yk);

  /*dilate matrix*/
  VideoSequence newMatrix(IszX, IszY, Nfr, seed);
  imdilate_disk(I, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I(x, y, k) = newMatrix(x, y, k);
      }
    }
  }

  /*define background, add noise*/
  setIf(0, 100, I);
  setIf(1, 228, I);
  /*add noise*/
  addNoise(I);

  return center;
}

/**
 * Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the last index
 */
int findIndex(float * CDF, int lengthCDF, float value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) {
    return lengthCDF - 1;
  }
  return index;
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */

template<typename VideoSequence>
particlefilter_results_t particleFilter(VideoSequence &Video, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
  int max_size = IszX * IszY*Nfr;
  //original particle centroid
  float xe = OO->xOrigin();
  float ye = OO->yOrigin();

  bool xe_isnan = false;
  bool ye_isnan = false;
  double average_kernel_time = 0;
  double offload_time = 0;

  for(int _= 0; _ < TRIALS; _++) {
  EventRecorder::CPUEvent Trial{"Trial"};
  Trial.recordStart();

  EventRecorder::CPUEvent CPU_Pre{"CPU Preprocess"};
  CPU_Pre.recordStart();

  //expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int * disk = (int*) calloc(diameter * diameter, sizeof (int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  int * objxy = (int *) calloc(countOnes * 2, sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);

  //initial weights are all equal (1/Nparticles)
  float * weights = (float *) calloc(Nparticles, sizeof(float));
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((float) (Nparticles));
  }
  /****************************************************************
   **************   B E G I N   A L L O C A T E *******************
   ****************************************************************/
  float * likelihood = (float *) calloc(Nparticles + 1, sizeof (float));
  float * arrayX = (float *) calloc(Nparticles, sizeof (float));
  float * arrayY = (float *) calloc(Nparticles, sizeof (float));
  float * xj = (float *) calloc(Nparticles, sizeof (float));
  float * yj = (float *) calloc(Nparticles, sizeof (float));
  float * CDF = (float *) calloc(Nparticles, sizeof(float));

  //GPU copies of arrays
  int * ind = (int*) calloc(countOnes * Nparticles, sizeof(int));
  float * u = (float *) calloc(Nparticles, sizeof(float));

  //Donnie - this loop is different because in this kernel, arrayX and arrayY
  //  are set equal to xj before every iteration, so effectively, arrayX and
  //  arrayY will be set to xe and ye before the first iteration.
  for (x = 0; x < Nparticles; x++) {
    xj[x] = xe;
    yj[x] = ye;
  }

  CPU_Pre.recordEnd();
  EventRecorder::LogEvent(CPU_Pre);

  auto offload_start = get_time();

  int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
#ifdef DEBUG
  printf("BLOCK_SIZE=%d \n",BLOCK_SIZE);
#endif

  float* likelihood_GPU;
  float* arrayX_GPU;
  float* arrayY_GPU;
  float* xj_GPU;
  float* yj_GPU;
  float* CDF_GPU;
  float* partial_sums_GPU;
  float* u_GPU;
  int* objxy_GPU;
  int* ind_GPU;
  int* seed_GPU;
  float* weights_GPU;
  unsigned char* I_GPU;
  VideoSequence *Video_GPU;

  EventRecorder::GPUEvent GPU_Alloc{"GPU Allocate"};
  GPU_Alloc.recordStart();
  cudaMalloc((void**)&likelihood_GPU, (Nparticles + 1)*sizeof(float));

  cudaMalloc((void**)&arrayX_GPU, Nparticles*sizeof(float));
  cudaMalloc((void**)&arrayY_GPU, Nparticles*sizeof(float));
  cudaMalloc((void**)&xj_GPU, Nparticles*sizeof(float));
  cudaMalloc((void**)&yj_GPU, Nparticles*sizeof(float));
  cudaMalloc((void**)&CDF_GPU, Nparticles*sizeof(float));
  cudaMalloc((void**)&u_GPU, Nparticles*sizeof(float));
  //cudaMemcpy(u_GPU, u, Nparticles*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&ind_GPU, countOnes*Nparticles*sizeof(int));
  //cudaMemcpy(ind_GPU, ind, countOnes*Nparticles*sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&weights_GPU, Nparticles*sizeof(float));


  cudaMalloc((void**)&partial_sums_GPU, (Nparticles+1)*sizeof(float));
  cudaMalloc((void**)&objxy_GPU, 2*countOnes*sizeof(int));
  cudaMalloc((void**)&seed_GPU, Nparticles*sizeof(int));
  cudaMalloc((void**)&I_GPU, IszX * IszY * Nfr * sizeof(unsigned char));
  cudaMalloc((void**)&Video_GPU, sizeof(VideoSequence));
  VideoSequence Vid_GPU(I_GPU, IszX, IszY, Nfr, seed_GPU);
  cudaDeviceSynchronize();
  auto start = get_time();


  cudaMemcpy(xj_GPU, xj, Nparticles*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(yj_GPU, yj, Nparticles*sizeof(float), cudaMemcpyHostToDevice);
  // memory copy is not needed, because all the weights are updated first before 
  // they are read in the likelihood kernel. 
  // Just be consistent with the original cuda version 
  cudaMemcpy(weights_GPU, weights, Nparticles*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(I_GPU, Video.data(), IszX * IszY * Nfr * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(Video_GPU, &Vid_GPU, sizeof(VideoSequence), cudaMemcpyHostToDevice);

  cudaMemcpy(seed_GPU, seed, Nparticles*sizeof(int), cudaMemcpyHostToDevice);

  //cudaMemcpy(partial_sums_GPU, likelihood, (Nparticles+1)*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(objxy_GPU, objxy, 2*countOnes*sizeof(int), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  GPU_Alloc.recordEnd();
  EventRecorder::LogEvent(GPU_Alloc);
  timepoint_t tst;
  timepoint_t tend;

  EventRecorder::GPUEvent GPU_Kernel{"GPU Kernel"};
  GPU_Kernel.recordStart();
  
  for (int k = 1; k < Nfr; k++) {
    /****************** L I K E L I H O O D ************************************/
    tst = get_time();
    kernel_likelihood<<<num_blocks, BLOCK_SIZE>>>(
        arrayX_GPU, arrayY_GPU, xj_GPU, yj_GPU, ind_GPU,
        objxy_GPU, likelihood_GPU, Video_GPU, weights_GPU, seed_GPU, partial_sums_GPU,
        Nparticles, countOnes, IszY, Nfr, k, max_size);
    cudaDeviceSynchronize();
    tend = get_time();
    // std::cout << "Elapsed time for likelihood: " << elapsed_time(tst, tend) << "\n";

#ifdef DEBUG
    float * sum = (float *) calloc(Nparticles + 1, sizeof (float));
    cudaMemcpy(sum, partial_sums_GPU, (Nparticles+1)*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < Nparticles+1; i++)
      printf("%f ", sum[i]);
    printf("\n");
#endif

    kernel_sum<<<1, 1>>>(partial_sums_GPU, Nparticles);

#ifdef DEBUG
    // this shows the sum of all partial_sum results
    cudaMemcpy(sum, partial_sums_GPU, sizeof(float), cudaMemcpyDeviceToHost);
    printf("kernel sum: frame=%d partial_sums[0]=%f\n", k, sum[0]);
    free(sum);
#endif

    tst = get_time();
    kernel_normalize_weights<<<num_blocks, BLOCK_SIZE>>>(
        weights_GPU,
        partial_sums_GPU,
        CDF_GPU,
        u_GPU,
        seed_GPU,
        Nparticles );
    cudaDeviceSynchronize();
    tend = get_time();
    // std::cout << "Elapsed time for normalize: " << elapsed_time(tst, tend) << "\n";



    tst = get_time();
    kernel_find_index<<<num_blocks, BLOCK_SIZE>>>(
        arrayX_GPU,
        arrayY_GPU,
        CDF_GPU,
        u_GPU,
        xj_GPU,
        yj_GPU,
        Nparticles );
    cudaDeviceSynchronize();
    tend = get_time();
    // std::cout << "Elapsed time for find index: " << elapsed_time(tst, tend) << "\n";

  } //end loop

  cudaDeviceSynchronize();
  GPU_Kernel.recordEnd();
  EventRecorder::LogEvent(GPU_Kernel);

  cudaMemcpy(arrayX, arrayX_GPU, Nparticles*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(arrayY, arrayY_GPU, Nparticles*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(weights, weights_GPU, Nparticles*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  timepoint_t end = get_time();
  average_kernel_time = elapsed_time(start, end);
  printf("Average execution time of kernels: %f (us)\n",
         elapsed_time(start, end) / (Nfr-1));


  cudaFree(likelihood_GPU);
  cudaFree(arrayX_GPU);
  cudaFree(arrayY_GPU);
  cudaFree(xj_GPU);
  cudaFree(yj_GPU);
  cudaFree(CDF_GPU);
  cudaFree(partial_sums_GPU);
  cudaFree(objxy_GPU);
  cudaFree(u_GPU);
  cudaFree(ind_GPU);
  cudaFree(seed_GPU);
  cudaFree(weights_GPU);
  cudaFree(I_GPU);

  timepoint_t offload_end = get_time();
  offload_time = elapsed_time(offload_start, offload_end);
  printf("Device offloading time: %f (us)\n", elapsed_time(offload_start, offload_end));



  xe = 0;
  ye = 0;
  // estimate the object location by expected values
  for (x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  float distance = sqrt(pow((float) (xe - (int) roundFloat(IszY / 2.0)), 2) + pow((float) (ye - (int) roundFloat(IszX / 2.0)), 2));

  //Output results
  FILE *fid;
  fid=fopen("output.txt", "w+");
  if( fid == NULL ){
    printf( "The file was not opened for writing\n" );
    return particlefilter_results_t{};
  }
  fprintf(fid, "XE: %f\n", xe);
  fprintf(fid, "YE: %f\n", ye);
  fprintf(fid, "distance: %f\n", distance);
  fclose(fid);

  xe_isnan = std::isnan(xe);
  ye_isnan = std::isnan(ye);

  //free regular memory
  free(likelihood);
  free(arrayX);
  free(arrayY);
  free(xj);
  free(yj);
  free(CDF);
  free(ind);
  free(u);
  Trial.recordEnd();
  EventRecorder::LogEvent(Trial);
}

  return {xe_isnan || ye_isnan, offload_time, average_kernel_time, xe, ye};
}


particlefilter_results_t particleFilterApprox(VideoSequence<FrameOuterAccessPolicy> &Video, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {

  using VideoSequence = VideoSequence<FrameOuterAccessPolicy>;
  unsigned char* I_GPU;
  VideoSequence *Video_GPU;
  int* seed_GPU;
  double elapsed = 0;
  float xe_out = 0;
  float ye_out = 0;
  for(int _ = 0; _ < TRIALS; _++) {
  EventRecorder::CPUEvent Trial{"Trial"};
  Trial.recordStart();


  EventRecorder::GPUEvent GPU_Alloc{"GPU Allocate"};
  GPU_Alloc.recordStart();
  cudaMalloc((void**)&I_GPU, IszX * IszY * Nfr * sizeof(unsigned char));
  cudaMalloc((void**)&Video_GPU, sizeof(VideoSequence));
  cudaMalloc((void**)&seed_GPU, Nparticles*sizeof(int));

  VideoSequence Vid_GPU(I_GPU, IszX, IszY, Nfr, seed_GPU);


  cudaMemcpy(I_GPU, Video.data(), IszX * IszY * Nfr * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(Video_GPU, &Vid_GPU, sizeof(VideoSequence), cudaMemcpyHostToDevice);
  cudaMemcpy(seed_GPU, seed, Nparticles*sizeof(int), cudaMemcpyHostToDevice);
  auto tst = get_time();

  float *xe = new float[Nfr];
  float *ye = new float[Nfr];
  float *xe_GPU;
  float *ye_GPU;
  cudaMalloc((void**)&xe_GPU, sizeof(float)*Nfr);
  cudaMalloc((void**)&ye_GPU, sizeof(float)*Nfr);
  GPU_Alloc.recordEnd();
  EventRecorder::LogEvent(GPU_Alloc);


  #pragma approx declare tensor_functor(ipt_functor: [i, j, k, 0:1] = ([i, j, k]))
  #pragma approx declare tensor_functor(opt_functor: [i, 0:2] = ([i], [i]))
  #pragma approx declare tensor(ipt_tensor: ipt_functor(I_GPU[0:Nfr, 0:IszX, 0:IszY]))

  #pragma approx ml(infer) in(ipt_tensor) out(opt_functor(xe_GPU[0:Nfr], ye_GPU[0:Nfr])) 
  {

  }

  EventRecorder::GPUEvent DtoH{"Result DtoH"};
  DtoH.recordStart();

  cudaMemcpy(xe, xe_GPU, sizeof(float)*Nfr, cudaMemcpyDeviceToHost);
  cudaMemcpy(ye, ye_GPU, sizeof(float)*Nfr, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  DtoH.recordEnd();
  EventRecorder::LogEvent(DtoH);

  auto tend = get_time();


  elapsed = elapsed_time(tst, tend);

  xe_out = xe[Nfr-1];
  ye_out = ye[Nfr-1];

  cudaFree(I_GPU);
  cudaFree(Video_GPU);
  cudaFree(seed_GPU);
  cudaFree(xe_GPU);
  cudaFree(ye_GPU);
  delete[] xe;
  delete[] ye;
  Trial.recordEnd();
  EventRecorder::LogEvent(Trial);
  }

  return {false, elapsed, elapsed, xe_out, ye_out};
}

int main(int argc, char * argv[]) {

  const char* usage = "float.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  //check number of arguments
  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }
  //check args deliminators
  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
    printf("%s\n", usage);
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  //converting a string to a integer
  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }

#ifdef DEBUG
  printf("dimX=%d dimY=%d Nfr=%d Nparticles=%d\n", 
      IszX, IszY, Nfr, Nparticles);
#endif

  approx::util::warmup();
  //establish seed
  int * seed = (int *) calloc(Nparticles, sizeof(int));
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = i+1;

  // calloc matrix
    // OO = ObjectOriginFactory::create(ObjectOriginType::UAR, "", 0, 84, 64, IszY, IszX);
  int radii[] = {5};
  // int radii[] = {15, 20, 25, 30};
  // int samples[] = {1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14};//, 1<<14, 1<<15, 1<<16};
  int samples[] = {1<<9};
  // int samples[] = {1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13};//, 1<<14, 1<<15, 1<<16};
  int first = 0;
  int second = 0;
  std::cout << "DATA" << ":" << "radius" << "," << "count" << "," << "samples" << "," << "nan" << "," << "kernel_time" << "," << "offload_time" << "," << "x0" << "," << "y0" << "," << "xe" << "," << "ye" << "\n";
  particlefilter_results_t result;
  for(int radius : radii) {
    OO = new NonRepeatingSquareObjectOrigin(IszY, IszX, radius);
    for(int sample : samples) {
      int nan_count = 0;
      int count = 0;
      double average_kernel_time = 0;
      double average_offload_time = 0;
      for(auto CoordPair : *dynamic_cast<NonRepeatingSquareObjectOrigin*>(OO)) {
        NonRepeatingSquareObjectOrigin *nonRepeatingOO = dynamic_cast<NonRepeatingSquareObjectOrigin*>(OO);
      auto I = VideoSequence<FrameOuterAccessPolicy>(IszX, IszY, Nfr, seed);
      timepoint_t start = get_time();

      //call video sequence
      auto final_center = videoSequence(I, IszX, IszY, CoordPair.first, CoordPair.second, Nfr, seed);
      timepoint_t endVideoSequence = get_time();
      printf("VIDEO SEQUENCE TOOK %f (s)\n", elapsed_time(start, endVideoSequence));

      // for(int _trial = 0; _trial < 10; _trial++) {

          //call particle filter
          #ifndef APPROX
          result = particleFilter(I, IszX, IszY, Nfr, seed, sample);
          #else
          result = particleFilterApprox(I, IszX, IszY, Nfr, seed, sample);
          #endif

      // }
      auto was_nan = result.num_nan;
      nan_count += was_nan;
      average_kernel_time = result.average_kernel_time;
      average_offload_time = result.offload_time;
      timepoint_t endParticleFilter = get_time();
      printf("PARTICLE FILTER TOOK %f (s)\n", elapsed_time(endVideoSequence, endParticleFilter));

      printf("ENTIRE PROGRAM TOOK %f (s)\n", elapsed_time(start, endParticleFilter));
      first = final_center.first;
      second = final_center.second;

      std::cout << "DATA:" << radius << "," << count << "," << sample << "," << was_nan 
      << "," << average_kernel_time << "," << average_offload_time
      << "," << first << "," << second 
      << "," << result.particle_x << "," << result.particle_y << "\n";

      count++;
      if(count >= 100 || count > nonRepeatingOO->size()) {
        break;
      }
    }
      nan_count = 0;
    }
      delete OO;
  }

   //OO = new NonRepeatingUARObjectOrigin(IszY, IszX, 13000);
   //NonRepeatingUARObjectOrigin &nonRepeatingOO = *dynamic_cast<NonRepeatingUARObjectOrigin*>(OO);
   //auto nframes = nonRepeatingOO.getCoordinates().size();
   //auto Video = VideoSequence<FrameOuterAccessPolicy>(IszX, IszY, nframes, seed);
   //videoSequenceGivenFrames(Video, IszX, IszY, nonRepeatingOO.getCoordinates(), seed);
  free(seed);
  return 0;
}


