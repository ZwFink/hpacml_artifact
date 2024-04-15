#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <approx_debug.h>
#include "shared.h"

#define MAX_PLATFORMS     8
#define MAX_DEVICES      32
#define MAX_INFO_STRING 256

#define DATA_DIR          "../data/bm1"
#define FILE_LIGAND       "/ligand.in"
#define FILE_PROTEIN      "/protein.in"
#define FILE_FORCEFIELD   "/forcefield.in"
#define FILE_POSES        "/poses.in"
#define FILE_REF_ENERGIES "/ref_energies.out"

#define REF_NPOSES 65536

// Energy evaluation parameters
#define CNSTNT   45.0f
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST  5.5f
#define NPPDIST   1.0f

void loadParameters(int argc, char *argv[]);
void freeParameters();
void printTimings(double start, double end, double poses_per_wi);
void checkError(int err, const char *op);
void runCUDA(float* results, int ni);

FILE* openFile(const char *parent, const char *child,
               const char* mode, long *length)
{
  char name[strlen(parent) + strlen(child) + 1];
  strcpy(name, parent);
  strcat(name, child);

  FILE *file = NULL;
  if (!(file = fopen(name, mode)))
  {
    fprintf(stderr, "Failed to open '%s'\n", name);
    exit(1);
  }
  if(length){
    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    rewind(file);
  }
  return file;
}

int main(int argc, char *argv[])
{
  loadParameters(argc, argv);
  freeParameters();
  approx::util::warmup();

  for(int i = 0; i < params.iterations; i++) {
  EventRecorder::CPUEvent Trial{"Trial"};
  Trial.recordStart();
  loadParameters(argc, argv);
  params.poses[0] += params.start_idx;
  params.poses[1] += params.start_idx;
  params.poses[2] += params.start_idx;
  params.poses[3] += params.start_idx;
  params.poses[4] += params.start_idx;
  params.poses[5] += params.start_idx;

  printf("\n");
  printf("Poses     : %d\n", params.nposes);
  printf("Start Idx : %d\n", params.start_idx);
  printf("Iterations: %d\n", params.iterations);
  printf("Ligands   : %d\n", params.natlig);
  printf("Proteins  : %d\n", params.natpro);
  printf("Deck      : %s\n", params.deckDir);
  printf("Num pack  : %d\n", params.ni);


  float *resultsCUDA = (float*) malloc(params.nposes*sizeof(float));
  float *resultsRef = (float*) malloc(params.nposes*sizeof(float));

    runCUDA(resultsCUDA, params.ni);

  // Load reference results from file
  FILE* ref_energies = openFile(params.deckDir, FILE_REF_ENERGIES, "r", NULL);
  size_t n_ref_poses = params.nposes;
  if (params.nposes > REF_NPOSES) {
    printf("Only validating the first %d poses.\n", REF_NPOSES);
    n_ref_poses = REF_NPOSES;
  }

  for (size_t i = 0; i < n_ref_poses; i++)
    fscanf(ref_energies, "%f", &resultsRef[i]);

  fclose(ref_energies);

  float maxdiff = -100.0f;
  printf("\n Reference        CUDA   (diff)\n");
  for (int i = 0; i < n_ref_poses; i++)
  {
    if (fabs(resultsRef[i]) < 1.f && fabs(resultsCUDA[i]) < 1.f) continue;

    float diff = fabs(resultsRef[i] - resultsCUDA[i]) / resultsCUDA[i];
    if (diff > maxdiff) {
      maxdiff = diff;
      // printf ("Maxdiff: %.2f (%.3f vs %.3f)\n", maxdiff, resultsRef[i], resultsCUDA[i]);
    }

    if (i < 64)
      printf("%7.2f    vs   %7.2f  (%5.2f%%)\n", resultsRef[i], resultsCUDA[i], 100*diff);
  }
  printf("\nLargest difference was %.3f%%\n\n", maxdiff*100);

  free(resultsCUDA);
  free(resultsRef);

  params.poses[0] -= params.start_idx;
  params.poses[1] -= params.start_idx;
  params.poses[2] -= params.start_idx;
  params.poses[3] -= params.start_idx;
  params.poses[4] -= params.start_idx;
  params.poses[5] -= params.start_idx;
  freeParameters();
  Trial.recordEnd();
  EventRecorder::LogEvent(Trial);
  }
}

int parseInt(const char *str)
{
  char *next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

void loadParameters(int argc, char *argv[])
{
  // Defaults
  params.deckDir        = DATA_DIR;
  params.iterations = 8;
  _cuda.wgsize      = 64;
  _cuda.posesPerWI  = 4;
  int nposes        = 65536;
  params.ni = 1;
  params.start_idx = 0;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--device") || !strcmp(argv[i], "-d"))
    {
      if (++i >= argc || (_cuda.deviceIndex = parseInt(argv[i])) < 0)
      {
        printf("Invalid device index\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || (params.iterations = parseInt(argv[i])) < 0)
      {
        printf("Invalid number of iterations\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--numposes") || !strcmp(argv[i], "-n"))
    {
      if (++i >= argc || (nposes = parseInt(argv[i])) < 0)
      {
        printf("Invalid number of poses\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--posesperwi") || !strcmp(argv[i], "-p"))
    {
      if (++i >= argc || (_cuda.posesPerWI = parseInt(argv[i])) < 0)
      {
        printf("Invalid poses-per-workitem value\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--wgsize") || !strcmp(argv[i], "-w"))
    {
      if (++i >= argc || (_cuda.wgsize = parseInt(argv[i])) < 0)
      {
        printf("Invalid work-group size\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--deck"))
    {
      if (++i >= argc)
      {
        printf("Invalid deck\n");
        exit(1);
      }
      params.deckDir = argv[i];
    }
    else if(!strcmp(argv[i], "--ni"))
    {
      if (++i >= argc || (params.ni = parseInt(argv[i])) < 1)
      {
        printf("Invalid number of items to pack into one tensor\n");
        exit(1);
      }
    }
    else if(!strcmp(argv[i], "--start") || !strcmp(argv[i], "-s"))
    {
      if (++i >= argc || (params.start_idx = parseInt(argv[i])) < 0)
      {
        printf("Invalid start index\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./bude [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print this message\n");
      printf("      --list               List available devices\n");
      printf("      --device     INDEX   Select device at INDEX\n");
      printf("  -i  --iterations I       Repeat kernel I times\n");
      printf("  -n  --numposes   N       Compute results for N poses\n");
      printf("  -p  --poserperwi PPWI    Compute PPWI poses per work-item\n");
      printf("  -w  --wgsize     WGSIZE  Run with work-group size WGSIZE\n");
      printf("      --deck       DECK    Use the DECK directory as input deck\n");
      printf("      --ni         NUM     Number of items to pack into one tensor\n");
      printf("  -s  --start      NUM     Start iteration of the poses\n");
      printf("\n");
      exit(0);
    }
    else
    {
      printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
      exit(1);
    }
  }

  FILE *file = NULL;
  long length;

  file = openFile(params.deckDir, FILE_LIGAND, "rb", &length);
  params.natlig = length / sizeof(Atom);
  params.ligand = (Atom*) malloc(params.natlig*sizeof(Atom));
  fread(params.ligand, sizeof(Atom), params.natlig, file);
  fclose(file);

  file = openFile(params.deckDir, FILE_PROTEIN, "rb", &length);
  params.natpro = length / sizeof(Atom);
  params.protein = (Atom*) malloc(params.natpro*sizeof(Atom));
  fread(params.protein, sizeof(Atom), params.natpro, file);
  fclose(file);

  file = openFile(params.deckDir, FILE_FORCEFIELD, "rb", &length);
  params.ntypes = length / sizeof(FFParams);
  params.forcefield = (FFParams*) malloc(params.ntypes*sizeof(FFParams));
  fread(params.forcefield, sizeof(FFParams), params.ntypes, file);
  fclose(file);

  file = openFile(params.deckDir, FILE_POSES, "rb", &length);
  long available = length / 6 / sizeof(float);
  for (int i = 0; i < 6; i++)
    params.poses[i] = (float*) malloc(available*sizeof(float));

  params.nposes = 0;
  for (int i = 0; i < 6; i++)
    {
      fseek(file, i*available*sizeof(float), SEEK_SET);
      fread(params.poses[i], sizeof(float), available, file);
    }
  fclose(file);
  params.nposes = nposes;
}

void freeParameters()
{
  free(params.ligand);
  free(params.protein);
  free(params.forcefield);
  for (int i = 0; i < 6; i++)
    free(params.poses[i]);
}
