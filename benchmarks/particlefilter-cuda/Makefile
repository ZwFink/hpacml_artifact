#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_80

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu 

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 --cuda-gpu-arch=$(ARCH) -g 

# Linker Flags
LDFLAGS = -L$(HPAC_LIBRARY_LOCATION)/approx-llvm/build_compiler/lib -lapprox -L$(CONDA_LOCATION)//miniconda3/envs/hpac_ml17/lib/ -L/sw/spack/delta-2022-03/apps/gcc/11.2.0-gcc-8.4.1-fxgnsyr/lib64 -lcudart -lhdf5

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu kernel_find_index.h  kernel_likelihood.h  kernel_normalize_weights.h  kernel_sum.h video_sequence.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) output.txt

run: $(program)
	./main -x 128 -y 128 -z 10 -np 400000
