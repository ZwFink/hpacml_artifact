#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_80
CAPTURE_OUTPUT?=0

#===============================================================================
# Program name & source code list
#===============================================================================

program = bonds

source = bondsEngine.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Xcompiler -Wall --cuda-gpu-arch=$(ARCH) -fapprox -DCAPTURE_OUTPUT=$(CAPTURE_OUTPUT)

# Linker Flags
LDFLAGS =-L$(HPAC_LIBRARY_LOCATION) -lapprox -L/sw/spack/delta-2022-03/apps/gcc/11.2.0-gcc-8.4.1-fxgnsyr/lib64 -lcudart -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/hdf5-1.14.3-7b3feas/lib -lhdf5

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu bondsKernelsCpu.h bondsKernelsGpu.h bondsStructs.h 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run:$(program)
	./$(program) 100
