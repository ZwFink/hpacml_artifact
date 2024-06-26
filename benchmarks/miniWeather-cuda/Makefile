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

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DNX=768 \
       	  -DNZ=384 \
	  -DSIM_TIME=600 \
	  -DDATA_SPEC=1 \
		-I/hpacml-spack-env/.spack-env/view/include \
          -fapprox -std=c++17 --cuda-gpu-arch=$(ARCH) -g -DCAPTURE_OUTPUT=$(CAPTURE_OUTPUT) -D_GLIBCXX_USE_CXX11_ABI=0

# Linker Flags
LDFLAGS =-L$(HPAC_LIBRARY_LOCATION) -lapprox -lcudart -fapprox -L/hpacml-spack-env/.spack-env/view/lib/ -lmpi

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
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

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	mpiexec -n 1 ./$(program)
