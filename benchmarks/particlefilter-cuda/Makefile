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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 --cuda-gpu-arch=$(ARCH) -g -D_GLIBCXX_USE_CXX11_ABI=0

# Linker Flags
LDFLAGS =-L$(HPAC_LIBRARY_LOCATION) -lapprox -lcudart -fapprox

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
