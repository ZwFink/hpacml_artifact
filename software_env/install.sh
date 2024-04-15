#!/bin/bash
spack env create hpacml ./spack.lock
spack env activate hpacml -p
spack concretize
spack install -j $1

python3 -m pip install -r ./requirements.txt
