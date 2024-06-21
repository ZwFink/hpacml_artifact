#!/bin/bash
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
export PATH="$PWD/spack/bin:$PATH"
spack env create hpacml ./spack.yaml
. $PWD/spack/share/spack/setup-env.sh
spack env activate hpacml -p
spack external find --all --not-buildable --exclude openssl --exclude openblas --exclude bzip2
spack concretize
spack install -j $1

python3 -m pip install -r ./requirements.txt
