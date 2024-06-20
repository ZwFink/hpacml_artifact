#!/bin/bash
export PATH="$PWD/spack/bin:$PATH"
. /scratch/bcbs/zanef2/hpacml_artifact/software_env/spack/share/spack/setup-env.sh
spack env activate hpacml -p
