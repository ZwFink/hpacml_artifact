FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel AS base
MAINTAINER Zane Fink <zanef2@illinois.edu>
ARG cuda_arch=70
ARG build_jobs=3
WORKDIR /

RUN \
    apt-get update &&\
    apt-get install -y vim git xz-utils autoconf automake libssl-dev unzip libcurl4-openssl-dev xz-utils patch bzip2 gfortran file &&\
    apt-get upgrade -y &&\
    apt-get clean

RUN \
    mkdir -p hpacml-spack-env
COPY spack.yaml hpacml-spack-env/spack.yaml

FROM base AS setup-spack-env
RUN \
    git clone --depth 1 --branch releases/v0.22 https://github.com/spack/spack.git &&\
    . spack/share/spack/setup-env.sh &&\
    spack compiler find &&\
    spack env activate -p hpacml-spack-env &&\
    spack external find --all --exclude openssl --exclude openblas --exclude bzip2 --exclude binutils --not-buildable


FROM setup-spack-env AS install-spack-env
RUN \
    . spack/share/spack/setup-env.sh &&\
    spack env activate -p hpacml-spack-env &&\
    spack install --fail-fast --no-checksum -j${build_jobs}

FROM install-spack-env AS clean-spack
RUN \
    . spack/share/spack/setup-env.sh &&\
    spack clean --all
FROM clean-spack AS install-pip
COPY requirements.txt hpacml-spack-env/requirements.txt
RUN \
    . spack/share/spack/setup-env.sh &&\
    spack env activate -p hpacml-spack-env &&\
    python3 -m pip install -r hpacml-spack-env/requirements.txt

from install-pip as setup-hpacml
RUN \
    . spack/share/spack/setup-env.sh &&\
    spack env activate -p hpacml-spack-env &&\
    git clone --single-branch --branch sc24 --depth 1 https://github.com/ZwFink/hpacml.git

RUN ln -s /usr/local/cuda-11.8/lib64/libcudart.so /usr/lib64/libcudart.so
RUN \ 
  . spack/share/spack/setup-env.sh &&\
    spack env activate -p hpacml-spack-env && cd hpacml && ./setup.sh /hpacml/build_compiler 3

CMD ["/bin/bash", "-c", "source /spack/share/spack/setup-env.sh && spack env activate -p hpacml-spack-env && source /hpacml/hpac_env.sh && exec /bin/bash"]
