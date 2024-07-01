# HPAC-ML Artifact
This repository helps you reproduce the key results of the SC24 paper "HPAC-ML: A Programming Model for Embedding ML Surrogates in Scientific Applications".
With a few commands, you can reproduce Figures 7 and 8, and have the data needed to produce Figures 5 and 6.

To ease the process, we make the software stack with the HPAC-ML compiler and runtime system available in a container.
Two container runtimes are supported: Docker and Apptainer.
**Note**: For docker use, we assume that `docker` can be run in [rootless mode](https://docs.docker.com/engine/install/linux-postinstall/) and that you have downloaded and installed [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The script `setup.sh` lets you download or build the container image and clones the necessary repositories and downloads any needed datasets.
Finally, `setup.sh` creates `run_container.sh`, which puts you in the container.
Inside the container, you'll want to `cd /srgt`, where the evaluation scripts have been mounted.

## Building/Downloading the Container
You have different options for obtaining the container image.
You can build an image on the machine where you'll be evaluating, or download the pre-built image from Dockerhub.
The build/download steps support two container runtimes: docker and apptainer.

No matter your choice, a script `run_container.sh` will be created, which runs the container when executed.

### Downloading the Container
You can download the container for a docker/apptainer runtime with:
```bash
./setup.sh docker download
./setup.sh apptainer download
```

This will download the container image, clone the HPAC-ML repositories and data.

#### Caution
The container was built on a machine with Xeon Gold 6230 CPUs and may not be portable to other systems.
For instance, when running the container on an AMD Epyc system, we experience crashes with `Illegal instruction`.

### Building the Container
You may wish to build the container.
This is done with:
```bash
./setup.sh docker build BUILD_JOBS
./setup.sh apptainer build BUILD_JOBS
```
Where `BUILD_JOBS` is the number of CPUs to use for the build.
On a system with 64 CPUs, the build takes about 30 minutes.


## Running the Evaluation
To enter the container, run:
```bash
./run_container.sh
cd /srgt/experimentation
```

From here, you can run the included `evaluate.py` script to evaluate the trained models and create plots for any benchmark.
More details are given in the corresponding [README](benchmark_evaluation/README.md).
The evaluation process runs both the original benchmark and the benchmark approximated using the models trained during the BO-based neural-architecture search detailed in the paper.
After running the benchmark, the two runs are compared to compute important statistics such as speedup and error, and timing information is captured.

We offer two different evaluation scenarios: _small_ and _large_.
The _large_ scenario evaluates all models found during BO, and _small_ evaluates only the models shown in any of the result figures.

The required runtime for each of the scenarios is given below:
|     Benchmark    | Large (m) | Small (m) |
|:----------------:|:---------:|:---------:|
|     MiniBUDE     |    240    |     23    |
| Binomial Options |     28    |     7     |
|       Bonds      |     13    |     8     |
|    MiniWeather   |     30    |    2.5    |
|  ParticleFilter  |    50     |     15    |


### Evaluating a Benchmark
To evaluate `binomialoptions` with the _small_ configuration, simply run:
```base
python3 evaluate.py --benchmark binomialoptions --size small
```

This will run to complection and produce the file `plots/binomialoptions.png` with the results.