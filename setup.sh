#!/bin/bash
image_type="$1"
build_type="$2"
build_jobs="$3"

if [ "$image_type" != "docker" ] && [ "$image_type" != "apptainer" ]; then
    echo "Error: image_type must be either 'docker' or 'apptainer'"
    exit 1
fi

if [ "$build_type" != "download" ] && [ "$build_type" != "build" ]; then
    echo "Error: build_type must be either 'download' or 'build'"
    exit 1
fi

if [ -z "$build_jobs" ]; then
    echo "Warning: using default build jobs of 4"
    build_jobs=4
fi

pushd software_env
if [ "$image_type" == "docker" ]; then
    if [ "$build_type" == "download" ]; then
        echo "Downloading Docker image..."
        echo "WARNING: This might cause issues if downloading the image built on a different architecture."
        echo "If you have issues running the image, please build the image locally."
        echo "You have been warned..."
        docker pull zanef2/hpacml:latest 
    else
        echo "Building Docker image with $build_jobs jobs..."
        docker build --build-arg build_jobs=$build_jobs -t hpacml -f Dockerfile .
    fi

popd
cat >run_container.sh <<EOL
docker run \
  -v $HOME:/home \
  -v ./experimentation/hpacml-experimentation:/srgt/experimentation \
  -v ./experimentation/hpacml_models:/srgt/models \
  -v ./benchmarks:/srgt/benchmarks \
  -v ./benchmarks/input_data:/srgt/input_data \
  --gpus all \
  library/hpacml
EOL
else
    if [ "$build_type" == "download" ]; then
        echo "Downloading image from Dockerhub and converting it to Apptainer..."
        apptainer pull hpacml.sif docker://zanef2/hpacml:latest
    else
        echo "Building Apptainer image..."
        apptainer build hpacml.sif hpacml_apptainer.def
    fi
popd
cat >run_container.sh <<EOL
#!/bin/bash
apptainer run --compat --bind $HOME:/home \
  --bind ./experimentation/hpacml-experimentation:/srgt/experimentation \
  --bind ./experimentation/hpacml_models:/srgt/models \
  --bind ./benchmarks:/srgt/benchmarks \
  --bind ./benchmarks/input_data:/srgt/input_data \
  --nv --fakeroot -f ./software_env/hpacml.sif
EOL
fi

chmod +x run_container.sh

echo "Performing setup for benchmarks..."
pushd benchmarks
./setup.sh
popd

echo "Performing setup for evaluation..."
pushd experimentation
./setup.sh
popd