#!/bin/bash 
echo Cloning hpacml-experimentation...
git clone git@github.com:ZwFink/hpacml-experimentation.git
echo Success!

echo Downloading trained models...
wget -O hpacml_models.zip https://zenodo.org/records/12586948/files/hpacml_models.zip\?download\=1
echo Success! Unpacking trained models...
unzip hpacml_models.zip
rm hpacml_models.zip
echo Success!
