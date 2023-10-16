#!/bin/bash

set -e

cuda_deb="cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb"
if [ ! -f "$cuda_deb" ] ; then
    wget http://127.0.0.1:8000/$cuda_deb
fi
sudo dpkg -i $cuda_deb
sudo cp /var/cuda-repo-*/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt install -qq -y --no-install-recommends cuda

cudnn_deb="cudnn-local-repo-ubuntu2204-8.9.4.25_1.0-1_amd64.deb"
if [ ! -f "$cudnn_deb" ]; then
    wget http://127.0.0.1:8000/$cudnn_deb
fi
sudo dpkg -i $cudnn_deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libcudnn8=8.9.4.25-1+cuda11.8 libcudnn8-dev=8.9.4.25-1+cuda11.8

tensorrt_deb="nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8_1.0-1_amd64.deb"
if [ ! -f "$tensorrt_deb" ]; then
    wget http://127.0.0.1:8000/$tensorrt_deb
fi
sudo dpkg -i $tensorrt_deb
sudo cp /var/nv-tensorrt-local-repo-*/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update &&
sudo apt install -qq -y --no-install-recommends tensorrt=8.6.1.6-1+cuda11.8

sudo rm -rf /var/cuda-repo-ubuntu2004-11-8-local \
            /var/cudnn-local-repo-ubuntu2004-8.8.1.3 \
            /var/nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8
