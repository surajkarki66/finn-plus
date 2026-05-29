#!/bin/bash

sudo pip install bitstring
sudo pip install finn-dataset-loading
sudo pip install onnx==1.17.0
sudo pip install qonnx==1.0.0
sudo pip install h5py
# Ensure other installs don't upgrade numpy too far:
sudo pip install "numpy<2.0.0"
# Somehow this was missing after above installs:
sudo pip install grpcio==1.64.0
# Required for boards with older PYNQ images (< 3.1.1):
#sudo pip install pynqmetadata==0.1.5 # Workaround for https://discuss.pynq.io/t/how-to-address-axilite-interface-in-pynq-v3-0/4831
