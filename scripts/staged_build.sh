#!/bin/bash
echo "Installing ..." 

read -p "Stage Number " stage

function reboot_server {
    for i in 3 2 1; do
        echo "$i..."
        sleep 1
        reboot
    done
}

if [ $stage -eq 1 ] ; then
    apt-get update
    apt-get upgrade -y
    reboot_server
fi

if [ $stage -eq 2 ] ; then
    lspci | grep -i nvidia
    ubuntu-drivers devices
    apt search nvidia-driver
    sudo ubuntu-drivers autoinstall
    nvidia-smi
fi

if [ $stage -eq 3 ] ; then
    read -p "State a driver version(i.e, 560) " version
    apt install pkg-config libglvnd-dev dkms build-essential libegl-dev libegl1 libgl-dev libgl1 libgles-dev libgles1 libglvnd-core-dev libglx-dev libopengl-dev gcc make
    apt update 
    apt install nvidia-driver-$version
    apt install python3.12-venv
    reboot_server
fi

if [ $stage -eq 4 ] ; then
    apt install nvidia-utils-535
    apt install nvidia-settings 
    apt install nvtop
    apt install nvidia-cuda-toolkit
    apt install python3-dev
    dpkg -L nvidia-utils-535 | grep bin
    export CUDA_HOME=/usr
    echo 'export CUDA_HOME=/usr' >> ~/.bashrc
    source ~/.bashrc
    echo $CUDA_HOME
    $CUDA_HOME/bin/nvcc --version
fi
  git config --global user.email "branden@branden.com"
  git config --global user.name "Branden A10"
if [ $stage -eq 4 ] ; then
    cd ..
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install moviepy
    python -m pip install flash-attn --no-build-isolation


if [ $stage -eq 0 ] ; then
    sudo apt-get remove --purge '^nvidia-.*'
    sudo apt autoremove
    reboot_server
fi
