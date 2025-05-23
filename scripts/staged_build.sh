#!/bin/bash
echo "Installing ..." 

read -p "Stage Number " stage

if [ $stage -eq 1] ; then
    apt-get update
    apt-get upgrade -y
done