#!/bin/bash
# Script to download and extract MNIST data from the web.

#Download Files
wget -i filelist.txt

#Decompress Them
gunzip *.gz
