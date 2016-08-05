#!/bin/sh
#
# Install Linux Dependencies
# ==========================
# This little script will automatically install all the packages needed to run
# EEGrunt on (ubuntu) Linux, if you're lucky.

sudo apt-get install python python-pip python-dev build-essential libatlas-base-dev gfortran python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

sudo pip install scipy numpy
