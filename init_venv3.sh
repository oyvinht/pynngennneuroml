#!/usr/bin/env sh

BASE=`pwd`
VENV=$BASE/venv3
GENN=$BASE/genn
PYNN_GENN=$BASE/pynn_genn

# Create and enter virtual env for python3
/usr/bin/env python3 -m venv venv3
source $VENV/bin/activate

# Get pip packages
pip install matplotlib
pip insntall numpy
pip install pynn
pip install pyneuroml

# Get GeNN and PyNN_GeNN
git clone https://github.com/genn-team/genn $GENN
cd $GENN
make LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
make DYNAMIC=True LIBRARY_DIRECTORY=$GENN/pygenn/genn_wrapper/
/usr/bin/env python3 setup.py develop

git clone https://github.com/genn-team/pynn_genn $PYNN_GENN
cd $PYNN_GENN
/usr/bin/env python3 setup.py develop

