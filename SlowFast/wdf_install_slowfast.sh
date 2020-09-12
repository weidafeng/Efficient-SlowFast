#!/bin/bash
# install slowfast script, writed by wdf 2020-07-09


# install fvcore mannually
cd /data1/config_slowfast/fvcore
pip install -e .

# install some libary
pip install simplejson av psutil sklearn ipython tensorboard opencv-python 

# install moviepy mannually
cd /data1/config_slowfast/moviepy-1.0.3
python setup.py install

# install detectron2
cd /data1/config_slowfast/cocoapi/PythonAPI
make

# install ptflops (note, we modify the raw code to support the two-stream slowfast models)
cd  /data1/config_slowfast/flops-counter.pytorch-master
pip install -e .

cd /data1/config_slowfast/detectron2
python -m pip install -e .

# install slowfast
export PYTHONPATH=/data1/SlowFast_vis_0709/SlowFast/slowfast:$PYTHONPATH
cd /data1/SlowFast_vis_0709/SlowFast/
python setup.py build develop
