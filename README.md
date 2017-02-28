TensorFlow implementation of RFCN
=================================

Paper is available on https://arxiv.org/abs/1605.06409.

Building
--------

The ROI pooling and the MS COCO loader needs to be compiled first. To do so, run make in the root directory of the project.

Testing
-------

You can run trained models with test.py. Model path should be given without file extension (without .data* and .index). An example:

![preview](https://cloud.githubusercontent.com/assets/2706617/23438600/d7d79e86-fe12-11e6-9fec-ecdb15ba8806.jpg)

License
-------

The software is under Apache 2.0 license. See http://www.apache.org/licenses/LICENSE-2.0 for further details.