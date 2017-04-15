TensorFlow implementation of RFCN
=================================

Paper is available on https://arxiv.org/abs/1605.06409.

Building
--------

The ROI pooling and the MS COCO loader needs to be compiled first. To do so, run make in the root directory of the project. You may need to edit *BoxEngine/ROIPooling/Makefile* if you need special linker/compiler options.

Testing
-------

You can run trained models with test.py. Model path should be given without file extension (without .data* and .index). An example:

![preview](https://cloud.githubusercontent.com/assets/2706617/25061919/2003e832-21c1-11e7-9397-14224d39dbe9.jpg)

Pretrained model
----------------

You can download a pretrained model from here:
http://xdever.engineerjs.com/rfcn-tensorflow-export.tar.bz2

Extract it to your project directory. Then you can run the network with the following command:
./test.py -n export/model -i \<input image\> -o \<output image\>

License
-------

The software is under Apache 2.0 license. See http://www.apache.org/licenses/LICENSE-2.0 for further details.

Notes
-----

This code requires TensorFlow 1.0.
