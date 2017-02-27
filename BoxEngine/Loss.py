# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import tensorflow as tf 
import BoxEngine.BoxUtils as BoxUtils

def smooth_l1(x):
    with tf.name_scope("smooth_l1"):
        abs_x = tf.abs(x)
        lessMask = tf.cast(abs_x < 1.0, tf.float32)

        return lessMask * (0.5 * tf.square(x)) + (1.0 - lessMask) * (abs_x - 0.5)

def boxRegressionLoss(boxes, refBoxes, boxSizes):
    with tf.name_scope("boxRegressionLoss"):
        x, y, w, h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(boxes, axis=1))
        boxH, boxW = tf.unstack(boxSizes, axis=1)
        ref_x, ref_y, ref_w, ref_h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(refBoxes, axis=1))

        x=tf.reshape(x,[-1])
        y=tf.reshape(y,[-1])
        w=tf.reshape(w,[-1])
        h=tf.reshape(h,[-1])

        boxH = tf.reshape(boxH,[-1])
        boxW = tf.reshape(boxW,[-1])

        ref_x=tf.reshape(ref_x,[-1])
        ref_y=tf.reshape(ref_y,[-1])
        ref_w=tf.reshape(ref_w,[-1])
        ref_h=tf.reshape(ref_h,[-1])

        # Smooth L1 loss is defined on NN output values, which is not available here. However
        # we can transform the loss back in the NN output space (the same holds for y and h):
        #
        # tx-tx' = (x-x')/wa
        # tw-tw' = log(w/w')

        return smooth_l1((x-ref_x)/boxW) + smooth_l1((y-ref_y)/boxH) + smooth_l1(tf.log(w/ref_w)) + smooth_l1(tf.log(h/ref_h))
