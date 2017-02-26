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
import tensorflow.contrib.slim as slim

import Utils.RandomSelect

from InceptionResnetV2 import *
from BoxEngine.BoxNetwork import BoxNetwork


class BoxInceptionResnet(BoxNetwork):
	def __init__(self, inputs, nCategories, name="BoxNetwork", weightDecay=0.00004, reuse=False, isTraining=True, trainFrom=None, hardMining=True):
		self.boxThreshold = 0.5

		if trainFrom == "0":
			trainFrom = "Conv2d_1a_3x3"
		elif trainFrom == "-1":
			trainFrom = None

		print("Training network from "+(trainFrom if trainFrom is not None else "end"))

		with tf.variable_scope(name, reuse=reuse) as scope:
			self.googleNet = InceptionResnetV2("features", inputs, trainFrom=trainFrom)
			self.scope=scope
		
			with tf.variable_scope("Box"):
				#scale_16 = self.googleNet.getOutput("Repeat_1")[:,1:-1,1:-1,:]
				scale_16 = self.googleNet.getOutput("Mixed_6a")
				scale_32 = self.googleNet.getOutput("PrePool")

				BoxNetwork.__init__(self, nCategories, scale_16, 16, [16,16], scale_32, 32, [32,32], weightDecay=weightDecay, hardMining=hardMining)

				# scale_32 = tf.image.resize_bilinear(scale_32, tf.shape(scale_16)[1:3])

				# with slim.arg_scope([slim.conv2d],
				# 		weights_regularizer=slim.l2_regularizer(weightDecay),
				# 		biases_regularizer=slim.l2_regularizer(weightDecay),
				# 		padding='SAME',
				# 		activation_fn = tf.nn.relu):

				# 	net = tf.concat(3, [scale_32, scale_16])
				# 	common = slim.conv2d(net, 1024, 1, scope='features_downsample')
				# 	rpnFeatureMap = slim.conv2d(common, 1024, 3, scope='features_conv')

	
	def getVariables(self, includeFeatures=False):
		if includeFeatures:
			return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
		else:
			vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name+"/Box/")
			vars += self.googleNet.getTrainableVars()

			print("Training variables: ", [v.op.name for v in vars])
			return vars

	def importWeights(self, sess, filename):
		self.googleNet.importWeights(sess, filename, includeTraining=True)
