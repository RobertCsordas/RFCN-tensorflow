#!/usr/bin/python
#
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
from tensorflow.python.ops import control_flow_ops
from Dataset.CocoDataset import *
from Utils.RunManager import *
from Utils.CheckpointLoader import *
from Utils.ArgSave import *
import sys
from BoxInceptionResnet import *
from Dataset import Augment
from Visualize import VisualizeOutput
from Utils import Model

parser = StorableArgparse(description='Kaggle fish trainer.')
parser.add_argument('-learningRate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-adamEps', type=float, default=1e-8, help='Adam epsilon')
parser.add_argument('-dataset', type=str, default="/data/Datasets/COCO", help="Path to kaggle dataset")
parser.add_argument('-name', type=str, default="save", help="Directory to save checkpoints")
parser.add_argument('-saveInterval', type=int, default=10000, help='Save model for this amount of iterations')
parser.add_argument('-reportInterval', type=int, default=30, help='Repeat after this amount of iterations')
parser.add_argument('-displayInterval', type=int, default=60, help='Display after this amount of iterations')
parser.add_argument('-optimizer', type=str, default='adam', help='sgd/adam/rmsprop')
parser.add_argument('-resume', type=str, help='Resume from this file', save=False)
parser.add_argument('-report', type=str, default="", help='Create report here', save=False)
parser.add_argument('-trainFrom', type=str, default="-1", help='Train from this layer. Use 0 for all, -1 for just the added layers')

opt=parser.parse_args()

if not os.path.isdir(opt.name):
	os.makedirs(opt.name)

opt = parser.load(opt.name+"/args.json")
parser.save(opt.name+"/args.json")

globalStep = tf.Variable(0, name='globalStep', trainable=False)
globalStepInc=tf.assign_add(globalStep,1)

if not os.path.isdir(opt.name+"/log"):
	os.makedirs(opt.name+"/log")

if not os.path.isdir(opt.name+"/save"):
	os.makedirs(opt.name+"/save")

if not os.path.isdir(opt.name+"/preview"):
	os.makedirs(opt.name+"/preview")

#dataset = KaggleFishLoader(opt.dataset, randZoom=opt.randZoom==1)

Model.download()

dataset = CocoDataset(opt.dataset)



images, boxes, classes = Augment.augment(*dataset.get())


print("Number of categories: "+str(dataset.categoryCount()))


net = BoxInceptionResnet(images, dataset.categoryCount(), name="boxnet", trainFrom=opt.trainFrom)

loss = net.getLoss(boxes, classes)
slim.losses.add_loss(loss)

optimizer=tf.train.AdamOptimizer(learning_rate=opt.learningRate, epsilon=opt.adamEps)

def createUpdateOp(gradClip=1):
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	totalLoss = loss #slim.losses.get_total_loss()
	grads = optimizer.compute_gradients(totalLoss, var_list=net.getVariables())
	if gradClip is not None:
		cGrads = []
		for g, v in grads:
			if g is None:
				print("WARNING: no grad for variable "+v.op.name)
				continue
			cGrads.append((tf.clip_by_value(g, -float(gradClip), float(gradClip)), v))
		grads = cGrads

	update_ops.append(optimizer.apply_gradients(grads))
	return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')

#loss = net.get

trainOp=createUpdateOp()

saver=tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=100)




with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
	if not loadCheckpoint(sess, opt.name+"/save/", opt.resume):
		print("Loading GoogleNet")
		net.importWeights(sess, "./inception_resnet_v2_2016_08_30.ckpt")
		#net.importWeights(sess, "initialWeights/", permutateRgb=False)
		print("Done.")

	dataset.startThreads(sess)

	runManager = RunManager(sess)
	runManager.add("train", [globalStepInc,trainOp], modRun=1)


	visualizer = VisualizeOutput.OutputVisualizer(opt, runManager, dataset, net, images, boxes, classes)

	i=1
	cycleCnt=0
	lossSum=0

	while True:
		#run various parts of the network
		res = runManager.modRun(i)
		i, loss=res["train"]

		lossSum+=loss
		cycleCnt+=1

		visualizer.draw(res)

		if i % opt.reportInterval == 0:
			if cycleCnt>0:
				loss=lossSum/cycleCnt

			# lossS=sess.run(trainLossSum, feed_dict={
			# 	trainLossFeed: loss
			# })
			# log.add_summary(lossS, global_step=samplesSeen)

			epoch="%.2f" % (float(i) / dataset.count())
			print("Iteration "+str(i)+" (epoch: "+epoch+"): loss: "+str(loss))
			lossSum=0
			cycleCnt=0

		if i % opt.saveInterval == 0:
			print("Saving checkpoint "+str(i))
			saver.save(sess, opt.name+"/save/model_"+str(i), write_meta_graph=False)
