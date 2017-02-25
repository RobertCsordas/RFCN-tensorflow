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

import Dataset.coco.pycocotools.coco as coco
import random
import numpy as np
import cv2
import tensorflow as tf
import threading
import time

class CocoDataset:
	QUEUE_CAPACITY=16

	def __init__(self, path, normalizeSize=True):
		print(path)
		with tf.name_scope('dataset') as scope:
			self.queue = tf.FIFOQueue(dtypes=[tf.float32, tf.float32, tf.uint8],
					capacity=self.QUEUE_CAPACITY)

			self.image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name="image")
			self.boxes = tf.placeholder(dtype=tf.float32, shape=[None,4], name="boxes")
			self.classes = tf.placeholder(dtype=tf.uint8, shape=[None], name="classes")
			
			self.enqueueOp = self.queue.enqueue([self.image, self.boxes, self.classes])

		self.path=path
		self.coco=None
		self.normalizeSize=normalizeSize


	def initCoco(self):
		self.coco=coco.COCO(self.path+"/annotations/instances_train2014.json")
		self.images=self.coco.getImgIds()

		self.toCocoCategory=[]
		self.fromCocoCategory={}

		cats = self.coco.dataset['categories']
		for i in range(len(cats)):
			self.fromCocoCategory[cats[i]["id"]] = i
			self.toCocoCategory.append(cats[i]["id"])

		print("Loaded "+str(len(self.images))+" COCO images")
	
  
	def getCaptions(self, categories):
		if categories is None:
			return None

		res = []
		if isinstance(categories, np.ndarray):
			categories = categories.tolist()

		for c in categories:
			res.append(self.coco.cats[self.toCocoCategory[c]]["name"])

		return res

	def categoryCount(self):
		return 80

	def load(self):
		while True:
			#imgId=self.images[1]
			#imgId=self.images[3456]
			imgId=self.images[random.randint(0, len(self.images)-1)]
	  
			instances = self.coco.loadAnns(self.coco.getAnnIds(imgId, iscrowd=False))
		
			#Ignore crowd images
			crowd = self.coco.loadAnns(self.coco.getAnnIds(imgId, iscrowd=True))
			if len(crowd)>0:
				continue;

			imgFile=self.path+"/train2014/"+self.coco.loadImgs(imgId)[0]["file_name"]
			img = cv2.imread(imgFile)
			
			if img is None:
				print("ERROR: Failed to load "+imgFile)
				continue

			sizeMul = 1.0
			padTop = 0
			padLeft = 0

			if self.normalizeSize:
				sizeMul = 640.0 / min(img.shape[0], img.shape[1])
				img = cv2.resize(img, (int(img.shape[1]*sizeMul), int(img.shape[0]*sizeMul)))

			m = img.shape[1] % 32
			if m != 0:
				padLeft = int(m/2)
				img = img[:,padLeft : padLeft + img.shape[1] - m]

			m = img.shape[0] % 32
			if m != 0:
				m = img.shape[0] % 32
				padTop = int(m/2)
				img = img[padTop : padTop + img.shape[0] - m]

			if img.shape[0]<256 or img.shape[1]<256:
				print("Warning: Image to small, skipping: "+str(img.shape))
				continue

			boxes=[]
			categories=[]
			for i in instances:
				x1,y1,w,h = i["bbox"]
				newBox=[int(x1*sizeMul) - padLeft, int(y1*sizeMul) - padTop, int((x1+w)*sizeMul) - padLeft, int((y1+h)*sizeMul) - padTop]
				newBox[0] = max(min(newBox[0], img.shape[1]),0)
				newBox[1] = max(min(newBox[1], img.shape[0]),0)
				newBox[2] = max(min(newBox[2], img.shape[1]),0)
				newBox[3] = max(min(newBox[3], img.shape[0]),0)

				if (newBox[2]-newBox[0]) >= 16 and (newBox[3]-newBox[1]) >= 16:
					boxes.append(newBox)
					categories.append(self.fromCocoCategory[i["category_id"]])

			if len(boxes)==0:
				print("Warning: No boxes on image. Skipping.")
				continue;

			boxes=np.array(boxes, dtype=np.float32)
			boxes=np.reshape(boxes, [-1,4])
			categories=np.array(categories, dtype=np.uint8)

			
			
			return img, boxes, categories

	def threadFn(self, tid, sess):
		if tid==0:
			self.initCoco()
		else:
			while self.coco is None:
				time.sleep(1)

		while True:
			img, boxes, classes=self.load()
			try:
				sess.run(self.enqueueOp,feed_dict={self.image:img, self.boxes:boxes, self.classes:classes})
			except tf.errors.CancelledError:
				return


	def startThreads(self, sess, nThreads=4):
		self.threads=[]
		for n in range(nThreads):
			t=threading.Thread(target=self.threadFn, args=(n,sess))
			t.daemon = True
			t.start()
			self.threads.append(t)

	def get(self):
		images, boxes, classes = self.queue.dequeue()
		images = tf.expand_dims(images, axis=0)

		images.set_shape([None, None, None, 3])
		boxes.set_shape([None,4])

		return images, boxes, classes

	def count(self):
		return len(self.images)
