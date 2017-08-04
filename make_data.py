import random,math,io,json,os,sys
import numpy as np
import cv2
import os
import scipy.misc as misc
import tensorflow as tf
import shutil

try:
    num_1 = int(sys.argv[1])
    num_2 = int(sys.argv[2])
except:
    print("Please give num of training and val images , example :")
    print("\t'python make_data.py 60000 2000'")
    exit()


colors_array = np.array([[0,0,0],[255, 0, 0]])

json_data_train = {
    u"categories" :[
        {u"id":1, u"name":u"number",u"supercategory":u"num"}],
    u"annotations":[],
    u"images":[]
}
json_data_val = {
    u"categories" :[
        {u"id":1, u"name":u"number",u"supercategory":u"num"}],
    u"annotations":[],
    u"images":[]
}
def relabel(oracle,colors_array = colors_array):
    return tf.argmin(tf.norm(colors_array-tf.cast(tf.reshape(oracle,[ image_height,
                                                         image_width,1,3]),tf.float64),
                                                             axis = 3),axis=2)
def add_bounding_box(im):
    cont=[]
    im = kk.astype(np.uint8)
    im = im*255
    ret, thresh=cv2.threshold(im, 127, 255, 0)
    p, contours, hierarchy=cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
            cnt=contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h > 20**2:
                cont.append([x-15, y-15, w+15, h+15])
    return cont


data_path = "/media/sterblue/generated_pylone_image_no_change_new_2/"

save_path_train = os.path.abspath("RFCN-tensorflow/data/train2014/")
save_path_val = os.path.abspath("RFCN-tensorflow/data/val2014/")
save_path_annotation = os.path.abspath("RFCN-tensorflow/data/annotations/")

if not os.path.exists(save_path_train):
    os.makedirs(save_path_train)
if not os.path.exists(save_path_annotation):
    os.makedirs(save_path_annotation)
if not os.path.exists(save_path_val):
    os.makedirs(save_path_val)

image_height = 2591
image_width = 1447
ann = tf.placeholder(tf.int32,shape=[None,None],name="ora_im")
oracle = tf.placeholder(
            tf.float32, shape=[None, None, 3],
            name="oracle_image")
n = tf.placeholder(tf.int32,name="num_label")

pred = relabel(oracle)
mask = tf.equal(ann, n)
sess = tf.Session()

print("got here_1")
num_train = 0
num_val = 0
idd_Annot_train = 0
idd_Annot_val = 0
print("Found",len(os.listdir(data_path)),"files ! ")
files = sorted(os.listdir(data_path))
for fc in files:
    if num_val == num_2:
        if num_train == num_1:
            print(
                "\nTraining folder Complete [{} images]\nVal folder Complete [{} images]\n".format(num_train, num_val))
            with io.open(save_path_annotation+"instances_train2014.json", 'w+', encoding='utf-8') as f:
                f.write(json.dumps(json_data_train, ensure_ascii=False))
            print("JSON train svaed sucessfully")
            with io.open(save_path_annotation+"instances_val2014.json", 'w+', encoding='utf-8') as f:
                f.write(json.dumps(json_data_val, ensure_ascii=False))
            print("JSON val svaed sucessfully")
            break
        if fc[]
        if fc[-12:] == 'oracle_2.png':
            output = "Moving training Images : {}".format(num_train)
            sys.stdout.write("\r\x1b[K" + output)
            sys.stdout.flush()
            img = misc.imread(data_path+fc,mode="RGB")
            img = misc.imresize(img,[image_height,image_width])
            #scene = misc.imread(data_path+fc[:-10]+"scene.png",mode="RGB")
            ora = sess.run(pred,feed_dict={oracle:img})
            if len(np.unique(ora)) <= 2:
                num_train+=1
                continue
            shutil.copy(data_path+fc[:-12]+"scene_1.png", save_path_train +
                            "image_" + str(num_train)+'.png')
            json_data_train["images"].append(
                        {
                            u"id": num_train,
                            u"file_name": unicode("image_" + str(num_train)+'.png'),
                            u"width" : image_width,
                            u"height" : image_height
                        }
                    )
            for d in range(4):
                kk = sess.run(mask,feed_dict = {ann:ora,n:d})
                cont = add_bounding_box(kk)
                for z in range(len(cont)):
                    json_data_train["annotations"].append(
                                    {
                                        u"id": idd_Annot_train,
                                        u"category_id": d+1,
                                        u"image_id" : num_train,
                                        u"bbox" : cont[z],
                                        u"iscrowd" : 0
                                    }
                                )
                    idd_Annot_train+=1
            num_train+=1

    else:
        if fc[-12:] == 'oracle_2.png':
            output = "Moving testing Images : {}".format(num_val)
            sys.stdout.write("\r\x1b[K" + output)
            sys.stdout.flush()
            img = misc.imread(data_path+fc,mode="RGB")
            img = misc.imresize(img,[image_height,image_width])
            #scene = misc.imread(data_path+fc[:-10]+"scene.png",mode="RGB")
            ora = sess.run(pred,feed_dict={oracle:img})
            if len(np.unique(ora)) <= 2:
                num_val+=1
                continue
            shutil.copy(data_path+fc[:-12]+"scene_1.png", save_path_val +
                            "image_" + str(num_val)+'.png')
            json_data_val["images"].append(
                        {
                            u"id": num_val,
                            u"file_name": unicode("image_" + str(num_val)+'.png'),
                            u"width" : image_width,
                            u"height" : image_height
                        }
                    )
            for d in range(4):
                kk = sess.run(mask,feed_dict = {ann:ora,n:d})
                cont = add_bounding_box(kk)
                for z in range(len(cont)):
                    json_data_val["annotations"].append(
                                    {
                                        u"id": idd_Annot_val,
                                        u"category_id": d+1,
                                        u"image_id" : num_val,
                                        u"bbox" : cont[z],
                                        u"iscrowd" : 0
                                    }
                                )
                    idd_Annot_val+=1
            num_val+=1
sess.close()
