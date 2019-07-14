from model_PSPMixed import *
from data import *
from parallel_model import *
from keras.callbacks import Callback, EarlyStopping
import numpy as np
import os
import cv2

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# GPU_COUNT = 4



# external
# data_gen_args = dict(rotation_range=0.05,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# internal
data_red_args = dict(rotation_range=0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.05,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(10,'data/membrane/train','image','label',data_red_args,save_to_dir = None)

train_reports = []


model = unet()

class roc_callback(Callback):
    def __init__(self):
    	train_reports = []
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        valid_data_Gene = testGenerator("data/membrane/test_houyushan")
        valid_mask_Gene = GroundTruthFunc("data/membrane/mask")
        y_pred = model.predict_generator(valid_data_Gene, 19, verbose=0)
        y_pred = np.squeeze(y_pred)
        print("len of y_pred {}".format(y_pred.shape))

        y_true = valid_mask_Gene
        y_true = np.squeeze(y_true)
        print("the y_true is like {}".format(y_true.shape))
        iou_total = 0.0
        for i in range(19):
        	y_true_tmp = np.reshape(y_true[i], -1)
        	y_pred_tmp = np.reshape(y_pred[i], -1)
        	# print(y_true_tmp.shape)
        	# val_roc = roc_auc_score(y_true_tmp, y_pred_tmp, sample_weight=None)
        	

        	y_pred_tmp = y_pred_tmp.astype(np.int32)

        	inter = np.bitwise_xor(y_true_tmp, y_pred_tmp)
        	inter_length = inter.shape[0]
        	iou = 1.0 - (np.sum(inter)/(inter_length))
        	iou_total += iou
        	# val_roc_total += val_roc
        # val_roc_total /= 19
        iou /= 19
        # train_reports.append(val_roc_total)
        train_reports.append(iou)
        print("now the iou is {}".format(iou))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def getFileName(path):
    f_list = os.listdir(path)
    name_list = []
    for i in f_list:
        if os.path.splitext(i)[1]=='.bmp':
            name_list.append(os.path.splitext(i)[0])
    return name_list

for e in range(5,8):
    input_dir = 'data/membrane/test_houyushan'
    out_dir = input_dir + str(e)
    out_resize_dir = out_dir + '_512'
    # my_callbacks = EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')
    model_checkpoint = ModelCheckpoint('unet_membrane2_'+str(e)+'.hdf5', monitor='loss',verbose=1, save_best_only=True)
    # model.fit_generator(myGene,steps_per_epoch=600,epochs=e,callbacks=[model_checkpoint, my_callbacks])
    # model.fit_generator(myGene,steps_per_epoch=600,epochs=e,callbacks=[my_callbacks])
    model.fit_generator(myGene,steps_per_epoch=600,epochs=e,callbacks=[model_checkpoint])
    testGene = testGenerator(input_dir,num_image=200, target_size = (256,256))

    folderExist = os.path.exists(out_dir)
    if not folderExist:
        os.makedirs(out_dir)
    folderExist = os.path.exists(out_resize_dir)
    if not folderExist:
        os.makedirs(out_resize_dir)

    results = model.predict_generator(testGene, 197, verbose=1)
    saveResult(out_dir,results)


    # resize from 256 to 512
    name_list = getFileName(out_dir)
    for i in name_list:
        name = out_dir+'/'+i+'.bmp'
        # print(name)
        img = cv2.imread(name)
        # interpolation=cv2.INTER_AREA
        img_resize = cv2.resize(img,(512,512))
        output_name = out_resize_dir+'/'+i+'.bmp'
        # print(output_name)
        cv2.imwrite(output_name,img_resize)

print(train_reports)

# model.fit_generator(myGene,steps_per_epoch=300,epochs=30,callbacks=[model_checkpoint])
# model.fit_generator(myGene,steps_per_epoch=500,epochs=1)

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene, 19, verbose=1)
# saveResult("data/membrane/test",results)