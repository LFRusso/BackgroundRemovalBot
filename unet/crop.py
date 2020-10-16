'''
    U-Net FCN implementation by zhixuhao (https://github.com/zhixuhao/unet) with a few tweaks 
'''
from . import model as unet_model
import skimage.io as io
import skimage.transform as trans
import numpy as np
import matplotlib.pyplot as plt
import cv2

def prepareImg(image_name,target_size = (256,256),as_gray = True):
    img = io.imread(image_name,as_gray = as_gray)
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,))
    img = np.reshape(img,(1,)+img.shape)
    return img

def crop_img(filename, model):
    img = prepareImg(f"tmp/{filename}.jpg")
    result = model.predict(img)[0]
    result = np.reshape(result, result.shape[:-1])
    result = np.array(255*result, dtype=np.uint8)

    original_img = cv2.imread(f"tmp/{filename}.jpg")
    img_mask = cv2.resize(result, (original_img.shape[1],original_img.shape[0]), interpolation = cv2.INTER_AREA)
    img_mask = cv2.GaussianBlur(img_mask,(7,7),0)

    thresh = 100
    img_mask[img_mask <= thresh] = 0
    img_mask[img_mask > thresh] = 1
    out = cv2.bitwise_and(original_img, original_img, mask=img_mask)
    out = np.dstack([out, 255*img_mask])
    cv2.imwrite(f"tmp/out-{filename}.png",out)
    return