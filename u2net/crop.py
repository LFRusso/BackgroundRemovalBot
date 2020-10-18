import os
from skimage import io, transform
from skimage.filters import gaussian
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from u2net.data_loader import RescaleT
from u2net.data_loader import ToTensor
from u2net.data_loader import ToTensorLab
from u2net.data_loader import SalObjDataset

from u2net.model import U2NET # full size version 173.6 MB

import matplotlib.pyplot as plt

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def pred2mask(pred):
    
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    #predict_np /= np.max(np.abs(predict_np), axis=0)
    predict_np[predict_np<0.05] = 0
    predict_np[predict_np!=0] = 1

    return predict_np

def applyMask(img, mask):
    resized_mask = transform.resize(mask, (img.shape[0], img.shape[1]))
    resized_mask = np.array(resized_mask, dtype=np.uint8)

    masked_img = img
    masked_img = np.dstack([img, 255*resized_mask])
    masked_img = Image.fromarray(masked_img).convert('RGBA')
    return masked_img


def crop_img(filename, model):
    # --------- dataloader ---------
    img_name_list = [f"tmp/{filename}.jpg"]

    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    
    # --------- inference for image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        #print("inferencing:",filename)

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        inputs_test = Variable(inputs_test)
        d1,d2,d3,d4,d5,d6,d7= model(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # apply mask to original image
        mask = pred2mask(pred)
        orignal_img = io.imread(img_name_list[i_test])
        masked_img = applyMask(orignal_img, mask)
        masked_img.save(f"tmp/out-{filename}.png")

        del inputs_test
        del d1,d2,d3,d4,d5,d6,d7
        break

    return
