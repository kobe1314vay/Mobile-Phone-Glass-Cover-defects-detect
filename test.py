# import torch
# from model.MDNet2 import MDNet2
# from model.MDNet import MDNet
# from model.ResNet import resnet34

# def test():
#     pre_model = resnet34()
#     pre_model.load_state_dict(torch.load('./pre_weights/resnet34.pth'))
#     pretrained_dict = pre_model.state_dict()
#     new_model = MDNet2()
#     model_dict = new_model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     new_model.load_state_dict(model_dict)
#     x = torch.rand(1,3,128,128)
#     y = new_model(x)
#     print(y)

# test()
# model = MDNet2()
# for p in model.layer1.parameters():
#     p.requires_grad = False


import cv2
import numpy as np
img = cv2.imread('F:/datasets/MPGCCLASS/IMAGES/defects_10002.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray,None)
cv2.drawKeypoints(gray, kp, img,(0,0,255),4)
cv2.imwrite('./sift_keypoints.jpg',img)

cv2.imshow('img',img)

cv2.waitKey(0)