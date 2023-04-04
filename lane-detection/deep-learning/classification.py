import numpy as np
import pandas as pd
import cv2
# import tensorflow as tf
import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time
from threading import Thread

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



class VideoClassification:
  
  def __init__(self,model_path,src=0):
    self.src=src
    # self.model=tf.keras.models.load_model("models/cat_dog.h5")
    print('fetching model')
    # self.model=get_model_instance_segmentation(2)
    # self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #xm.xla_device() 
    # self.model.to(self.device)
    # self.model.load_state_dict(torch.load('projec\lane detection\weights.pth', map_location=torch.device('cpu')))
    self.model=torch.load(model_path, map_location='cpu')
    self.model.eval()
    self.capture=cv2.VideoCapture(src)
    self._,self.img=self.capture.read()
    self.output=None
    self.label='one'
    self.cTime=0
    self.pTime=0
    self.data_transforms=torchvision.transforms.Compose([
      torchvision.transforms.ToTensor()
    ])
    self.i=0
    
    print('starting threads...')
    #Thread for running the CNN classification of each video frame
    self.t=Thread(target=self.frameClassify)
    self.t.daemon=True 
    self.t.start()
    
    # print('object created')
    
    
  #Thread function to classify each video frame using CNN
  #Heavy video processing functionality should be defined here
  def frameClassify(self):
    print('running model...')
    while True:
      # x=cv2.resize(self.img,(224,224))
      # x=x/255
      x=self.data_transforms(self.img)
      
      result=self.model([x])
      # self.label=self.__prediction((result))
      try:
        rm=result[0]['masks'][0,:,:].permute(1,2,0).detach().numpy()
        cv2.imshow('op_r', np.repeat(rm, 3, axis=2))
      except: continue

      try:
        lm=result[0]['masks'][1,:,:].permute(1,2,0).detach().numpy()
        cv2.imshow('op_l', np.repeat(lm, 3, axis=2))
      except: continue
      # cv2.imwrite(f'op/left_lanes/op_{self.i}.png', lm)
      # cv2.imwrite(f'op/right_lanes/op_{self.i}.png', rm)
      k=cv2.waitKey(1000) & 0xff
      if k==27:
        break 
      self.i+=1
      time.sleep(1/60)
    
    return 
  
  # Running the read/display of the video on thr main thread
  def display(self):
    print('displaying...')
    while True:
      # print('Displaying')
      self.img=cv2.flip(self.capture.read()[1],1)
      self.cTime=time.time()
      fps=1/(self.cTime - self.pTime)
      self.pTime=self.cTime
      # print(self.label)
      # cv2.putText(self.img,"FPS: "+str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
      # cv2.putText(self.img,"Class: "+self.label,(10,110),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
      cv2.imshow('Video',self.img)
      
      k=cv2.waitKey(1000) & 0xff
      if k==27:
        break 
  
    self.capture.release()
      
if __name__=='__main__':
   model_path='D:/Self-Driving-Car/lane-detection/deep-learning/weights/weights.pth' #path to model
   src='D:/Self-Driving-Car/data/videos/output.mp4' #path to video
   video=VideoClassification(model_path=model_path ,src=src)
   try:
     video.display()
   except:
     pass
  
      
  
  
      