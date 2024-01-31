import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from imageio import imread

import numpy as np


import warnings
warnings.filterwarnings("ignore")
vggnet = torchvision.models.vgg19(pretrained=True)
for p in vggnet.parameters():
      p.requires_grad = False
    
vggnet.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

vggnet.to(device)


def getFeatureMapActs(img,net):
  
  featuremaps = []
  featurenames = []

  convLayerIdx = 0

  for layernum in range(len(net.features)):
    
    img = net.features[layernum](img)

    # store the image if it's a conv2d layer
    if 'Conv2d' in str(net.features[layernum]):
      featuremaps.append( img )
      featurenames.append( 'ConvLayer_' + str(convLayerIdx) )
      convLayerIdx += 1
  
  return featuremaps,featurenames

def gram_matrix(M):
  
  _,chans,height,width = M.shape
  M = M.reshape(chans,height*width)  

  gram = torch.mm(M,M.t()) / (chans*height*width)
  return gram

def transfer(style, content, i):
  # img4content = imread('https://upload.wikimedia.org/wikipedia/commons/6/61/De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg')
  # img4style   = imread('https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg')
  img4content = imread(content)
  img4style = imread(style)
  img4target = np.random.randint(low=0,high=255,size=img4content.shape,dtype=np.uint8)
  
  Ts = T.Compose([ T.ToTensor(),
                   T.Resize(256),
                   T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                 ])
  
  
  img4content = Ts( img4content ).unsqueeze(0).to(device)
  img4style   = Ts( img4style   ).unsqueeze(0).to(device)
  img4target  = Ts( img4target  ).unsqueeze(0).to(device)
  
  contentFeatureMaps,contentFeatureNames = getFeatureMapActs(img4content,vggnet)
  styleFeatureMaps,styleFeatureNames = getFeatureMapActs(img4style,vggnet)

  layers4content = [ 'ConvLayer_1', 'ConvLayer_4', 'ConvLayer_2' ]
  layers4style   = [ 'ConvLayer_1','ConvLayer_2','ConvLayer_3','ConvLayer_4','ConvLayer_5' ]
  weights4style  = [      0.07      ,     .05      ,     .05      ,     .02      ,     .005 ]  
  
  target = img4target.clone()
  target.requires_grad = True
  target = target.to(device)
  styleScaling = 1e6

  numepochs = 1500
  
  optimizer = torch.optim.RMSprop([target],lr=.005)
  
  
  for epochi in range(numepochs):
  
    targetFeatureMaps,targetFeatureNames = getFeatureMapActs(target,vggnet)
  
  
    styleLoss = 0
    contentLoss = 0
  
    for layeri in range(len(targetFeatureNames)):
    
    
      if targetFeatureNames[layeri] in layers4content:
        contentLoss += torch.mean( (targetFeatureMaps[layeri]-contentFeatureMaps[layeri])**2 )
  

      if targetFeatureNames[layeri] in layers4style:
        
        Gtarget = gram_matrix(targetFeatureMaps[layeri])
        Gstyle  = gram_matrix(styleFeatureMaps[layeri])
  
        styleLoss += torch.mean( (Gtarget-Gstyle)**2 ) * weights4style[layers4style.index(targetFeatureNames[layeri])]
  
    
    combiloss = styleScaling*styleLoss + contentLoss

    optimizer.zero_grad()
    combiloss.backward()
    optimizer.step()
  picT = torch.sigmoid(target).cpu().detach().squeeze()
  picT.shape
  save_image(picT, f'output_{i}.png')

def main():
  pics = [
     ("style.jpg", "content1.jpg"),
     ("style.jpg", "content2.png"),
     ("style.jpg", "content3.jpg"),
  ]
  i=1
  for pic in pics:
     transfer(pic[0], pic[1], i)
     i+=1
if __name__ == "__main__":
    main()