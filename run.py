import torch
from torchvision import models 
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval() # load pretrained semantic segmentation model

resize = transforms.Compose([
 transforms.Resize(256),                    # resize image to 256x256 px 
 transforms.CenterCrop(224)                # center crop img to 224x224 px
])

transform = transforms.Compose([            
 resize,
 transforms.ToTensor(),                     # transform to tensor format
 transforms.Normalize(                      # normalization step --> ImageNet specific values
 mean=[0.485, 0.456, 0.406],                # means for normalization
 std=[0.229, 0.224, 0.225]                  # stdevs for normalization
 )])


raw_input_image = Image.open("image/dog.jpg")                 # load image

orig_x_dim = raw_input_image.size[0]
orig_y_dim = raw_input_image.size[1]
if orig_x_dim > orig_y_dim:
    centercrop_pixels = orig_y_dim
else:
    centercrop_pixels = orig_x_dim

# save square full resolution input image for later
square_rullres_input_image = transforms.CenterCrop(centercrop_pixels)(raw_input_image)
square_rullres_input_image_numpy = np.array(square_rullres_input_image)
im = Image.fromarray(square_rullres_input_image_numpy)
im.save("output/square_rullres_input_image_numpy.jpg")

# save square reduced resolution input image for later
resize_input_image = resize(raw_input_image)            
resize_input_image_numpy = np.array(resize_input_image)
im = Image.fromarray(resize_input_image_numpy)
im.save("output/resize_input_image_numpy.jpg")

# proprocess image to feed into image segmentation network
transformed_input_image = transform(raw_input_image)    
batch_input_image = torch.unsqueeze(transformed_input_image, 0)   

# forward pass input image through pre-trained semantic segmentation network
output_dict = fcn(batch_input_image)
out = output_dict['out']                   
# print("OUT", out.shape)
seg_map = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy() # reshape output to 2D image
# print("SEG MAP", seg_map.shape)

subject_or_not_map = np.copy(resize_input_image_numpy)

subject_or_not_map[seg_map != 12] = 0
subject_or_not_map[seg_map == 12] = 255

im = Image.fromarray(subject_or_not_map)
im.save("output/subject_or_not_map.jpg")

square_rullres_input_image_numpy = np.array(square_rullres_input_image)
mapping_resized = cv2.resize(subject_or_not_map, 
                             (square_rullres_input_image_numpy.shape[1],
                              square_rullres_input_image_numpy.shape[0]),
                              Image.ANTIALIAS)
print(mapping_resized.shape)

gray = cv2.cvtColor(mapping_resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(15,15),0)
ret3,thresholded_img = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
im = Image.fromarray(thresholded_img)
im.save("output/thresholded_img.jpg")


mapping = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2RGB)
np.unique(mapping)

blurred_original_image = cv2.GaussianBlur(square_rullres_input_image_numpy,(251,251),0)
im = Image.fromarray(blurred_original_image)
im.save("output/blurred_original_image.jpg")

layered_image = np.where(mapping != (0,0,0), square_rullres_input_image_numpy, blurred_original_image)
im = Image.fromarray(layered_image)
im.save("output/layered_image.jpg")