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


raw_input_image = Image.open("dog.jpg")                 # load image

orig_x_dim = raw_input_image.size[0]
orig_y_dim = raw_input_image.size[1]
if orig_x_dim > orig_y_dim:
    centercrop_pixels = orig_y_dim
else:
    centercrop_pixels = orig_x_dim

# save square full resolution and square reduced resolution input images for later
square_rullres_input_image = transforms.CenterCrop(centercrop_pixels)(raw_input_image)
square_rullres_input_image_numpy = np.array(square_rullres_input_image)
resize_input_image = resize(raw_input_image)            
resize_input_image_numpy = np.array(resize_input_image)

# proprocess image to feed into image segmentation network
transformed_input_image = transform(raw_input_image)    
batch_input_image = torch.unsqueeze(transformed_input_image, 0)   

# forward pass input image through pre-trained semantic segmentation network
output_dict = fcn(batch_input_image)

out = output_dict['out']                   
print("OUT", out.shape)

seg_map = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy() # reshape output to 2D image
print("SEG MAP", seg_map.shape)

# # function to decode output of semantic segmentation network
# def decode_segmap(image, nc=21):
#   label_colors = np.array([(0, 0, 0),  # 0=background
#                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
#                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
#                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
#                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
#                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
#                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
#   r = np.zeros_like(image).astype(np.uint8)
#   g = np.zeros_like(image).astype(np.uint8)
#   b = np.zeros_like(image).astype(np.uint8)
#   for l in range(0, nc):
#     idx = image == l
#     r[idx] = label_colors[l, 0]
#     g[idx] = label_colors[l, 1]
#     b[idx] = label_colors[l, 2]
#   rgb = np.stack([r, g, b], axis=2)
#   return rgb

# rgb = decode_segmap(seg_map)     # decode output information into visual segmantation map
# print("RGB",rgb.shape)            
# plt.imshow(rgb)             # display segmented image
# plt.show()

print("Numpy Image", resize_input_image_numpy.shape)
# plt.imshow(resize_input_image_numpy)             
# plt.show()

subject_or_not_map = np.copy(resize_input_image_numpy)

subject_or_not_map[seg_map != 12] = 0
subject_or_not_map[seg_map == 12] = 255

plt.imshow(subject_or_not_map)
plt.show()

square_rullres_input_image_numpy = np.array(square_rullres_input_image)
mapping_resized = cv2.resize(subject_or_not_map, 
                             (square_rullres_input_image_numpy.shape[1],
                              square_rullres_input_image_numpy.shape[0]),
                              Image.ANTIALIAS)
print(mapping_resized.shape)

gray = cv2.cvtColor(mapping_resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(15,15),0)
ret3,thresholded_img = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(thresholded_img)
plt.show()


mapping = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2RGB)
np.unique(mapping)

blurred_original_image = cv2.GaussianBlur(square_rullres_input_image_numpy,(251,251),0)
plt.imshow(blurred_original_image)
plt.show()

layered_image = np.where(mapping != (0,0,0), square_rullres_input_image_numpy, blurred_original_image)
plt.imshow(layered_image)
plt.show()
