import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

directory  = "C:\\Users\\SUSHANT\\Desktop\\SDP\\UpdatedDataset\\trainset\\mel"
test = "C:\\Users\\SUSHANT\\Desktop\\SDP\\UpdatedDataset\\testset\\mel"
# parameters 
augmentPara=  ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

for image in os.listdir(directory):
    if image.endswith(".jpg"): 
         path = os.path.join(directory, image)
         img = load_img(path) 
         x = img_to_array(img) 
         y = x.reshape((-1, ) + x.shape) 
         i = 0
         for pic in augmentPara.flow(y, batch_size = 32,save_to_dir = test,save_prefix ='ISIC', save_format ='jpg'): 
             i += 1
             if i >=1: 
                 break
         
         
         continue
    else:
         continue





