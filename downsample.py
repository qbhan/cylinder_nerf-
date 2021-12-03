from PIL import Image
from data_loader_split import find_files

def downsample_dir(image_dir='data/360/room2_downscale/test', scale=4):
    pass
    img_files = find_files('{}/rgb'.format(image_dir), exts=['*.png', '*.jpg'])
    for img_dir in img_files:
        print(img_dir)
        img = Image.open(img_dir)
        img = downsample(img, scale)
        img.save(img_dir)    
    
    
    
def downsample(img, scale=4):
    w, h = img.size
    if 'P' in img.mode: # check if image is a palette type
     img = img.convert("RGB") # convert it to RGB
     img = img.resize((w//scale,h//scale),Image.ANTIALIAS) # resize it
     img = img.convert("P",dither=Image.NONE, palette=Image.ADAPTIVE) 
           #convert back to palette
    else:
        img = img.resize((w//scale,h//scale),Image.ANTIALIAS) # regular resize
    return img

downsample_dir('data/360/room4_downscale/test')