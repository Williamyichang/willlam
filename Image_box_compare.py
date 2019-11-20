from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from skimage import exposure
import cv2
import glob
%matplotlib inline





def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


###########################mixup#####################################
def get_random_data(annotation_line, input_shape, random=True, max_boxes=90, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    
    ############# orignal #####################################
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    print(iw)
    print(ih)
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    
#     hisEqualization = rand()<.5
#     if hisEqualization: 
#         image = plt.imread(image, format=np.uint8)
#         image = exposure.equalize_hist(image)

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)
    
    # Adaptive histogram equalization
#     hisEqualization = rand()<.5
#     if hisEqualization:
#         new_image = exposure.equalize_adapthist(image, clip_limit=0.03)
#         image = new_image
    
    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image
    
    
    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
   
    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1
    
    # Adaptive histogram equalization
    AdhisEqualization = rand()<.5
    if AdhisEqualization:
        new_image = exposure.equalize_adapthist(image_data, clip_limit=0.02)
        image_data = new_image
    
    
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data



#############################End#############################################################

train_path = 'testdataset_1110.txt'

#classes_path = 'model_data/classes_20.txt'
input_shape = (416, 416)
# with open(train_path) as f:
#     lines = f.readlines()
#     data_generator(lines,input_shape)
def data_generator(annotation_lines,input_shape):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 1
    image_data = []
    box_data = []
    ######mixup#########
    for b in range(n):
    #######orignal######
    #for b in range(n):
        if i==0:
            np.random.shuffle(annotation_lines)
        #############mixup###############################################################################
        image, box = get_random_data(annotation_lines[i],input_shape, random=True)
        ################orignal##########################################################################
        #image, box = get_random_data(annotation_lines[i],input_shape, random=True)
        image_data.append(image)
        box_data.append(box)
        i = (i+1) % n


    image_data = np.array(image_data)
    box_data = np.array(box_data)
    
    for i, img in enumerate(image_data):
        #filename = "images/file_1110_%d.jpg"%i
        #image = cv2.imread(filename)
        #plotted_img = draw_rect(image, box_data[i])
        plotted_img = draw_rect(img, box_data[i]) # img format aleady had been changed to np.array, so no need to be cv2.imread
        plt.imshow(np.squeeze(plotted_img))
        plt.show()
        #cv2.imwrite(filename, img)
        
    #image_data = image_data[0]
    #image_data = image_data.copy()
    box_data = np.array(box_data)
    
    return image_data, box_data
    
def draw_rect(im, cords, color = None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    im : numpy.ndarray
        numpy image 
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """
    
    im = im.copy()
    #print(cords)
    #cords = cords[:,:4]
    cords = cords[...,0:4]
    #cords = cords[1]
    print(cords.shape)
    #cords = cords.reshape(-1,4)
    if not color:
        color = [255,255,255]
    for cord in cords:
        
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])       
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
    
        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
    return im

# for image in glob.glob('images/*.jpg'):
#     image = cv2.imread(image)[:,:,::-1]
#image = cv2.imread("images/file_1.jpg")[:,:,::-1]
with open(train_path) as f:
    lines = f.readlines()
    np.random.shuffle(lines)
    img, bboxes = data_generator(lines,input_shape)
