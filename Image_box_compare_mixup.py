from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from skimage import exposure
import cv2
import cv2 as cv
import glob
%matplotlib inline

def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''

    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    #img_hsv = img

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img




def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def random_flip(img, bbox, px=0, py=0):
    '''
    Randomly flip the image and correct the bbox.
    param:
    px:
        the probability of horizontal flip
    py:
        the probability of vertical flip
    '''
    height, width = img.shape[:2]
    if np.random.uniform(0, 1) < px:
        img = cv2.flip(img, 1)
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax

    if np.random.uniform(0, 1) < py:
        img = cv2.flip(img, 0)
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    return img, bbox

def resize_with_bbox(img, bbox, new_width, new_height, interp=0, letterbox=False):
    '''
    Resize the image and correct the bbox accordingly.
    '''

    if letterbox:
        image_padded, resize_ratio, dw, dh = letterbox_resize(img, new_width, new_height, interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * resize_ratio + dw
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * resize_ratio + dh

        return image_padded, bbox
    else:
        ori_height, ori_width = img.shape[:2]

        img = cv2.resize(img, (new_width, new_height), interpolation=interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / ori_width * new_width
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / ori_height * new_height

        return img, bbox

###########################mixup#####################################
def get_random_data(annotation_line,annotation_line_1, input_shape, random=True, max_boxes=90, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    line_1 = annotation_line_1.split()
#     print('line:{%s}'%line)
#     print('line_1{%s}'%line_1)
    #image = Image.open(line[0])
    image = cv2.imread(line[0])[:,:,::-1]  #bigger
    #image_1 =Image.open(line_1[0])
    image_1 =cv2.imread(line_1[0])[:,:,::-1] #smaller
    img_array = np.array(image)
    # new image size created
    resize_ratio = 0.5
    h_new = int(image_1.shape[0] * resize_ratio)
    w_new = int(image_1.shape[1] * resize_ratio)
    # resize orignal image to new size image
    img_array_1 = cv2.resize(image_1,(w_new, h_new),interpolation=cv2.INTER_CUBIC)
    img_array_1 = np.array(img_array_1)
    img_array_1 = cv2.GaussianBlur(img_array_1,(3,3),0)
    #c = img_array[330,90]
    #img_array[0:img_array_1.shape[0], 0:img_array_1.shape[1]] = c
    #iw, ih = image.size
    iw, ih = image.shape[0],image.shape[1]
    #iw, ih = iw*0.6,ih*0.6
    #iw_1, ih_1 = image_1.size
    iw_1, ih_1 = image_1.shape[0],image_1.shape[1]
    
    h, w = input_shape # for new h,w , in case should be (416,416)
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    # Another method for above code, please see below
    # boxes = []
    # for box in line[1:]:
    #    box = np.array(list(map(int,box.split(','))))
    #    boxes.append(box)
    # box = np.array(boxes)
    #box = box*resize_ratio
    #print(box.shape) (1,5)
    box_1 = np.array([np.array(list(map(int,box.split(',')))) for box in line_1[1:]])
    box_1 = box_1*resize_ratio
#     height = max(image.shape[0], image_1.shape[0])
#     width = max(image.shape[1], image_1.shape[1])
#     height = max(ih, ih_1)
#     width = max(iw, iw_1)
    height = max(img_array.shape[0], img_array_1.shape[0])
    height = np.array(height)
    width = max(img_array.shape[1], img_array_1.shape[1])
    width = np.array(width)
    mix_img = np.zeros(shape=(height, width, 3), dtype='float32')

    # rand_num = np.random.random()
    rand_num = np.random.beta(1.5, 1.5)
    #rand_num = 1.2
    #rand_num = max(0, min(1, rand_num))
    
    #mix_img = img_array + img_array_1
    #mix_img[:image.shape[0], :image.shape[1], :] = img_array.astype('float32') * rand_num
    mix_img[:img_array.shape[0], :img_array.shape[1], :] = img_array.astype('float32')*0.6
    #mix_img[:img_array.shape[0], :img_array.shape[1], :] = img_array.astype('float32')*6
    
    #mix_img[:image_1.shape[0], :image_1.shape[1], :] += img_array_1.astype('float32') * (1. - rand_num)
    mix_img[:img_array_1.shape[0], :img_array_1.shape[1], :] += img_array_1.astype('float32')* 0.4
    #mix_img[:img_array_1.shape[0], :img_array_1.shape[1], :] = img_array_1.astype('float32')*2
#     alpha = 0.5
#     beta = (1.0 - alpha)
#     mix_img = cv.addWeighted(img_array, alpha, img_array_1, beta, 0.0)
    #dst = np.uint8(alpha*(img1)+beta*(img2))

    mix_img = mix_img.astype('uint8')
    #mix_img = image
    #print(mix_img.shape)
    
    #mix_img = np.resize(mix_img,(1080,1920,3))

    # the last element of the 2nd dimention is the mix up weight
    bbox1 = np.concatenate((box, np.full(shape=(box.shape[0], 1), fill_value=rand_num)), axis=-1)
    bbox2 = np.concatenate((box_1, np.full(shape=(box_1.shape[0], 1), fill_value=1. - rand_num)), axis=-1)
    mix_bbox =np.concatenate((bbox1, bbox2), axis=0)
    #mix_bbox =np.concatenate((box, box_1), axis=0)
    
    #boxes_data = np.zeros((max_boxes,6))
    
    #box_data[:len(mix_bbox)] = mix_bbox # the range 0~len(mix_bbox) of box_data, replace by mix_bbox 
    
    #image_data = random_color_distort(mix_img,brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5)
    image_data = random_color_distort(mix_img,brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5)
    
    image_data, box_data = random_flip(image_data, mix_bbox, px=0.5)
    #image_data, box_data = random_flip(image_data, box_data, px=0.5)
    
    image_data, box_data  = resize_with_bbox(image_data, box_data, w, h, interp=0, letterbox=False)
    
    boxes_data = np.zeros((max_boxes,6))
    
    if len(box_data)>0:
        np.random.shuffle(box_data)
        box_data[:, 0:2][box_data[:, 0:2]<0] = 0
        box_w = box_data[:, 2] - box_data[:, 0]
        box_h = box_data[:, 3] - box_data[:, 1]
        box_data = box_data[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box_data)>max_boxes: box_data = box_data[:max_boxes]
        boxes_data[:len(box_data)] = box_data
        box_data = boxes_data


  
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
    for b in range(n-1):
    #######orignal######
    #for b in range(n):
        if i==0:
            np.random.shuffle(annotation_lines)
        #############mixup###############################################################################
        image, box = get_random_data(annotation_lines[i],annotation_lines[i-1],input_shape, random=True)
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
