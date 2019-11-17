
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

#path_folder = '/media/william/HDD_1TB/AI_competition/train_dataset_70/'
test_folder = '/media/william/HDD_1TB/AI_competition/test_dataset/testdata_2'
# final_image_path = open("image_path_final.txt", "w")
# final_image_label = open("label_path_final.txt", "w")
#xmlAndImage = open("xmlAndImage.txt", "w")
test = open("testdataset_2.txt", "w")

def get_convert_annotation(file,imagepath ):
    tree=ET.parse(file)
#     print(tree)
    root = tree.getroot()
    ImageXml = ""
    
#     classes = ["4710128020106", "8886467102400", "4714431053110", "4710594924427", 
#            "4713507024627", "8801111390064", "4710126041004", "4711162821520", "4710298161234", 
#            "8888077102092", "4710543006693", "7610700600863", "4710126035003", "4710626186519", "8888077101101",
#            "4711402892921", "4719264904219", "4710043001433", "4710105030326", "4710126100923","4710126045460","4710423051096",
#            "4711202224892","4902777062013","4719264904233","4710095046208","04003906",
#            "4710757030200","748675116052","4710174114095"]

    classes = ["4710128020106","8934680033282","80768258","8886467102400","4714431053110","4710035337007","4710105027180","4710594924427"
                ,"4710126043909","4719264904226","4710911022676","4713507024627","8852008300215","8801111390064","8998389162346","4710126041004"
                ,"4711162821520","4710018024207","4710757039104","4710298161234","4710247019845","4712318101244","4716814182011","8888077102092"
            ,"4710126038882","4712972150268","4710543006693","8858674711010","4029787422436","7610700600863","4710126035003","4710626186519","4710421071447"
            ,"8888077101101","9002859055553","8801045571263","4710043001457","4710970847685","4710022102892","4711402892921","4719264904219","4710126046092"
                ,"4710757015207","4710626193012","4710043001433","6970097000563","4710105030326","4710126100923","4710126045460","4710015101291"
                ,"4710423051096","4711202224892","4902777062013","4712693000279","4719264904233","4719264904240","4710370373265","4710095046208"
                 ,"04003906","4710757030200","748675116052","4710247018664","4710126025646","4710126100930","4710015140498","4710126020733","041143025123"
                 ,"80310167","4710174114095","4710452210051"]
    
    
    # to get bounding box data for each image
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        #print(cls)
        if cls not in classes or int(difficult)==1:
        #if cls not in classes:
            continue
        cls_id = classes.index(cls)
        #print(cls_id)
        xmlbox = obj.find('bndbox')
#         b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
#         ImageXml += " " + ",".join([str(a) for a in b]) + ',' + str(cls)
        ImageXml += " " + str(cls)
    test.write(imagepath + ImageXml)     
    test.write('\n')
    #list_file_filename.write('\n')






def imagePathName(imagedir):
    exts = ['.jpg', 'png']
    for root, dirs, files in os.walk(imagedir, topdown=False, followlinks=True):
        level = root.replace(test_folder,'').count(os.sep)
        if level ==0:
            for file in files:
                for ext in exts:
                    if file.endswith(ext):
                        JPGImage = file.split('.')[0]
                        Xmlfile = JPGImage + '.xml'
                        Xmlfile = os.path.join(root, Xmlfile)
                        #compare if xml exist or not
                        if os.path.isfile(Xmlfile):
                            ImageFile = file.split('.')[0]+'.'+file.split('.')[1]
                            #ImageFile = file
                            ImagePath = os.path.join(root, ImageFile)
                            get_convert_annotation(Xmlfile,file)

             

                    
                    
imagePathName(test_folder)