# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer

class MyVisualizer(Visualizer):
    def _jitter(self, color ):
        return color
    
class MyModel():
    def __init__(self):
        self.cuda0 = torch.device('cuda:0')
        
        self.flag1 = False
        self.flag2 = False
        self.old_old_outputs = None
        self.old_outputs = None

        classes = ['icy_tile', 'icy_asphalt', 'powdery_snow', 'snow_drift', 'puffy_road_snow', 'snowy_road']
        colors = [(255,255,0),(0,0,255),(0,255,0),(255,0,255), (180,165,0), (187,132,156)]

        MetadataCatalog.get("category").set(thing_classes=classes, thing_colors = colors) 

        self.microcontroller_metadata = MetadataCatalog.get("category")

        cfg_instance_seg = get_cfg()
        cfg_instance_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_instance_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg_instance_seg.MODEL.WEIGHTS = os.path.join("..", "output", "model_final.pth")
        cfg_instance_seg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        self.predictor = DefaultPredictor(cfg_instance_seg)

    def instance2semantic(self, outputs):
        classes = []
        masks = []
        for i in range(len(outputs['instances'].pred_classes)):
            for j in range(i, len(outputs['instances'].pred_classes)):
                if outputs['instances'].pred_classes[i] == outputs['instances'].pred_classes[j]:
                    if (int(outputs['instances'].pred_classes[i].cpu()) not in classes):
                        classes.append(int(outputs['instances'].pred_classes[i].cpu()))

        for cls in classes:
            last_mask = None
            #print(cls)
            for i in range(len(outputs['instances'].pred_classes)):
                if cls == int(outputs['instances'].pred_classes[i].cpu()):
                    if last_mask == None:
                        last_mask = outputs['instances'].pred_masks[i]
                    last_mask = torch.logical_or(outputs['instances'].pred_masks[i], last_mask)
            masks.append(last_mask.cpu().numpy().astype('bool'))

        classes = torch.tensor(classes)
        masks = torch.tensor(masks)
        
        new_ouputs = detectron2.structures.instances.Instances(image_size = (1080, 1920))
        new_ouputs.set('pred_classes', classes)
        new_ouputs.set('pred_masks', masks)
        return new_ouputs
        
    def statistic_optimisation(self, new_outputs, old_outputs, old_old_outputs):
        merged_classes = torch.cat((old_old_outputs.pred_classes, old_outputs.pred_classes, new_outputs.pred_classes))
        merged_masks = torch.cat((old_old_outputs.pred_masks, old_outputs.pred_masks, new_outputs.pred_masks))
        
        classes_merged = []
        masks_merged = []

        for i in range(len(merged_classes)):
            for j in range(i, len(merged_classes)):
                if merged_classes[i] == merged_classes[j]:
                    if (int(merged_classes[i].cpu()) not in classes_merged):
                        classes_merged.append(int(merged_classes[i].cpu()))

        for cls in classes_merged:
            last_mask = None
            for i in range(len(merged_classes)):
                if cls == int(merged_classes[i].cpu()):
                    if last_mask == None:
                        last_mask = merged_masks[i]
                    last_mask = torch.logical_or(merged_masks[i], last_mask)
            masks_merged.append(last_mask.to('cpu').numpy().astype('bool'))

        classes_merged = torch.tensor(classes_merged)
        masks_merged = torch.tensor(masks_merged)

        merged_ouputs = detectron2.structures.instances.Instances(image_size = (1080, 1920))
        merged_ouputs.set('pred_classes', classes_merged)
        merged_ouputs.set('pred_masks', masks_merged)
        
        return merged_ouputs
    
    def colission_solver(self, outputs):
        classes = outputs.pred_classes.to('cpu').numpy().astype(int)
        masks = outputs.pred_masks
        for i in range(6):
            if i in classes:
                if i == 1:
                    if 2 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(2)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask )
                if i == 0:
                    if 2 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(2)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 1 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(1)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                if i == 5:
                    if 2 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(2)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 1 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(1)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask) 
                    if 0 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(0)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask) 
                if i == 3:
                    if 2 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(2)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 1 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(1)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask) 
                    if 0 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(0)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 5 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(5)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                if i ==4:
                    if 5 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(5)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 3 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(3)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask) 
                    if 0 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(0)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 1 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(1)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 2 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(2)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
        outputs.pred_masks = masks
        return outputs
    
    def image_segmentation(self, img):
        img = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
        outputs = self.predictor(img)
        if (self.flag1==True) and (self.flag2==True):
            new_outputs = self.statistic_optimisation(outputs['instances'], self.old_outputs, self.old_old_outputs)
            new_outputs = self.colission_solver(new_outputs)
            self.old_old_outputs = self.old_outputs
            self.old_outputs = outputs["instances"]
        else:
            new_outputs = self.instance2semantic(outputs)
            new_outputs = self.colission_solver(new_outputs)
        
        v = MyVisualizer(img[:, :, ::-1],
                        metadata=self.microcontroller_metadata, 
                        scale=0.8, 
                        instance_mode=ColorMode.SEGMENTATION, 
        )
        v = v.draw_instance_predictions(new_outputs.to("cpu"))
        if (self.flag1 == True) and (self.flag2 == False):
            self.flag2 = True
            self.old_outputs = outputs["instances"]
        if self.flag1 == False:
            self.flag1 = True
            self.old_old_outputs = outputs["instances"]

        return v.get_image()[:, :, ::-1]