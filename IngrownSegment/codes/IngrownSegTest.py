# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:30:00 2024

@author: fafsari

"""

import torch
from tqdm import tqdm
import numpy as np
from skimage import filters
from codes.IngrownModels import create_model
from codes.IngrownSegXML import xml_suey   
from skimage.transform import resize
import cv2
import os
import json

TISSUE_JSON = '{"name": "Tissue Ingrowth","description": "Overlaying one item onto another","elements": [{"type": "image","girderId": "None","opacity": 1,"hasAlpha":true,"transform": {"xoffset":0,"yoffset":0}}]}' 
        
def Test_Network(model_path, dataset_valid, test_parameters, args):

    model_details = test_parameters['model_details']
    gc = args.gc
    if 'scaler_means' not in model_details:
        test_parameters['model_details']['scaler_means'] = None       
  
    ann_classes = model_details['ann_classes']
    active = model_details['active']
    target_type = model_details['target_type']

    if active == 'None':
        active = None

    if target_type=='binary':
        n_classes = len(ann_classes)
    elif target_type == 'nonbinary':
        n_classes = 1

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    model = create_model(model_details, n_classes,args)

    state_dict =torch.load(model_path, map_location=device)
        
    # Update the state dict: rename 'combine_layers' to 'segmentation_head'
    if 'combine_layers.weight' in state_dict:
        state_dict['segmentation_head.weight'] = state_dict.pop('combine_layers.weight')
    if 'combine_layers.bias' in state_dict:
        state_dict['segmentation_head.bias'] = state_dict.pop('combine_layers.bias')
    
    # Load the updated state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
      
    print("model successfully loaded!")
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        
        data_iterator = iter(dataset_valid)

        if dataset_valid.patch_batch:
            print('Using patch prediction pipeline')
        else:
            print('Images are the same size as the model inputs')

        with tqdm(range(len(dataset_valid)),desc='Predicting') as pbar:
            for i in range(0,len(dataset_valid.images)):

                try:
                    image, target, input_name = next(data_iterator)
                    # input_name = ''.join(input_name)
                except StopIteration:
                    data_iterator = iter(data_iterator)
                    image, target, input_name = next(data_iterator)
                    # input_name = ''.join(input_name)
                    
                tmp_image = image[None,:,:,:]
                image = torch.transpose(image, 0, 2)
                pred_mask = model(tmp_image.to(device))
                pred_mask = pred_mask[0].squeeze()

                if target_type=='binary':        
                    print('do nothing')

                elif target_type=='nonbinary':
                    pred_mask_img = pred_mask.detach().cpu().numpy()
                    
                    thresh_Li = filters.threshold_li(pred_mask_img)
                    
                    binary_pred_mask = (pred_mask_img > thresh_Li).astype(np.uint8)                    
                    
                    resized_binary_pred_mask = resize(binary_pred_mask, dataset_valid.original_image_size[:2], anti_aliasing=False, order=0)
                                        
                    final_mask = np.zeros(dataset_valid.original_image_size[:2])
                    final_mask[resized_binary_pred_mask > 0] = 1
                    
                    # Read the image and convert to RGB
                    image_rgb = cv2.imread(input_name, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)/255                   
                    
                    sac_zone_image = image_rgb * resized_binary_pred_mask[..., None]
                    zero_indices = sac_zone_image == 0
                    sac_zone_image[zero_indices] = 1
                    resized_sac_zone_image = resize(sac_zone_image, dataset_valid.original_image_size, anti_aliasing=True)                      
                    
                    resized_sac_zone_image_gray_image = np.dot(resized_sac_zone_image[...,:3], [0.2989, 0.5870, 0.1140])
                    #final_mask[resized_sac_zone_image_gray_image < 0.85] = 2
                    
                    final_mask_rgb = np.zeros((dataset_valid.original_image_size[0], dataset_valid.original_image_size[1], 3))
                    final_mask_rgb[final_mask==1] = [0, 255, 0]
                    final_mask_rgb[final_mask==0] = [255, 255, 255]
                                                            
                    ingrown_area_mask = final_mask_rgb.copy()
                    ingrown_area_mask[final_mask==1] = [255, 255, 255]                 
                    
                    ingrown_area_mask = ingrown_area_mask.astype(np.uint8)
                    # Convert to grayscale
                    ingrown_area_mask_gray = np.dot(ingrown_area_mask[...,:3] // 255, [0.2989, 0.5870, 0.1140])                    
                    
                    ingrown_ratio = np.sum(ingrown_area_mask_gray<0.85) / np.sum(resized_binary_pred_mask)
                    print(f'Ingrowing ratio (%): {np.sum(ingrown_area_mask_gray)} / {np.sum(resized_binary_pred_mask)} = {ingrown_ratio * 100}')
                                        
                    # Ensure both inputs are in the same data type and range
                    image_rgb = (image_rgb * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
                    ingrown_area_mask = ingrown_area_mask.astype(np.uint8)  # Ensure mask is uint8

                    overlay_image = image_rgb.copy()
                    overlay_image[resized_sac_zone_image_gray_image < 0.85] = [255, 175, 50]
                                        
                    #  UPLOAD THIS IMAGE TO DSA: overlay_image AND THEN UPLOAD THE JSON FILE
                    overlay_image_name = f'{input_name[:-4]}_overlay_image.jpg'
                    cv2.imwrite(overlay_image_name, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
                                        
                    xml_suey(final_mask, args)

                    upload_response = gc.uploadFileToFolder(args.folder_id,overlay_image_name)
                    print(upload_response['itemId'],'1')

                    TISSUE_JSON_dict = json.loads(TISSUE_JSON)
                    TISSUE_JSON_dict["elements"][0]["girderId"]=upload_response['itemId']
                    
                    print(TISSUE_JSON_dict,'2')
                    _ = gc.post(path='annotation',parameters={'itemId':args.item_id}, data = json.dumps(TISSUE_JSON_dict))
                    print('done')

                pbar.update(1)
