# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:30:00 2024

@author: fafsari

"""
import cv2
import numpy as np
import os
import json
import lxml.etree as ET
from .xml_to_json import convert_xml_json

NAMES = ['Saccular Zone']
XML_COLOR = [255, 16776960]#, 65535, 65280, 16711680, 33023]


def xml_suey(wsiMask, args, classNum=2, downsample=1, glob_offset=[0,0]):
    # make xml
    Annotations = xml_create()
    # add annotation
    for i in range(classNum)[1:]: # exclude background class
        Annotations = xml_add_annotation(Annotations=Annotations, annotationID=i)
    
    # unique_mask = []
    # for i in range(0, len(wsiMask), 7000):
    #     unique_mask.extend(np.unique(wsiMask[i:i + 7000]))
    
    for value in np.unique(wsiMask)[1:]:
        # print output
        print('\t working on: annotationID ' + str(int(value)))
        # get only 1 class binary mask
        binary_mask = np.zeros(np.shape(wsiMask),dtype='uint8')
        binary_mask[wsiMask == value] = 1

        # add mask to xml
        pointsList = get_contour_points(binary_mask, args=args, downsample=downsample,value=value,offset={'X':glob_offset[0],'Y':glob_offset[1]})

        for i in range(len(pointsList)):
            pointList = pointsList[i]

            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=int(value))
    gc = args.gc
    annots = convert_xml_json(Annotations, NAMES)
    for annot in annots:
        _ = gc.post(path='annotation',parameters={'itemId':args.item_id}, data = json.dumps(annot))
        print('uploating layers')
    print('annotation uploaded...\n')

def get_contour_points(mask, args, downsample,value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []

    for j in np.array(range(len(maskPoints))):
        if len(maskPoints[j])>2:
            if cv2.contourArea(maskPoints[j]) > 10:
                pointList = []
                for i in np.array(range(0,len(maskPoints[j]),4)):
                    point = {'X': (maskPoints[j][i][0][0] * downsample) + offset['X'], 'Y': (maskPoints[j][i][0][1] * downsample) + offset['Y']}
                    pointList.append(point)
                pointsList.append(pointList)
    return pointsList

### functions for building an xml tree of annotations ###
def xml_create(): # create new xml tree
    # create new xml Tree - Annotations
    Annotations = ET.Element('Annotations')
    return Annotations

def xml_add_annotation(Annotations, annotationID=None): # add new annotation
    # add new Annotation to Annotations
    # defualts to new annotationID
    if annotationID == None: # not specified
        annotationID = len(Annotations.findall('Annotation')) + 1
    # if annotationID in [1,2]:
    #     Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '0', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(XML_COLOR[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    # else:
    Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(XML_COLOR[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    Regions = ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList, annotationID=-1, regionID=None): # add new region to annotation
    # add new Region to Annotation
    # defualts to last annotationID and new regionID
    Annotation = Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
    Regions = Annotation.find('Regions')
    if regionID == None: # not specified
        regionID = len(Regions.findall('Region')) + 1
    Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': '0', 'Id': str(regionID)})
    Vertices = ET.SubElement(Region, 'Vertices')
    for point in pointList: # add new Vertex
        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point['X']), 'Y': str(point['Y']), 'Z': '0'})
    # add connecting point
    ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0]['X']), 'Y': str(pointList[0]['Y']), 'Z': '0'})
    return Annotations
