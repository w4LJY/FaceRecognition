# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 06:20:19 2021

@author: sandman
"""
import os
import json

Group = 'Muyao4'
name = 'Me'
info = {'sex':'Male','age':'24'}
#info = {'sex':'Female','age':'25'}


path = os.path.join('../featureID',Group,name,'name.json')


with open(path, 'w') as outfile:
    json.dump(info, outfile)

