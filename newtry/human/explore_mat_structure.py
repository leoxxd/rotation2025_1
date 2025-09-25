#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探索MAT文件结构
"""

import numpy as np
from scipy.io import loadmat

def explore_mat_structure():
    """探索MAT文件结构"""
    print("探索MAT文件结构...")
    
    # 加载MAT文件
    data = loadmat('roi_rdm_results/all_subjects_roi_rdms.mat')
    
    print("\n顶层键:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(data[key])}")
    
    # 探索s1数据结构
    print("\n探索s1数据结构:")
    s1_data = data['s1']
    print(f"  类型: {type(s1_data)}")
    print(f"  形状: {s1_data.shape}")
    print(f"  dtype: {s1_data.dtype}")
    
    if s1_data.size > 0:
        print(f"  第一个元素: {s1_data[0, 0]}")
        print(f"  第一个元素类型: {type(s1_data[0, 0])}")
        
        if hasattr(s1_data[0, 0], 'dtype'):
            print(f"  第一个元素dtype: {s1_data[0, 0].dtype}")
        
        if hasattr(s1_data[0, 0], 'shape'):
            print(f"  第一个元素形状: {s1_data[0, 0].shape}")
        
        # 如果是结构化数组，显示字段
        if hasattr(s1_data[0, 0], 'dtype') and s1_data[0, 0].dtype.names:
            print(f"  字段: {s1_data[0, 0].dtype.names}")
            
            # 探索第一个ROI的字段
            first_roi = s1_data[0, 0]
            for field_name in first_roi.dtype.names:
                field_value = first_roi[field_name]
                print(f"    {field_name}: {type(field_value)} - {field_value.shape if hasattr(field_value, 'shape') else 'no shape'}")

if __name__ == "__main__":
    explore_mat_structure()
