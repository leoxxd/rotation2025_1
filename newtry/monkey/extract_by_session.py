"""
按session提取猴子数据的简化脚本

专门用于按session提取数据，并提供数据格式检验功能
"""

import pickle
import numpy as np
import os

def load_pkl_data(pkl_file="extracted_monkey_responses.pkl"):
    """加载pkl数据"""
    if not os.path.exists(pkl_file):
        print(f"❌ 文件不存在: {pkl_file}")
        return None
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 成功加载数据文件: {pkl_file}")
        return data
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None

def extract_session_data(data, session_num):
    """
    提取特定session的数据
    
    Args:
        data: 加载的pkl数据
        session_num: session编号
        
    Returns:
        session_data: session数据字典
    """
    if session_num not in data['extracted_data']:
        print(f"❌ Session {session_num} 不存在")
        print(f"可用的session: {list(data['extracted_data'].keys())}")
        return None
    
    session_data = data['extracted_data'][session_num]
    
    print(f"\n=== Session {session_num} 数据 ===")
    print(f"文件: {session_data['session_file']}")
    print(f"ROI数量: {len(session_data['rois'])}")
    
    # 收集所有ROI的response数据
    all_responses = []
    roi_info = []
    
    for roi_index, roi_data in session_data['rois'].items():
        print(f"\nROI {roi_index} ({roi_data['arealabel']}):")
        print(f"  神经元数量: {roi_data['n_neurons']}")
        print(f"  位置范围: [{roi_data['y1']}, {roi_data['y2']}]")
        print(f"  Response形状: {roi_data['responses'].shape}")
        print(f"  数据范围: {roi_data['responses'].min():.3f} 到 {roi_data['responses'].max():.3f}")
        
        all_responses.append(roi_data['responses'])
        roi_info.append({
            'roi_index': roi_index,
            'arealabel': roi_data['arealabel'],
            'n_neurons': roi_data['n_neurons'],
            'y1': roi_data['y1'],
            'y2': roi_data['y2'],
            'responses_shape': roi_data['responses'].shape
        })
    
    # 合并所有ROI的response数据
    if all_responses:
        combined_responses = np.vstack(all_responses)
        print(f"\n合并后的数据:")
        print(f"  总形状: {combined_responses.shape}")
        print(f"  总神经元数: {combined_responses.shape[0]}")
        print(f"  图片数量: {combined_responses.shape[1]}")
        print(f"  数据范围: {combined_responses.min():.3f} 到 {combined_responses.max():.3f}")
        
        return {
            'session_num': session_num,
            'session_file': session_data['session_file'],
            'combined_responses': combined_responses,
            'roi_info': roi_info,
            'individual_responses': all_responses
        }
    
    return None

def save_session_data(session_data, output_file):
    """保存session数据"""
    if session_data is None:
        print("❌ 没有数据可保存")
        return False
    
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(session_data, f)
        print(f"✅ Session {session_data['session_num']} 数据已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def validate_data_format(session_data):
    """
    检验数据格式
    
    Args:
        session_data: session数据字典
        
    Returns:
        validation_result: 验证结果字典
    """
    if session_data is None:
        return {'valid': False, 'errors': ['数据为空']}
    
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'data_info': {}
    }
    
    # 检查必需字段
    required_fields = ['session_num', 'session_file', 'combined_responses', 'roi_info']
    for field in required_fields:
        if field not in session_data:
            validation_result['errors'].append(f"缺少必需字段: {field}")
            validation_result['valid'] = False
    
    if not validation_result['valid']:
        return validation_result
    
    # 检查数据类型
    if not isinstance(session_data['combined_responses'], np.ndarray):
        validation_result['errors'].append("combined_responses 不是numpy数组")
        validation_result['valid'] = False
    
    if not isinstance(session_data['roi_info'], list):
        validation_result['errors'].append("roi_info 不是列表")
        validation_result['valid'] = False
    
    # 检查数据形状
    responses = session_data['combined_responses']
    if len(responses.shape) != 2:
        validation_result['errors'].append(f"combined_responses 形状错误: {responses.shape}，应该是2维")
        validation_result['valid'] = False
    
    # 检查图片数量
    if responses.shape[1] != 1000:
        validation_result['warnings'].append(f"图片数量不是1000: {responses.shape[1]}")
    
    # 检查数据范围
    if np.any(np.isnan(responses)):
        validation_result['warnings'].append("数据中包含NaN值")
    
    if np.any(np.isinf(responses)):
        validation_result['warnings'].append("数据中包含无穷大值")
    
    # 收集数据信息
    validation_result['data_info'] = {
        'session_num': session_data['session_num'],
        'session_file': session_data['session_file'],
        'shape': responses.shape,
        'dtype': responses.dtype,
        'min_value': float(responses.min()),
        'max_value': float(responses.max()),
        'mean_value': float(responses.mean()),
        'std_value': float(responses.std()),
        'n_rois': len(session_data['roi_info']),
        'total_neurons': responses.shape[0],
        'n_images': responses.shape[1]
    }
    
    return validation_result

def print_validation_result(validation_result):
    """打印验证结果"""
    print(f"\n=== 数据格式验证结果 ===")
    
    if validation_result['valid']:
        print("✅ 数据格式验证通过")
    else:
        print("❌ 数据格式验证失败")
        for error in validation_result['errors']:
            print(f"   错误: {error}")
    
    if validation_result['warnings']:
        print("⚠️  警告:")
        for warning in validation_result['warnings']:
            print(f"   {warning}")
    
    print(f"\n数据信息:")
    info = validation_result['data_info']
    print(f"  Session: {info['session_num']}")
    print(f"  文件: {info['session_file']}")
    print(f"  形状: {info['shape']}")
    print(f"  数据类型: {info['dtype']}")
    print(f"  数据范围: {info['min_value']:.3f} 到 {info['max_value']:.3f}")
    print(f"  平均值: {info['mean_value']:.3f}")
    print(f"  标准差: {info['std_value']:.3f}")
    print(f"  ROI数量: {info['n_rois']}")
    print(f"  总神经元数: {info['total_neurons']}")
    print(f"  图片数量: {info['n_images']}")

def main():
    """主函数"""
    print("=== 按Session提取猴子数据 ===")
    
    # 加载数据
    data = load_pkl_data()
    if data is None:
        return
    
    # 显示可用的session
    available_sessions = list(data['extracted_data'].keys())
    print(f"\n可用的session: {available_sessions}")
    
    # 提取特定session的数据
    session_num = 1  # 可以修改这个数字
    print(f"\n提取Session {session_num}的数据...")
    
    session_data = extract_session_data(data, session_num)
    
    if session_data is not None:
        # 验证数据格式
        validation_result = validate_data_format(session_data)
        print_validation_result(validation_result)
        
        # 保存数据
        output_file = f"session_{session_num}_data.pkl"
        save_session_data(session_data, output_file)
        
        # 显示如何使用提取的数据
        print(f"\n=== 如何使用提取的数据 ===")
        print(f"import pickle")
        print(f"import numpy as np")
        print(f"")
        print(f"# 加载数据")
        print(f"with open('{output_file}', 'rb') as f:")
        print(f"    session_data = pickle.load(f)")
        print(f"")
        print(f"# 获取response数据")
        print(f"responses = session_data['combined_responses']  # 形状: {session_data['combined_responses'].shape}")
        print(f"")
        print(f"# 获取ROI信息")
        print(f"roi_info = session_data['roi_info']")
        print(f"for roi in roi_info:")
        print(f"    print(f\"ROI {{roi['roi_index']}} ({{roi['arealabel']}}): {{roi['n_neurons']}}个神经元\")")
        print(f"")
        print(f"# 获取单个ROI的数据")
        print(f"individual_responses = session_data['individual_responses']")
        print(f"roi_0_responses = individual_responses[0]  # 第一个ROI的数据")

if __name__ == "__main__":
    main()
