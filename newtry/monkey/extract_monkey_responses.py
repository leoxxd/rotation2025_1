"""
猴子数据response提取脚本

根据exclude_area.xls中的ROI信息，从mat文件中提取符合条件的神经元response数据
- 根据pos和y1、y2范围筛选神经元
- 根据reliability_best > 0.4筛选可靠神经元
- 提取前1000张图片的response数据
- 跳过arealabel为"Unknown"的ROI
"""

import pandas as pd
import scipy.io as sio
import numpy as np
import os
import glob
from pathlib import Path

class MonkeyResponseExtractor:
    def __init__(self, monkey_data_dir, exclude_area_path):
        """
        初始化提取器
        
        Args:
            monkey_data_dir: 猴子数据文件夹路径
            exclude_area_path: exclude_area.xls文件路径
        """
        self.monkey_data_dir = monkey_data_dir
        self.exclude_area_path = exclude_area_path
        self.exclude_df = None
        self.extracted_data = {}
        
    def load_exclude_area(self):
        """加载exclude_area.xls文件"""
        try:
            self.exclude_df = pd.read_excel(self.exclude_area_path)
            print(f"成功加载exclude_area.xls，共{len(self.exclude_df)}行数据")
            print("列名:", self.exclude_df.columns.tolist())
            
            # 显示arealabel的分布
            print("\narealabel分布:")
            print(self.exclude_df['AREALABEL'].value_counts())
            
            # 过滤掉Unknown的ROI
            self.exclude_df = self.exclude_df[self.exclude_df['AREALABEL'] != 'Unknown']
            print(f"\n过滤掉Unknown后，剩余{len(self.exclude_df)}行数据")
            
        except Exception as e:
            print(f"加载exclude_area.xls失败: {e}")
            return False
        return True
    
    def get_session_files(self):
        """获取所有session的mat文件"""
        mat_files = glob.glob(os.path.join(self.monkey_data_dir, "Processed_ses*.mat"))
        mat_files.sort()
        print(f"找到{len(mat_files)}个mat文件")
        return mat_files
    
    def extract_session_number(self, filename):
        """从文件名中提取session编号"""
        # 例如: Processed_ses01_240629_M1_2.mat -> 1
        basename = os.path.basename(filename)
        ses_part = basename.split('_')[1]  # ses01
        ses_num = int(ses_part[3:])  # 01 -> 1
        return ses_num
    
    def load_mat_data(self, mat_file):
        """加载单个mat文件的数据"""
        try:
            mat_data = sio.loadmat(mat_file)
            return mat_data
        except Exception as e:
            print(f"加载{mat_file}失败: {e}")
            return None
    
    def find_roi_neurons(self, pos, y1, y2, reliability, reliability_threshold=0.4):
        """
        根据位置和可靠性筛选ROI神经元
        
        Args:
            pos: 神经元位置数组 (1, n_neurons)
            y1: ROI起始位置
            y2: ROI结束位置
            reliability: 可靠性数组 (1, n_neurons)
            reliability_threshold: 可靠性阈值
            
        Returns:
            valid_indices: 符合条件的神经元索引
        """
        # 将pos从(1, n)转换为(n,)
        pos_flat = pos.flatten()
        reliability_flat = reliability.flatten()
        
        # 位置筛选：pos在[y1, y2]范围内
        pos_mask = (pos_flat >= y1) & (pos_flat <= y2)
        
        # 可靠性筛选：reliability > threshold
        # 处理NaN值
        reliability_mask = ~np.isnan(reliability_flat) & (reliability_flat > reliability_threshold)
        
        # 两个条件都满足
        valid_mask = pos_mask & reliability_mask
        valid_indices = np.where(valid_mask)[0]
        
        return valid_indices
    
    def extract_session_responses(self, mat_file, session_rois):
        """
        提取单个session的response数据
        
        Args:
            mat_file: mat文件路径
            session_rois: 该session的ROI信息DataFrame
            
        Returns:
            session_data: 该session的提取数据
        """
        print(f"\n处理文件: {os.path.basename(mat_file)}")
        
        # 加载mat数据
        mat_data = self.load_mat_data(mat_file)
        if mat_data is None:
            return None
        
        # 获取关键数据
        response_best = mat_data['response_best']  # (n_neurons, n_images)
        reliability_best = mat_data['reliability_best']  # (1, n_neurons)
        pos = mat_data['pos']  # (1, n_neurons)
        
        print(f"  数据形状: response_best={response_best.shape}, reliability_best={reliability_best.shape}, pos={pos.shape}")
        
        session_data = {
            'session_file': os.path.basename(mat_file),
            'rois': {}
        }
        
        # 处理每个ROI
        for _, roi_row in session_rois.iterrows():
            roi_index = roi_row['RoiIndex']
            arealabel = roi_row['AREALABEL']
            y1 = roi_row['y1']
            y2 = roi_row['y2']
            
            print(f"  处理ROI {roi_index} ({arealabel}): y范围[{y1}, {y2}]")
            
            # 找到符合条件的神经元
            valid_indices = self.find_roi_neurons(pos, y1, y2, reliability_best)
            
            if len(valid_indices) == 0:
                print(f"    未找到符合条件的神经元")
                continue
            
            print(f"    找到{len(valid_indices)}个符合条件的神经元")
            
            # 提取前1000张图片的response
            roi_responses = response_best[valid_indices, :1000]  # (n_valid_neurons, 1000)
            
            # 存储数据
            session_data['rois'][roi_index] = {
                'arealabel': arealabel,
                'y1': y1,
                'y2': y2,
                'neuron_indices': valid_indices,
                'responses': roi_responses,
                'n_neurons': len(valid_indices)
            }
        
        return session_data
    
    def extract_all_responses(self, reliability_threshold=0.4):
        """
        提取所有session的response数据
        
        Args:
            reliability_threshold: 可靠性阈值
        """
        if not self.load_exclude_area():
            return False
        
        # 获取所有mat文件
        mat_files = self.get_session_files()
        
        print(f"\n开始提取数据，可靠性阈值: {reliability_threshold}")
        
        for mat_file in mat_files:
            # 提取session编号
            session_num = self.extract_session_number(mat_file)
            
            # 获取该session的ROI信息
            session_rois = self.exclude_df[self.exclude_df['SesIdx'] == session_num]
            
            if len(session_rois) == 0:
                print(f"Session {session_num} 没有对应的ROI信息，跳过")
                continue
            
            # 提取该session的response数据
            session_data = self.extract_session_responses(mat_file, session_rois)
            
            if session_data is not None:
                self.extracted_data[session_num] = session_data
        
        print(f"\n数据提取完成，共处理{len(self.extracted_data)}个session")
        return True
    
    def save_extracted_data(self, output_file):
        """
        保存提取的数据
        
        Args:
            output_file: 输出文件路径
        """
        try:
            # 准备保存的数据
            save_data = {
                'extracted_data': self.extracted_data,
                'exclude_area_info': self.exclude_df.to_dict('records'),
                'summary': self.get_summary()
            }
            
            # 保存为pickle文件（推荐，保持数据结构完整）
            import pickle
            pickle_file = output_file.replace('.mat', '.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"数据已保存到: {pickle_file}")
            
            # 保存为mat文件（简化版本，只保存关键数据）
            mat_data = self._prepare_mat_data()
            sio.savemat(output_file, mat_data)
            print(f"简化数据已保存到: {output_file}")
            
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def _prepare_mat_data(self):
        """
        准备用于保存到mat文件的数据（简化版本）
        """
        mat_data = {
            'summary': self.get_summary(),
            'exclude_area_info': self.exclude_df.to_dict('records')
        }
        
        # 将extracted_data转换为更简单的格式
        all_responses = []
        session_info = []
        neuron_mapping = []  # 记录每个神经元属于哪个session和ROI
        
        for session_num, session_data in self.extracted_data.items():
            for roi_index, roi_data in session_data['rois'].items():
                responses = roi_data['responses']
                all_responses.append(responses)
                
                session_info.append({
                    'session': session_num,
                    'roi': roi_index,
                    'arealabel': roi_data['arealabel'],
                    'n_neurons': roi_data['n_neurons'],
                    'y1': roi_data['y1'],
                    'y2': roi_data['y2']
                })
                
                # 为这个ROI的每个神经元记录映射信息
                n_neurons = roi_data['n_neurons']
                for neuron_idx in range(n_neurons):
                    neuron_mapping.append({
                        'session': session_num,
                        'roi': roi_index,
                        'arealabel': roi_data['arealabel'],
                        'neuron_in_roi': neuron_idx,  # 在这个ROI中的索引
                        'neuron_in_combined': len(neuron_mapping)  # 在合并数组中的索引
                    })
        
        # 保存为numpy数组格式
        if all_responses:
            # 计算总神经元数
            total_neurons = sum(resp.shape[0] for resp in all_responses)
            n_images = all_responses[0].shape[1]  # 假设所有response都有相同的图片数
            
            # 创建合并的response数组
            combined_responses = np.zeros((total_neurons, n_images))
            neuron_start_idx = 0
            
            for i, responses in enumerate(all_responses):
                n_neurons = responses.shape[0]
                combined_responses[neuron_start_idx:neuron_start_idx + n_neurons, :] = responses
                neuron_start_idx += n_neurons
            
            # 创建神经元映射数组
            neuron_mapping_array = np.array([
                (mapping['session'], mapping['roi'], mapping['arealabel'], 
                 mapping['neuron_in_roi'], mapping['neuron_in_combined'])
                for mapping in neuron_mapping
            ], dtype=[
                ('session', 'i4'),
                ('roi', 'i4'), 
                ('arealabel', 'U10'),
                ('neuron_in_roi', 'i4'),
                ('neuron_in_combined', 'i4')
            ])
            
            mat_data['combined_responses'] = combined_responses
            mat_data['session_info'] = session_info
            mat_data['neuron_mapping'] = neuron_mapping_array
            mat_data['total_neurons'] = total_neurons
            mat_data['n_images'] = n_images
        
        return mat_data
    
    def get_summary(self):
        """获取数据提取的摘要信息"""
        summary = {
            'total_sessions': len(self.extracted_data),
            'total_rois': 0,
            'total_neurons': 0,
            'session_details': {}
        }
        
        for session_num, session_data in self.extracted_data.items():
            session_neurons = 0
            session_rois = len(session_data['rois'])
            
            for roi_index, roi_data in session_data['rois'].items():
                session_neurons += roi_data['n_neurons']
            
            summary['total_rois'] += session_rois
            summary['total_neurons'] += session_neurons
            
            summary['session_details'][session_num] = {
                'n_rois': session_rois,
                'n_neurons': session_neurons,
                'rois': list(session_data['rois'].keys())
            }
        
        return summary
    
    def print_summary(self):
        """打印数据提取摘要"""
        summary = self.get_summary()
        
        print("\n=== 数据提取摘要 ===")
        print(f"总session数: {summary['total_sessions']}")
        print(f"总ROI数: {summary['total_rois']}")
        print(f"总神经元数: {summary['total_neurons']}")
        
        print("\n各session详情:")
        for session_num, details in summary['session_details'].items():
            print(f"  Session {session_num}: {details['n_rois']}个ROI, {details['n_neurons']}个神经元")
            print(f"    ROI索引: {details['rois']}")
    
    def print_usage_examples(self):
        """打印使用示例"""
        print("\n=== 数据使用示例 ===")
        print("1. 从pickle文件访问完整数据:")
        print("   import pickle")
        print("   with open('extracted_monkey_responses.pkl', 'rb') as f:")
        print("       data = pickle.load(f)")
        print("   session_data = data['extracted_data'][1]  # 获取session 1的数据")
        print("   roi_data = session_data['rois'][3]  # 获取ROI 3的数据")
        print("   responses = roi_data['responses']  # 获取response数据")
        print()
        print("2. 从mat文件使用映射访问数据:")
        print("   import scipy.io as sio")
        print("   mat_data = sio.loadmat('extracted_monkey_responses.mat')")
        print("   combined_responses = mat_data['combined_responses']  # (12581, 1000)")
        print("   neuron_mapping = mat_data['neuron_mapping']  # 神经元映射")
        print("   # 提取session 1的神经元:")
        print("   session_mask = neuron_mapping['session'] == 1")
        print("   session_neurons = combined_responses[session_mask, :]")
        print()
        print("3. 提取特定ROI的数据:")
        print("   roi_mask = neuron_mapping['roi'] == 3")
        print("   roi_neurons = combined_responses[roi_mask, :]")
        print()
        print("4. 按arealabel提取数据:")
        print("   arealabel_mask = neuron_mapping['arealabel'] == 'MB1'")
        print("   mb1_neurons = combined_responses[arealabel_mask, :]")


def main():
    """主函数"""
    # 设置路径
    monkey_data_dir = r"E:\lunzhuan1\rotation2025\Monkey"
    exclude_area_path = r"E:\lunzhuan1\rotation2025\Monkey\exclude_area.xls"
    output_file = r"E:\lunzhuan1\visuo_llm-main\newtry\monkey\extracted_monkey_responses.mat"
    
    # 创建提取器
    extractor = MonkeyResponseExtractor(monkey_data_dir, exclude_area_path)
    
    # 提取数据
    success = extractor.extract_all_responses(reliability_threshold=0.4)
    
    if success:
        # 打印摘要
        extractor.print_summary()
        
        # 保存数据
        extractor.save_extracted_data(output_file)
        
        # 打印使用示例
        extractor.print_usage_examples()
        
        print("\n数据提取完成！")
    else:
        print("数据提取失败！")


if __name__ == "__main__":
    main()
