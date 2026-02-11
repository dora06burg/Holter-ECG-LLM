import torch
import wfdb
import os
from glob import glob
from tqdm import tqdm
from D.hierarchical import HierarchicalECGEncoder
from structure.abnormality import LSTMAutoEncoder, AbnormalityScorer
from training.trainer import StreamingTrainer
from utils.logger import get_logger

# model
encoder = HierarchicalECGEncoder(emb_dim=128)

# anomaly scorer
ae = LSTMAutoEncoder()
scorer = AbnormalityScorer(ae)

optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
writer = get_logger("runs/stream_ecg")

trainer = StreamingTrainer(
    model=encoder,
    scorer=scorer,
    optimizer=optimizer,
    fs=128,
    writer=writer
)

class LTAFDBLoader:
    def __init__(self, data_dir='.'):
        """
        初始化LTAF数据库加载器
        
        Args:
            data_dir: 数据文件所在的目录路径
        """
        self.data_dir = data_dir
        
    def get_all_record_names(self):
        """获取目录中所有的记录名称"""
        hea_files = glob(os.path.join(self.data_dir, "*.hea"))
        record_names = [os.path.basename(f).replace('.hea', '') for f in hea_files]
        # 按数字排序
        record_names.sort(key=lambda x: int(x) if x.isdigit() else x)
        return record_names
    
    def load_single_record(self, record_name):
        """
        加载单个心电记录
        
        Args:
            record_name: 记录名称，如 '00', '01'
            
        Returns:
            dict: 包含信号、元数据和注释的字典
        """
        try:
            record_path = os.path.join(self.data_dir, record_name)
            
            # 读取信号和头文件信息
            signals, fields = wfdb.rdsamp(record_path)
            
            # 读取注释文件
            annotation = wfdb.rdann(record_path, 'atr')
            
            return {
                'record_name': record_name,
                'signals': signals,
                'fields': fields,
                'annotation': annotation,
                'fs': fields['fs'],
                'sig_names': fields['sig_name'],
                'n_samples': signals.shape[0],
                'n_channels': signals.shape[1],
                'duration_hours': signals.shape[0] / fields['fs'] / 3600
            }
            
        except Exception as e:
            print(f"\n加载记录 {record_name} 时出错: {e}")
            return None
    
    def load_all_records(self, limit=None, desc="加载心电记录"):
        """
        使用tqdm进度条加载所有心电记录
        
        Args:
            limit: 可选，限制加载的记录数量
            desc: 进度条描述
            
        Returns:
            list: 包含所有记录数据的列表
        """
        record_names = self.get_all_record_names()
        
        if limit:
            record_names = record_names[:limit]
        
        ecg_records = []
        failed_records = []
        
        print(f"开始加载 {len(record_names)} 条记录...")
        
        # 使用tqdm创建进度条
        for record_name in tqdm(record_names, desc=desc, unit="record", 
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            record_data = self.load_single_record(record_name)
            
            if record_data is not None:
                ecg_records.append(record_data)
            else:
                failed_records.append(record_name)
        
        # 显示统计信息
        if failed_records:
            print(f"\n警告: {len(failed_records)} 条记录加载失败: {failed_records}")
        
        print(f"成功加载 {len(ecg_records)}/{len(record_names)} 条记录")
        if ecg_records:
            total_hours = sum([rec['duration_hours'] for rec in ecg_records])
            print(f"总时长: {total_hours:.2f} 小时")
        
        return ecg_records

data_dir = "ltafdb"
loader = LTAFDBLoader(data_dir)
all_ecg_records = loader.load_all_records()

for record in all_ecg_records:
    ecg = record["signals"][:, 0]
    trainer.train_record(ecg)
