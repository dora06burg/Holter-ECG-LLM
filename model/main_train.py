import torch
import wfdb
import os
from glob import glob
from tqdm import tqdm
from structure.hierarchical import HierarchicalECGEncoder
from structure.abnormality import LSTMAutoEncoder, AbnormalityScorer
from structure.reprogram import ECGReprogrammer
from training.loss import InfoNCELoss
from training.trainer import StreamingTrainer
from utils.logger import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 实例化所有网络组件并移至 GPU =================
encoder = HierarchicalECGEncoder(emb_dim=128).to(device)
reprogrammer = ECGReprogrammer(ecg_dim=128, llm_dim=768).to(device)
contrastive_loss = InfoNCELoss(initial_temp=0.07).to(device)

ae = LSTMAutoEncoder().to(device)
scorer = AbnormalityScorer(ae)

# ================= 2. 构造 Dummy Text Encoder (用于占位测试) =================
class DummyTextEncoder(torch.nn.Module):
    def forward(self, text):
        # 无论输入什么文本，都返回一个随机的 768 维特征模拟大模型 Token
        # 实际工程中，这里将被替换为真实的 PubMedBERT 等模型
        return torch.randn(1, 768).to(device)

text_encoder = DummyTextEncoder()

# ================= 3. 将所有需要训练的参数打包交给 Optimizer =================
params_to_optimize = (
    list(encoder.parameters()) + 
    list(reprogrammer.parameters()) + 
    list(contrastive_loss.parameters())
)
optimizer = torch.optim.Adam(params_to_optimize, lr=1e-4)

writer = get_logger("runs/stream_ecg")

# ================= 4. 实例化 Trainer =================
trainer = StreamingTrainer(
    model=encoder,
    scorer=scorer,
    optimizer=optimizer,
    fs=128,
    writer=writer,
    text_encoder=text_encoder,        # 传入假的文本编码器
    reprogrammer=reprogrammer,        # 传入翻译官
    contrastive_loss=contrastive_loss # 传入 Loss 函数
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
record_names = loader.get_all_record_names()

for record_name in record_names:
    print(f"\nProcessing Record: {record_name}")
    # 每次只读一条进内存
    record = loader.load_single_record(record_name)

    if record is None:
        print(f"[{record_name}] 加载失败，跳过。")
        continue
        
    ecg = record["signals"][:, 0]
    total_points = len(ecg)
    print(f"[{record_name}] 加载完成！总数据点: {total_points}。启动流式训练...")
    
    # 这里会触发 trainer.py 里面的 tqdm 进度条
    trainer.train_record(ecg)
    
    print(f"[{record_name}] 训练完毕！正在清理内存...")
    # 极其关键：释放内存，防止内存泄漏
    del record
    del ecg