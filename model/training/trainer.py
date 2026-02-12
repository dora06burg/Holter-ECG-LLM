import torch
from tqdm import tqdm
from data.stream import ECGStream
from structure.splitter import MultiScaleSplitter

class StreamingTrainer:
    """
    流式训练器 (Streaming Trainer)。
    对应架构图的【幕后调度引擎】。
    
    核心职责：
    1. 管理 24小时超长数据的流式切分。
    2. 维护 Mamba 的流式记忆 (State Management)。
    3. 执行 TBPTT (Truncated Backpropagation Through Time) 反向传播。
    4. 显存防爆控制 (Gradient Checkpointing / Detach)。
    """
    def __init__(self, model, scorer, optimizer, fs, writer, text_encoder, reprogrammer, contrastive_loss):
        self.model = model            # 层次化 ECG Encoder
        self.scorer = scorer          # 异常打分器
        self.optimizer = optimizer    # 优化器
        self.splitter = MultiScaleSplitter(fs) # 心搏切分器
        self.writer = writer          # Tensorboard 日志
        self.step = 0                 # 全局步数
        self.device = next(model.parameters()).device
        
        # 接收从 main_train 传来的 Stage 2 组件
        self.reprogrammer = reprogrammer      # 跨模态翻译官
        self.contrastive_loss = contrastive_loss # 损失函数
        self.text_encoder = text_encoder      # 文本编码器 (Teacher)

    def train_record(self, ecg, text_label=""): 
        """
        训练单一病人的完整 24小时 记录。
        
        Args:
            ecg: 24小时心电数据 (1D Array)
            text_label: 该段心电对应的文本标签 (如 "Atrial Fibrillation")
        """
        # 初始化流式数据迭代器
        stream = ECGStream(ecg, fs=128)
        
        # 【解耦核心 1】：让模型自己去初始化它的专属“记忆体”
        # 这是一个持续存在的变量，会跨越 5分钟 窗口一直传递下去
        # 假设最大序列长度为 150000 (足以覆盖极长范围)，batch_size 为 1
        stream_state = self.model.init_stream_state(max_seqlen=150000, batch_size=1)

        # 【可视化】：tqdm 进度条，显示当前记录处理了百分之多少
        for window in tqdm(stream, desc="Streaming Windows", leave=False):
            # 1. 极速切分当前 5 分钟内的所有心跳
            beats = self.splitter.split(window)
            
            # 过滤掉数据极少（如开头结尾）的无效窗口
            if len(beats) < 3:
                continue
                
            beats = beats.to(self.device)

            # 2. 计算异常得分 (Gating 权重)
            # 这一步不计算梯度 (no_grad)，节省显存
            with torch.no_grad():
                importance = self.scorer.score_batch(beats)

            self.optimizer.zero_grad()
            
            # 3. 前向传播：ECG -> Mamba -> High-dim Features
            # 关键：传入旧的 stream_state，接收更新后的 stream_state
            emb, stream_state = self.model(beats, importance, state=stream_state)
            
            # 4. 【信息压缩与聚合】：Pooling
            # emb shape: [B=1, T=几百个心跳, Dim=128]
            # 沿着时间轴取平均 -> [B=1, Dim=128]
            # 这一步实现了“把 5 分钟压缩成 1 个核心 Token”的思想，同时保留了 Gating 筛选后的异常信息
            global_ecg_emb = emb.mean(dim=1) 
            
            # 5. 跨模态映射：128维数字 -> 768维语义 Token
            mapped_ecg_tokens = self.reprogrammer(global_ecg_emb)
            
            # 6. 获取文本的“标准答案”特征 (Teacher)
            with torch.no_grad():
                # text_label 是字符串，如："患者心率不齐，伴随房颤"
                text_features = self.text_encoder(text_label) 
            
            # 7. 计算对比损失
            loss = self.contrastive_loss(mapped_ecg_tokens, text_features)
            
            loss.backward()
            self.optimizer.step()
            
            # ======== 【极其关键的显存防爆层：剥离计算图】 ========
            # TBPTT (Truncated BPTT) 的核心步骤。
            # 必须使用 .detach() 切断流式记忆的梯度回传链条。
            # 否则，处理第 100 个窗口时，PyTorch 会试图一直反向传播回第 1 个窗口，瞬间爆显存。
            if stream_state is not None:
                if hasattr(stream_state, 'key_value_memory_dict'):
                    # 针对 Mamba 的 InferenceParams 结构进行 detach
                    for k, v in stream_state.key_value_memory_dict.items():
                        stream_state.key_value_memory_dict[k] = (v[0].detach(), v[1].detach())
                elif isinstance(stream_state, list):
                    # 针对 S4 的隐状态 list 进行 detach
                    stream_state = [s.detach() if s is not None else None for s in stream_state]

            # 记录 Loss 曲线
            self.writer.add_scalar("loss/stream", loss.item(), self.step)
            self.step += 1