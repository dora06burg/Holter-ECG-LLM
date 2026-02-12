import torch
from tqdm import tqdm
from data.stream import ECGStream
from structure.splitter import MultiScaleSplitter

class StreamingTrainer:
    def __init__(self, model, scorer, optimizer, fs, writer, text_encoder, reprogrammer, contrastive_loss):
        self.model = model
        self.scorer = scorer
        self.optimizer = optimizer
        self.splitter = MultiScaleSplitter(fs)
        self.writer = writer
        self.step = 0
        self.device = next(model.parameters()).device
        
        # 接收从 main_train 传来的组件，确保它们都在 optimizer 的监控下
        self.reprogrammer = reprogrammer
        self.contrastive_loss = contrastive_loss
        self.text_encoder = text_encoder # PubMedBERT 或者 ClinicalBERT

    def train_record(self, ecg, text_label=""): # 【新增】：需要传入当前这段心电对应的文本标签
        stream = ECGStream(ecg, fs=128)
        
        # 【解耦核心 1】：让模型自己去初始化它的专属“记忆体”
        # 如果是 Mamba，模型会返回 InferenceParams；如果是 S4，模型会返回 hidden state tuple
        # 假设最大序列长度为 150000，batch_size 为 1
        stream_state = self.model.init_stream_state(max_seqlen=150000, batch_size=1)

        # 【极其关键的可视化修改】：给 stream 套上 tqdm！
        # leave=False 表示这条记录跑完后，进度条会自动消失，保持终端整洁
        for window in tqdm(stream, desc="Streaming Windows", leave=False):
            beats = self.splitter.split(window)
            
            if len(beats) < 3:
                continue
                
            beats = beats.to(self.device)

            with torch.no_grad():
                importance = self.scorer.score_batch(beats)


            self.optimizer.zero_grad()
            
            # 1. Mamba 吐出高维时序特征
            emb, stream_state = self.model(beats, importance, state=stream_state)
            
            # 2. pooling 降维：因为 emb 是序列 [B, T, dim]，
            # 我们对时间维度求平均，得到当前 5 分钟窗口的全局向量 [B, dim]
            global_ecg_emb = emb.mean(dim=1) 
            
            # 3. 通过翻译官映射到 LLM 维度
            mapped_ecg_tokens = self.reprogrammer(global_ecg_emb)
            
            # 4. 提取文本特征 (实际工程中，这步不计算梯度)
            with torch.no_grad():
                # text_label 是字符串，如："患者心率不齐，伴随房颤"
                text_features = self.text_encoder(text_label) 
            
            # 5. 【真正的 Loss】：对比学习，拉近心电与文本的距离
            loss = self.contrastive_loss(mapped_ecg_tokens, text_features)
            
            loss.backward()
            self.optimizer.step()
            
            if stream_state is not None:
                if hasattr(stream_state, 'key_value_memory_dict'):
                    for k, v in stream_state.key_value_memory_dict.items():
                        stream_state.key_value_memory_dict[k] = (v[0].detach(), v[1].detach())
                elif isinstance(stream_state, list):
                    stream_state = [s.detach() if s is not None else None for s in stream_state]

            self.writer.add_scalar("loss/stream", loss.item(), self.step)
            self.step += 1