import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stage2.Qformer import BertConfig, BertLMHeadModel
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class CrossSpaceInteraction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = 6
        self.fuse_s = nn.MultiheadAttention(hidden_size, self.num_heads, batch_first=True)
        self.fuse_l = nn.MultiheadAttention(hidden_size, self.num_heads, batch_first=True)
        self.fuse_se = nn.MultiheadAttention(hidden_size, self.num_heads, batch_first=True)

        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, s, l, se):
        s_fused, _ = self.fuse_s(query=s, key=torch.cat([l, se], dim=1), value=torch.cat([l, se], dim=1))
        s_out = self.norm(s + s_fused)
        s_out = s_out + self.ffn(s_out)

        l_fused, _ = self.fuse_l(query=l, key=torch.cat([s, se], dim=1), value=torch.cat([s, se], dim=1))
        l_out = self.norm(l + l_fused)
        l_out = l_out + self.ffn(l_out)

        se_fused, _ = self.fuse_se(query=se, key=torch.cat([s, l], dim=1), value=torch.cat([s, l], dim=1))
        se_out = self.norm(se + se_fused)
        se_out = se_out + self.ffn(se_out)
        return s_out, l_out, se_out


class InnovativeStage2Bridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config['model'].get('device', 'cuda'))
        model_cfg = config['model']
        self.hidden_size = 768
        self.embed_dim = model_cfg['embed_dim']

        bert_dir = model_cfg['bert_model_dir']
        q_config = BertConfig.from_pretrained(bert_dir, local_files_only=False)
        q_config.encoder_width = self.hidden_size
        q_config.add_cross_attention = True
        q_config.cross_attention_freq = 2
        q_config.gradient_checkpointing = True

        self.Qformer = BertLMHeadModel(config=q_config)
        self.Qformer.bert.encoder.gradient_checkpointing = True

        print(f"PAD ID: {self.Qformer.config.pad_token_id}")
        print(f"BOS/CLS ID: {self.Qformer.config.bos_token_id}")
        print(f"EOS/SEP ID: {self.Qformer.config.eos_token_id}")
        print(f"Vocab Size: {self.Qformer.config.vocab_size}")

        # 2. 保留你的精华：多模态/多空间投影层
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.motif_projs = nn.ModuleDict({
            'structural': nn.Linear(self.hidden_size, self.hidden_size),
            'logical': nn.Linear(self.hidden_size, self.hidden_size),
            'semantic': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        })
        self.fusion_interactor = CrossSpaceInteraction(self.hidden_size)

        self.num_query_token = model_cfg.get('num_query_token', 32)
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_query_token, self.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=q_config.initializer_range)

        self.graph_proj = nn.Linear(self.hidden_size, self.embed_dim)
        self.text_proj = nn.Linear(self.hidden_size, self.embed_dim)
        self.itm_head = nn.Linear(self.hidden_size, 2)

        self.log_temp = nn.Parameter(torch.ones([]) * np.log(0.07))

        # 在 Stage2_AlignmentBridge.py 的 __init__ 末尾添加
        print(f"DEBUG: Qformer gradient checkpointing flag: {self.Qformer.config.gradient_checkpointing}")

        # 检查 Encoder 的具体层是否识别到该属性
        for i, layer in enumerate(self.Qformer.bert.encoder.layer):
            # 实际上属性在 encoder 这一级控制，我们可以直接打印 encoder 的状态
            pass
        print(f"DEBUG: Encoder training state: {self.Qformer.bert.encoder.training}")

    def _prepare_group_mask(self, bs, num_nodes, device):
        """Cross-Attention Mask: 规范 Query 怎么看图"""
        mask = torch.zeros((bs, 1, self.num_query_token, 3 * num_nodes), device=device)
        mask[:, :, 0:10, 0:num_nodes] = 1
        mask[:, :, 10:20, num_nodes:2*num_nodes] = 1
        mask[:, :, 20:30, 2*num_nodes:3*num_nodes] = 1
        mask[:, :, 30:32, :] = 1
        return mask

    def _prepare_self_attn_mask(self, bs, device):
        """
        =================【源头优化：Q-former 组间隔离自注意力掩码】=================
        核心目的：防止结构 Query 和语义 Query 在层层 Self-Attention 中发生跨通道特征走私。
        阻断 0-9、10-19、20-29 组内的横向泄漏，但保持 30-31 全局可见。
        """
        sz = self.num_query_token
        mask = torch.zeros((sz, sz), device=device)

        # 专家组自我闭环通信
        mask[0:10, 0:10] = 1
        mask[10:20, 10:20] = 1
        mask[20:30, 20:30] = 1

        # 30-31 领航组能够读取所有人的信息，且全场都能读取领航组的整合信息
        mask[30:32, :] = 1
        mask[:, 30:32] = 1

        # 广播匹配 RoBERTa 的 [bs, num_heads, q_len, q_len] 架构需求
        # 转为 0 表示允许看，-10000 表示强力遮蔽
        extended_mask = (1.0 - mask).unsqueeze(0).unsqueeze(1).repeat(bs, 1, 1, 1) * -10000.0
        return extended_mask

    def compute_ortho_loss(self, query_outputs):
        """
        =================【源头优化：专家表征空间互斥损失】=================
        提取结构组、逻辑组、语义组的特征平均向量，强行最大化它们在高维流形上的正交角度。
        """
        # query_outputs: [bs, 32, 768]
        q_s = query_outputs[:, 0:10, :].mean(dim=1)  # [bs, 768]
        q_l = query_outputs[:, 10:20, :].mean(dim=1)  # [bs, 768]
        q_se = query_outputs[:, 20:30, :].mean(dim=1)  # [bs, 768]

        # 规范化
        q_s_n = F.normalize(q_s, dim=-1)
        q_l_n = F.normalize(q_l, dim=-1)
        q_se_n = F.normalize(q_se, dim=-1)

        # 两两计算余弦相似度的绝对值，我们希望这个绝对值无限趋近于 0（即彼此正交）
        loss_sl = torch.abs(torch.sum(q_s_n * q_l_n, dim=-1)).mean()
        loss_sse = torch.abs(torch.sum(q_s_n * q_se_n, dim=-1)).mean()
        loss_lse = torch.abs(torch.sum(q_l_n * q_se_n, dim=-1)).mean()

        return loss_sl + loss_sse + loss_lse

    def compute_itc(self, graph_feats, text_feats):
        bs = graph_feats.size(0)
        t = self.log_temp.exp().float()
        sim_q2t = torch.matmul(graph_feats, text_feats.t())
        sim_g2t, _ = sim_q2t.max(dim=1)
        sim_g2t = sim_g2t / t
        sim_t2g, _ = sim_q2t.permute(2, 0, 1).max(dim=-1)
        sim_t2g = sim_t2g / t
        labels = torch.arange(bs, device=self.device)
        return (F.cross_entropy(sim_g2t, labels) + F.cross_entropy(sim_t2g, labels)) / 2

    def encode_graph(self, node_embs):
        h = self.shared_bottleneck(node_embs)
        s = self.motif_projs['structural'](h)
        l = self.motif_projs['logical'](h)
        se = self.motif_projs['semantic'](h)
        s_fused, l_fused, se_fused = self.fusion_interactor(s, l, se)
        return s_fused, l_fused, se_fused

    def forward(self, samples, task='train'):
        s, l, se = self.encode_graph(samples['node_embs'])
        graph_embeds = torch.cat([s, l, se], dim=1)

        bs = graph_embeds.size(0)
        num_nodes = s.size(1)
        device = graph_embeds.device

        group_mask = self._prepare_group_mask(bs, num_nodes, device)
        # 生成强力自注意力硬隔离屏障
        self_attn_mask = self._prepare_self_attn_mask(bs, device)

        # 触发全隔离模式下的 Q-former 扫描
        query_output = self.Qformer.bert(
            query_embeds=self.query_tokens.expand(bs, -1, -1),
            attention_mask=self_attn_mask, # <--- 锁死 Self-Attention 通道
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=group_mask,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
            is_decoder=False
        )
        final_queries = query_output.last_hidden_state
        cross_attentions = query_output.cross_attentions
        first_layer_feat = getattr(query_output, 'first_layer_cross_out', None)

        # --- 任务 A: ITC (使用最后 2 个领航 Query) ---
        pure_pilot_queries = final_queries[:, 30:32, :]
        graph_feat_itc = F.normalize(self.graph_proj(pure_pilot_queries), dim=-1)

        # ================== ✨ 核心修复：显式支持 test 模式 ==================
        if task == 'test':
            return {
                "graph_feat": graph_feat_itc,      # 完美匹配提取脚本所需的单数 key
                "query_output": final_queries,
                "struct_feats": s,
                "logic_feats": l,
                "semantic_feats": se
            }
        # ===================================================================

        text_output = self.Qformer.bert(
            samples['text_input_ids'],
            attention_mask=samples['text_atts'],
            return_dict=True,
        )
        text_feat_itc = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        loss_itc = self.compute_itc(graph_feat_itc, text_feat_itc)

        loss_ortho = self.compute_ortho_loss(final_queries)

        # --- 任务 B: ITM (使用 10-19 号逻辑组 Query + 硬负样本) ---
        # 1. 构造负样本索引 (你提供的逻辑)
        with torch.no_grad():
            sim_q2t = torch.matmul(graph_feat_itc, text_feat_itc.t())
            sim_g2t, _ = sim_q2t.max(dim=1)
            weights_g2t = F.softmax(sim_g2t, dim=1)
            weights_g2t.fill_diagonal_(0)
            hard_neg_idx = torch.multinomial(weights_g2t, 1).squeeze()
            itm_neg_idx = torch.where(torch.rand(bs, device=device) > 0.5, hard_neg_idx,
                                      torch.randperm(bs, device=device))

            current_indices = torch.arange(bs, device=device)
            collision_mask = (itm_neg_idx == current_indices)
            if collision_mask.any():
                itm_neg_idx[collision_mask] = (itm_neg_idx[collision_mask] + 1) % bs

        itm_queries = self.query_tokens[:, [i for i in range(10, 20)] + [30, 31], :].expand(2 * bs, -1, -1)
        itm_group_mask = group_mask[:, :, [i for i in range(10, 20)] + [30, 31], :]
        itm_group_mask_all = torch.cat([itm_group_mask, itm_group_mask], dim=0)

        text_ids_all = torch.cat([samples['text_input_ids'], samples['text_input_ids'][itm_neg_idx]], dim=0)
        text_atts_all = torch.cat([samples['text_atts'], samples['text_atts'][itm_neg_idx]], dim=0)

        itm_query_atts = torch.ones((2 * bs, 12), device=device)
        itm_attention_mask = torch.cat([itm_query_atts, text_atts_all], dim=1)
        graph_embeds_all = torch.cat([graph_embeds, graph_embeds], dim=0)

        # 在 ITM 中也同样注入自注意力组隔离掩码
        txt_len = text_ids_all.size(1)  # 文本序列长度 (例如 512)
        q_len = 12                     # ITM 任务选中的核心 Query 数量 (10个逻辑组 + 2个领航组)
        tot_len = q_len + txt_len      # 混合总长度 (524)

        itm_mixed_mask = torch.zeros((tot_len, tot_len), device=device)
        base_q_mask = self._prepare_self_attn_mask(1, device)[0, 0] # 取出单张 [32, 32] 掩码
        base_q_mask = (base_q_mask < -1000).float() 
        itm_q_sub_mask = base_q_mask[[i for i in range(10, 20)] + [30, 31], :][:, [i for i in range(10, 20)] + [30, 31]]
        itm_mixed_mask[:q_len, :q_len] = itm_q_sub_mask
        itm_self_mask = itm_mixed_mask.unsqueeze(0).unsqueeze(1).repeat(2 * bs, 1, 1, 1)
        txt_pad_mask = (1.0 - text_atts_all).float().unsqueeze(1).unsqueeze(2) # [2*bs, 1, 1, txt_len]
        itm_self_mask[:, :, :q_len, q_len:] = itm_self_mask[:, :, :q_len, q_len:] + txt_pad_mask
        itm_self_mask[:, :, q_len:, q_len:] = itm_self_mask[:, :, q_len:, q_len:] + txt_pad_mask
        itm_self_mask = (itm_self_mask > 0).float() * -10000.0

        itm_output = self.Qformer.bert(
            text_ids_all,
            attention_mask=itm_self_mask,  # <--- 此时形状完美转换为 [2*bs, 1, 524, 524]
            query_embeds=itm_queries,
            encoder_hidden_states=graph_embeds_all,
            encoder_attention_mask=itm_group_mask_all,
            is_decoder=False,
            return_dict=True,
        )
        logic_rep = itm_output.last_hidden_state[:, :10, :].mean(dim=1)
        global_rep = itm_output.last_hidden_state[:, 10:, :].mean(dim=1)
        itm_logit = self.itm_head(logic_rep + global_rep)
        itm_labels = torch.cat([torch.ones(bs, device=device), torch.zeros(bs, device=device)], dim=0).long()
        loss_itm = F.cross_entropy(itm_logit, itm_labels)

        # --- 任务 C: GTG (使用 20-29 号语义组 Query) ---
        labels = samples['text_input_ids'].clone()
        pad_id = self.Qformer.config.pad_token_id
        labels[labels == pad_id] = -100
        labels[:, 0] = -100

        gtg_queries = self.query_tokens[:, [i for i in range(0, 10)] + [i for i in range(20, 32)], :].expand(bs, -1, -1)
        gtg_query_atts = torch.ones((bs, 22), device=device) # 10个结构组 + 12个语义全局组
        gtg_attention_mask = torch.cat([gtg_query_atts, samples['text_atts']], dim=1)
        gtg_group_mask = group_mask[:, :, [i for i in range(0, 10)] + [i for i in range(20, 32)], :]

        gtg_output = self.Qformer(
            samples['text_input_ids'],
            attention_mask=gtg_attention_mask,
            query_embeds=gtg_queries,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=gtg_group_mask,
            labels=labels,
            is_decoder=True,
            return_dict=True,
        )
        loss_gtg = gtg_output.loss

        # --- 为更好获得可视化所做的注意力防御准备 ---
        soft_attentions = None
        if cross_attentions is not None:
            # 保持原始张量结构返回，安全解耦，不破坏前向计算与主损失的梯度分布
            soft_attentions = cross_attentions

        # --- 返回汇总 ---
        return {
            "loss": loss_itc + loss_itm + loss_gtg + loss_ortho,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_gtg": loss_gtg,
            "loss_ortho": loss_ortho,  # 吐给 Trainer 监控
            "graph_feats": graph_feat_itc,
            "text_feats": text_feat_itc,
            "query_output": final_queries,
            "first_layer_feat": first_layer_feat,
            "attentions": soft_attentions,
            "struct_feats": s,
            "logic_feats": l,
            "semantic_feats": se
        }

    @torch.no_grad()
    def generate(self, samples, tokenizer, max_len=128, num_beams=1, length_penalty=1.0, repetition_penalty=1.2,
                 **kwargs):
        """
        配合 Qformer 原生生成器进行代码序列解码
        """
        s, l, se = self.encode_graph(samples['node_embs'])
        graph_context = torch.cat([s, l, se], dim=1)
        bs = graph_context.size(0)
        device = graph_context.device
        num_nodes = s.size(1)

        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        input_ids = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)

        # 选取 20-29 号语义组 Query + 30-31 号全局领航 Query        
        curr_gtg_queries = self.query_tokens[:, [i for i in range(0, 10)] + [i for i in range(20, 32)], :].expand(bs, -1, -1)
        full_group_mask = self._prepare_group_mask(bs, num_nodes, device)
        gtg_group_mask = full_group_mask[:, :, [i for i in range(0, 10)] + [i for i in range(20, 32)], :]

        # 强制将梯度检查点关闭以保护生成器流控
        self.Qformer.bert.config.gradient_checkpointing = False

        # 清理多余可能引发未知报错的冲突项
        kwargs.pop("max_length", None)
        kwargs.pop("min_length", None)

        # 调用我们刚刚注入 Q-former 的原生生成机制
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=curr_gtg_queries,
            encoder_hidden_states=graph_context,
            encoder_attention_mask=gtg_group_mask,
            max_new_tokens=max_len,
            num_beams=num_beams,  # 这里为了跟热力图保持完全同步，默认采用 1 或通过 trainer 控制
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            bos_token_id=bos_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs
        )

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)