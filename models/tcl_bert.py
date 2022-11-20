import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TCLBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, h2, h2_pos, h2_neg, adj_ag, pooled_output = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        adj_ag_T = adj_ag.transpose(1, 2)
        # identity = torch.eye(adj_ag.size(1))
        identity = torch.eye(adj_ag.size(1)).cuda()
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag@adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).cuda()
            # ortho[i] += torch.eye(ortho[i].size(0))

        penal = (torch.norm(ortho - identity) / adj_ag.size(0)).cuda()
        # penal = (torch.norm(ortho - identity) / adj_ag.size(0))
        penal = self.opt.alpha * penal
        
        return logits, penal, F.normalize(outputs1, dim=-1), F.normalize(outputs2, dim=-2), \
               F.normalize(h2, dim=-1), F.normalize(h2_neg, dim=-1), F.normalize(h2_pos, dim=-1)


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, src_mask, aspect_mask, dep_tag, pos_mask = inputs
        h1, h2, h2_pos, h2_neg, adj_ag, pooled_output = self.gcn(inputs)
        
        # avg pooling asp feature
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2) 
        outputs1 = (h1*aspect_mask).sum(dim=1) / asp_wn
        outputs2 = (h2*aspect_mask).sum(dim=1) / asp_wn
        return outputs1, outputs2, h2, h2_pos, h2_neg, adj_ag, pooled_output


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        self.deptag_emb = nn.Embedding(opt.deptag_size, opt.deptag_dim,
                                       padding_idx=0) if opt.deptag_dim > 0 else None

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.fc1 = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = opt.deptag_dim if layer == 0 else self.mem_dim
            self.fc1.append(nn.Linear(input_dim, self.mem_dim))

        self.fc2 = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.fc2.append(nn.Linear(input_dim, self.mem_dim))


        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, src_mask, aspect_mask, deptag, pos_mask = inputs
        src_mask = src_mask.unsqueeze(-2)
        
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        adj_ag = None
        
        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag /= self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
            # adj_ag[j] += torch.eye(adj_ag[j].size(0))
        adj_ag = src_mask.transpose(1, 2) * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs
        pos_outputs_dep = gcn_inputs
        neg_outputs_dep = gcn_inputs

        dep_emb = self.deptag_emb(deptag)
        dep_mask = (torch.zeros_like(deptag) != deptag).float()
        final_pos_mask = torch.where(pos_mask != 0, pos_mask, torch.full_like(pos_mask, 1e-6)) * dep_mask
        final_neg_mask = torch.where(pos_mask != 0, torch.full_like(pos_mask, 1e-6), pos_mask) * dep_mask

        dep_tag = dep_emb
        for l in range(self.layers):
            # ************SynGCN*************
            dep_tag = self.fc1[l](dep_tag)
            outputs_dep = self.fc2[l](outputs_dep)
            pos_outputs_dep = self.fc2[l](pos_outputs_dep)
            neg_outputs_dep = self.fc2[l](neg_outputs_dep)

            att = torch.bmm(outputs_dep, dep_tag.transpose(1, 2)) / np.power(outputs_dep.size(-1), 0.5)
            att_pos = torch.bmm(pos_outputs_dep, dep_tag.transpose(1, 2)) / np.power(pos_outputs_dep.size(-1), 0.5)
            att_neg = torch.bmm(pos_outputs_dep, dep_tag.transpose(1, 2)) / np.power(pos_outputs_dep.size(-1), 0.5)

            dep_mask_out = F.softmax(att * dep_mask.unsqueeze(2) + (1 - dep_mask).unsqueeze(2) * (-1e30), dim=-1)
            pos_mask_out = F.softmax(
                att_pos * final_pos_mask.unsqueeze(2) + (1 - final_pos_mask).unsqueeze(2) * (-1e30), dim=-1)
            neg_mask_out = F.softmax(
                att_neg * final_neg_mask.unsqueeze(2) + (1 - final_neg_mask).unsqueeze(2) * (-1e30), dim=-1)

            gAxW_dep = torch.bmm(dep_mask_out, outputs_dep)
            pos_gAxW_dep = torch.bmm(pos_mask_out, pos_outputs_dep)
            neg_gAxW_dep = torch.bmm(neg_mask_out, neg_outputs_dep)

            # ************SemGCN*************
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A1_pos = F.softmax(torch.bmm(torch.matmul(pos_gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)),
                               dim=-1)
            A1_neg = F.softmax(torch.bmm(torch.matmul(neg_gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)),
                               dim=-1)

            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)

            gAxW_dep, pos_gAxW_dep, neg_gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A1_pos, gAxW_ag), \
                                                            torch.bmm(A1_neg, gAxW_ag), torch.bmm(A2, gAxW_dep)

            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            pos_outputs_dep = self.gcn_drop(pos_gAxW_dep) if l < self.layers - 1 else pos_gAxW_dep
            neg_outputs_dep = self.gcn_drop(neg_gAxW_dep) if l < self.layers - 1 else neg_gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        return outputs_ag, outputs_dep, pos_outputs_dep, neg_outputs_dep, adj_ag, pooled_output


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn