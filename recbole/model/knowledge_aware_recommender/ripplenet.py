# -*- coding: utf-8 -*-
# @Time   : 2020/9/28
# @Author : gaole he
# @Email  : hegaole@ruc.edu.cn

r"""
RippleNet
#####################################################
Reference:
    Hongwei Wang et al. "RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems."
    in CIKM 2018.
"""

import collections

import numpy as np
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class RippleNet(KnowledgeRecommender):
    r"""RippleNet is an knowledge enhanced matrix factorization model.
    The original interaction matrix of :math:`n_{users} \times n_{items}`
    and related knowledge graph is set as model input,
    we carefully design the data interface and use ripple set to train and test efficiently.
    We just implement the model following the original author with a pointwise training mode.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RippleNet, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_weight = config['kg_weight']
        self.reg_weight = config['reg_weight']
        self.n_hop = config['n_hop']
        self.n_memory = config['n_memory']#每层hit最少的实体数
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        head_entities = dataset.head_entities.tolist()
        tail_entities = dataset.tail_entities.tolist()
        relations = dataset.relations.tolist()
        kg = {}
        for i in range(len(head_entities)):
            head_ent = head_entities[i]
            tail_ent = tail_entities[i]
            relation = relations[i]
            kg.setdefault(head_ent, [])
            kg[head_ent].append((tail_ent, relation))#头结点作为key值，（改头结点所有的实体集(tail_ent, relation)作为value），吧交互数据存成{‘head’:[(),()...]}形式
        self.kg = kg
        users = self.interaction_matrix.row.tolist()
        items = self.interaction_matrix.col.tolist()
        user_dict = {}
        for i in range(len(users)):
            user = users[i]
            item = items[i]
            user_dict.setdefault(user, [])
            user_dict[user].append(item)#吧交互数据存成{‘user’:[item1,item2,...]}形式
        self.user_dict = user_dict
        self.ripple_set = self._build_ripple_set()

        # define layers and loss
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)# 一个实体识一个一维向量
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size * self.embedding_size)#一个关系是一个二维向量
        self.transform_matrix = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.rec_loss = BPRLoss()
        self.l2_loss = EmbLoss()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['ripple_set']

    def _build_ripple_set(self):
        r"""Get the normalized interaction matrix of users and items according to A_values.
        Get the ripple hop-wise ripple set for every user, w.r.t. their interaction history

        Returns:
            ripple_set (dict)
        """
        ripple_set = collections.defaultdict(list)
        n_padding = 0 #记录有多少个用户一开始的所有历史item实体都没有实体关系需要padding的
        for user in self.user_dict:
            for h in range(self.n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = self.user_dict[user]  #第一层用用户历史浏览item
                else:
                    #上一个hit的尾实体，作为本次hit头实体,即memories_t，是一个list形式
                    tails_of_last_hop = ripple_set[user][-1][2] #每个用户的实体传播矩阵ripple_set[user]为[(memories_h, memories_r, memories_t),(memories_h, memories_r, memories_t)...],有多少个就有多少个 元祖

                for entity in tails_of_last_hop:#上一层的实体作为头实体找尾实体
                    if entity not in self.kg:
                        continue
                    for tail_and_relation in self.kg[entity]: # tail_and_relation 为以entity为头实体的所有尾实体 ，tail_and_relation形式(tail_ent, relation)
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                # if the current ripple set of the given user is empty,
                # we simply copy the ripple set of the last hop here
                if len(memories_h) == 0:#用户当前hit所有实体都没有关系实体走下去
                    if h == 0:#0-hop
                        # self.logger.info("user {} without 1-hop kg facts, fill with padding".format(user))
                        # raise AssertionError("User without facts in 1st hop")
                        n_padding += 1
                        memories_h = [0 for _ in range(self.n_memory)]
                        memories_r = [0 for _ in range(self.n_memory)]
                        memories_t = [0 for _ in range(self.n_memory)]
                        memories_h = torch.LongTensor(memories_h).to(self.device)
                        memories_r = torch.LongTensor(memories_r).to(self.device)
                        memories_t = torch.LongTensor(memories_t).to(self.device)
                        ripple_set[user].append((memories_h, memories_r, memories_t))
                    else:#1-n -hop
                        ripple_set[user].append(ripple_set[user][-1])#
                else:# 
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < self.n_memory #当前hit实体数少于最低实体数
                    indices = np.random.choice(len(memories_h), size=self.n_memory, replace=replace)#
                    memories_h = [memories_h[i] for i in indices]#本操作目的是保证memories_h长度为self.n_memory，如果超过，则随机不重复抽取self.n_memory个，如果低于，则重复抽取self.n_memory个
                    memories_r = [memories_r[i] for i in indices]#同上
                    memories_t = [memories_t[i] for i in indices]#同上
                    memories_h = torch.LongTensor(memories_h).to(self.device)
                    memories_r = torch.LongTensor(memories_r).to(self.device)
                    memories_t = torch.LongTensor(memories_t).to(self.device)
                    ripple_set[user].append((memories_h, memories_r, memories_t))
        self.logger.info("{} among {} users are padded".format(n_padding, len(self.user_dict)))
        return ripple_set

    def forward(self, interaction):#输入为interaction类
        users = interaction[self.USER_ID].cpu().numpy()
        memories_h, memories_r, memories_t = {}, {}, {}
        for hop in range(self.n_hop):
            memories_h[hop] = []
            memories_r[hop] = []
            memories_t[hop] = []
            for user in users:
                memories_h[hop].append(self.ripple_set[user][hop][0])#memories_h为{'0':[[item1,item2],[item3,item4],...](长度为batchsize),'n':[[item1,item2],[item3,item4],...]}二维形式，其中0,n为hit,[item1,item2]，[item3,item4]分别对应不同user
                memories_r[hop].append(self.ripple_set[user][hop][1])#memories_r为{'0':[[r1,r2],[r3,r4],...],'n':[[r1,r2],[r3,r4],...]}二维形式，其中[item1,item2]，[item3,item4]分别对应不同user
                memories_t[hop].append(self.ripple_set[user][hop][2])
        # memories_h, memories_r, memories_t = self.ripple_set[user]
        item = interaction[self.ITEM_ID] 
        self.item_embeddings = self.entity_embedding(item)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size * n_memory]
            head_ent = torch.cat(memories_h[i], dim=0)#变成一维 （batch size * n_memory，），通过后面self.entity_embedding方便转成向量
            relation = torch.cat(memories_r[i], dim=0)
            tail_ent = torch.cat(memories_t[i], dim=0)
            # self.logger.info("Hop {}, size {}".format(i, head_ent.size(), relation.size(), tail_ent.size()))

            # [batch size * n_memory, dim]
            self.h_emb_list.append(self.entity_embedding(head_ent))#self.h_emb_list最终形态[[batch size * n_memory, dim]（0-hit）,[batch size * n_memory, dim]（1-hit）...]

            # [batch size * n_memory, dim * dim]
            self.r_emb_list.append(self.relation_embedding(relation))#self.h_emb_list最终形态[[batch size * n_memory, dim * dim]（0-hit）,[batch size * n_memory, dim * dim]1-hit）...]

            # [batch size * n_memory, dim]
            self.t_emb_list.append(self.entity_embedding(tail_ent))

        o_list = self._key_addressing()
        #将o相加起来，即得到用户兴趣向量
        y = o_list[-1]
        for i in range(self.n_hop - 1):
            y = y + o_list[i]
        scores = torch.sum(self.item_embeddings * y, dim=1)
        return scores

    def _key_addressing(self):
        r"""Conduct reasoning for specific item and user ripple set

        Returns:
            o_list (dict -> torch.cuda.FloatTensor): list of torch.cuda.FloatTensor n_hop * [batch_size, embedding_size]
        """
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size * n_memory, dim, 1]
            h_emb = self.h_emb_list[hop].unsqueeze(2)

            # [batch_size * n_memory, dim, dim]
            r_mat = self.r_emb_list[hop].view(-1, self.embedding_size, self.embedding_size)
            # [batch_size, n_memory, dim]
            Rh = torch.bmm(r_mat, h_emb).view(-1, self.n_memory, self.embedding_size)#torch.bmm(r_mat, h_emb)结果为 [batch_size * n_memory, dim, 1]，view之后为[batch_size, n_memory, dim]

            # [batch_size, dim, 1]
            v = self.item_embeddings.unsqueeze(2)

            # [batch_size, n_memory]#相似分数
            probs = torch.bmm(Rh, v).squeeze(2)

            # [batch_size, n_memory]
            probs_normalized = self.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = probs_normalized.unsqueeze(2)

            tail_emb = self.t_emb_list[hop].view(-1, self.n_memory, self.embedding_size)#[batch_size, n_memory, dim]

            # [batch_size, dim]
            o = torch.sum(tail_emb * probs_expanded, dim=1)

            self.item_embeddings = self.transform_matrix(self.item_embeddings + o)#更新item_embedding， ？？？？？？
            # item embedding update
            o_list.append(o)
        return o_list

    def calculate_loss(self, interaction):## 能看到fit的时候用的interaction类的那个属性作为数据
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        rec_loss = self.loss(output, label)

        kge_loss = None
        for hop in range(self.n_hop):
            # (batch_size * n_memory, 1, dim)
            h_expanded = self.h_emb_list[hop].unsqueeze(1)
            # (batch_size * n_memory, dim)
            t_expanded = self.t_emb_list[hop]
            # (batch_size * n_memory, dim, dim)
            r_mat = self.r_emb_list[hop].view(-1, self.embedding_size, self.embedding_size)
            # (N, 1, dim) (N, dim, dim) -> (N, 1, dim)
            hR = torch.bmm(h_expanded, r_mat).squeeze(1)
            # (N, dim) (N, dim)
            hRt = torch.sum(hR * t_expanded, dim=1)
            if kge_loss is None:
                kge_loss = torch.mean(self.sigmoid(hRt))
            else:
                kge_loss = kge_loss + torch.mean(self.sigmoid(hRt))

        reg_loss = None
        for hop in range(self.n_hop):
            tp_loss = self.l2_loss(self.h_emb_list[hop], self.t_emb_list[hop], self.r_emb_list[hop])
            if reg_loss is None:
                reg_loss = tp_loss
            else:
                reg_loss = reg_loss + tp_loss
        reg_loss = reg_loss + self.l2_loss(self.transform_matrix.weight)
        loss = rec_loss - self.kg_weight * kge_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        scores = self.forward(interaction)
        return scores

    def _key_addressing_full(self):
        r"""Conduct reasoning for specific item and user ripple set

        Returns:
            o_list (dict -> torch.cuda.FloatTensor): list of torch.cuda.FloatTensor
                n_hop * [batch_size, n_item, embedding_size]
        """
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size * n_memory, dim, 1]
            h_emb = self.h_emb_list[hop].unsqueeze(2)

            # [batch_size * n_memory, dim, dim]
            r_mat = self.r_emb_list[hop].view(-1, self.embedding_size, self.embedding_size)
            # [batch_size, n_memory, dim]
            Rh = torch.bmm(r_mat, h_emb).view(-1, self.n_memory, self.embedding_size)

            batch_size = Rh.size(0)

            if len(self.item_embeddings.size()) == 2:
                # [1, n_item, dim]
                self.item_embeddings = self.item_embeddings.unsqueeze(0)
                # [batch_size, n_item, dim]
                self.item_embeddings = self.item_embeddings.expand(batch_size, -1, -1)
                # [batch_size, dim, n_item]
                v = self.item_embeddings.transpose(1, 2)
                # [batch_size, dim, n_item]
                v = v.expand(batch_size, -1, -1)
            else:
                assert len(self.item_embeddings.size()) == 3
                # [batch_size, dim, n_item]
                v = self.item_embeddings.transpose(1, 2)

            # [batch_size, n_memory, n_item]
            probs = torch.bmm(Rh, v)

            # [batch_size, n_memory, n_item]
            probs_normalized = self.softmax(probs)

            # [batch_size, n_item, n_memory]
            probs_transposed = probs_normalized.transpose(1, 2)

            # [batch_size, n_memory, dim]
            tail_emb = self.t_emb_list[hop].view(-1, self.n_memory, self.embedding_size)

            # [batch_size, n_item, dim]
            o = torch.bmm(probs_transposed, tail_emb)

            # [batch_size, n_item, dim] [batch_size, n_item, dim] -> [batch_size, n_item, dim]
            self.item_embeddings = self.transform_matrix(self.item_embeddings + o)
            # item embedding update
            o_list.append(o)
        return o_list

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID].cpu().numpy()
        memories_h, memories_r, memories_t = {}, {}, {}
        for hop in range(self.n_hop):
            memories_h[hop] = []
            memories_r[hop] = []
            memories_t[hop] = []
            for user in users:
                memories_h[hop].append(self.ripple_set[user][hop][0])
                memories_r[hop].append(self.ripple_set[user][hop][1])
                memories_t[hop].append(self.ripple_set[user][hop][2])
        # memories_h, memories_r, memories_t = self.ripple_set[user]
        # item = interaction[self.ITEM_ID]
        self.item_embeddings = self.entity_embedding.weight[:self.n_items]
        # self.item_embeddings = self.entity_embedding(item)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size * n_memory]
            head_ent = torch.cat(memories_h[i], dim=0)
            relation = torch.cat(memories_r[i], dim=0)
            tail_ent = torch.cat(memories_t[i], dim=0)
            # self.logger.info("Hop {}, size {}".format(i, head_ent.size(), relation.size(), tail_ent.size()))

            # [batch size * n_memory, dim]
            self.h_emb_list.append(self.entity_embedding(head_ent))

            # [batch size * n_memory, dim * dim]
            self.r_emb_list.append(self.relation_embedding(relation))

            # [batch size * n_memory, dim]
            self.t_emb_list.append(self.entity_embedding(tail_ent))

        o_list = self._key_addressing_full()
        y = o_list[-1]
        for i in range(self.n_hop - 1):
            y = y + o_list[i]
        # [batch_size, n_item, dim] [batch_size, n_item, dim]
        scores = torch.sum(self.item_embeddings * y, dim=-1)
        return scores.view(-1)
