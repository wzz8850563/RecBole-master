# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/29, 2021/7/15
# @Author : Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com

"""
recbole.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader, NegSampleDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType


class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self._set_neg_sample_args(config, dataset, config['MODEL_INPUT_TYPE'], config['train_neg_sample_args'])#设置好参数
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        if self.neg_sample_args['strategy'] == 'by':
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data


class NegSampleEvalDataLoader(NegSampleDataLoader):
    #保证每个用户所有交互数据在同一个batch,并且同一个用户，正样本要在负样本之前
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self._set_neg_sample_args(config, dataset, InputType.POINTWISE, config['eval_neg_sample_args'])
        if self.neg_sample_args['strategy'] == 'by':
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)#按用户进行排序，保证每个用户交互数据相互在一起不分散
            self.uid_list = []#保存所有用户唯一id
            start, end = dict(), dict()
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):#确定每个用户数据的索引开始，结束为止，用于划分数据。用start,end两个字典存储
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)#self.uid2index是数组形式，不是字典，用slice函数保存切片，用法  np.array([])[slice(a，b)]
                self.uid2items_num[uid] = end[uid] - start[uid] + 1 #每个user数据的数量
            self.uid_list = np.array(self.uid_list)

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):#初始化batchsize
        batch_size = self.config['eval_batch_size']
        if self.neg_sample_args['strategy'] == 'by':#逻辑，先排序每个用户数据量，结合设定的‘eval_batch_size’，对
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)#self.uid2items_num * self.times即值表示的数据量变大，如[1,2]*3=[3,6]
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):#循环，找出即不超过预设eval_batch_size，也能极可能涵盖更多用户的最终batchsize，主要确定step,一个step是一个用户，确定self.step即每批次用多少用户全部数据
                if new_batch_size + inters_num[i] > batch_size:#累计的batchsize超过预设的，停止循环，用上一个累计batchsize作为初始
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:#self.neg_sample_args['strategy'] == 'full'
            self.step = batch_size
            self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        if self.neg_sample_args['strategy'] == 'by':
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        self.logger.warnning('NegSampleEvalDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if self.neg_sample_args['strategy'] == 'by':
            uid_list = self.uid_list[self.pr:self.pr + self.step]#
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):#对batch user，循环读取每个user数据，再拼接
                index = self.uid2index[uid]#获取每个用户数据开始结束的index位置
                data_list.append(self._neg_sampling(self.dataset[index]))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]#返回[1,2...self.uid2items_num[uid]* self.times],,反映所有样本的索引位置
                positive_u += [idx for i in range(self.uid2items_num[uid])] #返回[1,2...self.uid2items_num[uid]],反映正样本的索引位置
                positive_i = torch.cat((positive_i, self.dataset[index][self.iid_field]), 0)#正样本itemid

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list))
            positive_u = torch.from_numpy(np.array(positive_u))

            self.pr += self.step

            return cur_data, idx_list, positive_u, positive_i
        else:
            cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
            self.pr += self.step
            return cur_data, None, None, None


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config['MODEL_TYPE'] == ModelType.SEQUENTIAL
        if not self.is_sequential:#不是序列
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            self.uid2positive_item = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)#按userid排好序
            last_uid = None
            positive_item = set()
            uid2used_item = sampler.used_ids#user和所有交互的itemd对应表，类似形式[(1,2),(3),(),(1,5)],索引位置代表userid
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                if uid != last_uid:#遍历数据，当又到了新的userid,执行代码
                    self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                positive_item.add(iid)#保存每个用户所交互的所有item集合
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):#
        if uid is None:
            return
        history_item = used_item - positive_item#找出used_item中不包含positive_item的item集合
        self.uid2positive_item[uid] = torch.tensor(list(positive_item), dtype=torch.int64)#每个user对应的positive_item
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config['eval_batch_size']
        if not self.is_sequential:#非序列
            batch_num = max(batch_size // self.dataset.item_num, 1)#为了保证每个batch能评估所有itemid的前提下，和设置的batch_size尽可能接近，
            new_batch_size = batch_num * self.dataset.item_num #这样使得每次eval都覆盖所有itemid，并且是完整的batch_num被
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    @property
    def pr_end(self):#pr和step是对应以来的，pr用len(self.dataset)，则step用batch_size，pr用len(self.uid_list)，则step用batch_num,表示
        if not self.is_sequential:
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        self.logger.warnning('FullSortEvalDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if not self.is_sequential:
            user_df = self.user_df[self.pr:self.pr + self.step]
            uid_list = list(user_df[self.uid_field])

            history_item = self.uid2history_item[uid_list]
            positive_item = self.uid2positive_item[uid_list]

            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)])
            positive_i = torch.cat(list(positive_item))

            self.pr += self.step
            return user_df, (history_u, history_i), positive_u, positive_i
        else:
            interaction = self.dataset[self.pr:self.pr + self.step]
            inter_num = len(interaction)
            positive_u = torch.arange(inter_num)
            positive_i = interaction[self.iid_field]

            self.pr += self.step
            return interaction, None, positive_u, positive_i
