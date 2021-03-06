# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : sampler.py

# UPDATE
# @Time   : 2021/7/23, 2020/8/31, 2020/10/6, 2020/9/18, 2021/3/19
# @Author : Xingyu Pan, Kaiyuan Li, Yupeng Hou, Yushuo Chen, Zhichao Feng
# @email  : xy_pan@foxmail.com, tsotfsk@outlook.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, fzcbupt@gmail.com

"""
recbole.sampler
########################
"""

import copy

import numpy as np
from numpy.random import sample
import torch
from collections import Counter


class AbstractSampler(object):
    """:class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports returning
    a certain number of random value_ids according to the input key_id, and it also supports to prohibit
    certain key-value pairs by setting used_ids.

    Args:
        distribution (str): The string of distribution, which is used for subclass.

    Attributes:
        used_ids (numpy.ndarray): The result of :meth:`get_used_ids`.
    """

    def __init__(self, distribution):
        self.distribution = ''
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()

    def set_distribution(self, distribution):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        self.distribution = distribution
        if distribution == 'popularity':
            self._build_alias_table()

    def _uni_sampling(self, sample_num):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            sample_num (int): the number of samples.
        
        Returns:
            sample_list (np.array): a list of samples. 
        """
        raise NotImplementedError('Method [_uni_sampling] should be implemented')

    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()

        Returns:
            candidates_list (list): a list of candidates id.
        """
        raise NotImplementedError('Method [_get_candidates_list] should be implemented')

    def _build_alias_table(self):
        """Build alias table for popularity_biased sampling.
        """
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))#item出现频次的分布，是个dict
        self.alias = self.prob.copy()
        large_q = []
        small_q = []

        for i in self.prob:#i是key值，itemid等
            self.alias[i] = -1 #先全部变成-1
            self.prob[i] = self.prob[i] / len(candidates_list) * len(self.prob)#相当于做了过缩放，使得出现一次的都在0-1的范围，出现2词以上有可能在0-1范围，也有可能>1
            if self.prob[i] > 1:#出现频次大的
                large_q.append(i)
            elif self.prob[i] < 1: #出现频次小的
                small_q.append(i)
        #最终 self.alias中频率少的id对应的值为平频率大的id，代表选到这个id时，换成相应大频率id,并且越大的频率索引就越多的成为小频率索引得值，而对应大频率id值为-1,应该是选到这个id时，直接取。
        while len(large_q) != 0 and len(small_q) != 0:#循环直到large_q或small_q有一个为空，这个过程除了small_q中的索引对应的self.alias变成1外，large_q中的部分相对频率小的索引也会在self.alias变成1
            l = large_q.pop(0)
            s = small_q.pop(0)
            self.alias[s] = l #alias中对应small_q中index的位置的值都变成large_q.pop(0)如类似形式 {23：32，24：-1...}
            self.prob[l] = self.prob[l] - (1 - self.prob[s])#进行此操作后出现频率大的索引概率会减少  0-1de 范围
            #所以每次循环只会large_q或者small_q其中一个少一个元素
            if self.prob[l] < 1:
                small_q.append(l)
            elif self.prob[l] > 1:
                large_q.append(l)

    def _pop_sampling(self, sample_num):#对每个目标id采样对应的多个neg——id,根据热度采样，如有的itemid出现的频率多，相应的概率也会大
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.
        
        Returns:
            sample_list (np.array): a list of samples. 
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)#从0--len(keys)-1 这么多个整数随机采样sample_num个，可能会重复
        random_prob_list = np.random.random(sample_num)#sample_num个0-1的概率值
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:#self.prob越大越可能触发该条件
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])#keys[idx]为小频率id时得对应的大频率id,keys[idx]为大频率id时-1

        return np.array(final_random_list)

    def sampling(self, sample_num):
        """Sampling [sample_num] item_ids.
        
        Args:
            sample_num (int): the number of samples.
        
        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if self.distribution == 'uniform':
            return self._uni_sampling(sample_num)
        elif self.distribution == 'popularity':
            return self._pop_sampling(sample_num)
        else:
            raise NotImplementedError(f'The sampling distribution [{self.distribution}] is not implemented.')

    def get_used_ids(self):# 以kg为例 #每个hid会对应多个tid，类似形式[(1,2),(3),(),(1,5)],索引位置代表headid
        """
        Returns:
            numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.
        """
        raise NotImplementedError('Method [get_used_ids] should be implemented')

    def sample_by_key_ids(self, key_ids, num):#为每个key_id（如itemid）取num个样本
        """Sampling by key_ids.

        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.

        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num #一共要采样的id数
        if (key_ids == key_ids[0]).all():#判断特殊情况，是否key_ids全部一样，或者只有一个
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))#以kg为例，获取hid所对应的所有tid
            value_ids = self.sampling(total_num)#从全部候选中选出了一堆ids
            check_list = np.arange(total_num)[np.isin(value_ids, used)]#找出正样本，check_list为返回正样本itemid
            while len(check_list) > 0:#如果采样有正样本，重新对正样本相同数量采样，至到全部为负样本
                value_ids[check_list] = value = self.sampling(len(check_list))
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:#一般情况
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)#[key_ids[0],key_ids[1]...key_ids[0],key_ids[1]...]重复num
            while len(check_list) > 0:#和上面作用一样，以kg为例，保证只有每个hid采样的tid只有负样本
                value_ids[check_list] = self.sampling(len(check_list))
                check_list = np.array([
                    i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                    if v in used
                ])
        return torch.tensor(value_ids)#返回一维的tensor数组，如tensor([ 865,  393, 1348,  631,   92, 1662])


class Sampler(AbstractSampler):#负采样，要防止训练集的正样本在验证集被采样，训练验证集的正样本被测试集采样,因此一开始把所有数据混合做采样,
    #但是，注意会有eval出现的正样本，被train采样为负样本，见235-239行代码实现，与RepeatableSampler区别
    """:class:`Sampler` is used to sample negative items for each input user. In order to avoid positive items
    in train-phase to be sampled in valid-phase, and positive items in train-phase or valid-phase to be sampled
    in test-phase, we need to input the datasets of all phases for pre-processing. And, before using this sampler,
    it is needed to call :meth:`set_phase` to get the sampler of corresponding phase.

    Args:
        phases (str or list of str): All the phases of input.
        datasets (Dataset or list of Dataset): All the dataset for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.

    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, datasets, distribution='uniform'):#phases, datasets两个列表登场，元素是相对应起来的
        if not isinstance(phases, list):
            phases = [phases]  #'train' or 'evaluation'，见recbole.data.utils.get_dataloader
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError(f'Phases {phases} and datasets {datasets} should have the same length.')

        self.phases = phases
        self.datasets = datasets

        self.uid_field = datasets[0].uid_field
        self.iid_field = datasets[0].iid_field

        self.user_num = datasets[0].user_num
        self.item_num = datasets[0].item_num

        super().__init__(distribution=distribution)

    def _get_candidates_list(self):#把所有的itemid放入candidates_list，包括train,valid,test
        candidates_list = []
        for dataset in self.datasets:
            candidates_list.extend(dataset.inter_feat[self.iid_field].numpy())
        return candidates_list

    def _uni_sampling(self, sample_num):#uniform sampling
        return np.random.randint(1, self.item_num, sample_num)

    def get_used_ids(self):#每个uid会对应多个iid，类似形式[(1,2),(3),(),(1,5)],索引位置代表userid
        """
        Returns:
            dict: Used item_ids is the same as positive item_ids.
            Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        used_item_id = dict()
        last = [set() for _ in range(self.user_num)]
        for phase, dataset in zip(self.phases, self.datasets):#used_item_id按phase分开，即trian统计本数据，valid统计train,valid数据，test统计所有数据
            cur = np.array([set(s) for s in last])#用set(s)去掉重复，从循环到第二个phase, dataset开始有作用
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                cur[uid].add(iid)#cur类似[(1,2，1),(3),(),(1,5)],itemid可能有重复如(1,2，1)，索引位置代表userid
            last = used_item_id[phase] = cur

        for used_item_set in used_item_id[self.phases[-1]]:
            if len(used_item_set) + 1 == self.item_num:  # [pad] is a item.
                raise ValueError(
                    'Some users have interacted with all items, '
                    'which we can not sample negative items for them. '
                    'Please set `user_inter_num_interval` to filter those users.'
                )
        return used_item_id

    def set_phase(self, phase):#由于Sampler一次传入trian,valid,test等一起处理，该函数用于保留具体某一个
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, :attr:`phase` is set the same as input phase, and :attr:`used_ids`
            is set to the value of corresponding phase.
        """
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(user_ids, num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')


class KGSampler(AbstractSampler):
    """:class:`KGSampler` is used to sample negative entities in a knowledge graph.

    Args:
        dataset (Dataset): The knowledge graph dataset, which contains triplets in a knowledge graph.
        distribution (str, optional): Distribution of the negative entities. Defaults to 'uniform'.
    """

    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset

        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field
        self.hid_list = dataset.head_entities
        self.tid_list = dataset.tail_entities

        self.head_entities = set(dataset.head_entities)
        self.entity_num = dataset.entity_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.entity_num, sample_num)

    def _get_candidates_list(self):
        return list(self.hid_list) + list(self.tid_list)

    def get_used_ids(self):#获取每个hid都对应了那些tid
        """
        Returns:
            numpy.ndarray: Used entity_ids is the same as tail_entity_ids in knowledge graph.
            Index is head_entity_id, and element is a set of tail_entity_ids.
        """
        used_tail_entity_id = np.array([set() for _ in range(self.entity_num)])
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)  #每个hid会对应多个tid，类似形式[(1,2),(3),(),(1,5)],索引位置代表headid

        for used_tail_set in used_tail_entity_id:
            if len(used_tail_set) + 1 == self.entity_num:  # [pad] is a entity.
                raise ValueError(
                    'Some head entities have relation with all entities, '
                    'which we can not sample negative entities for them.'
                )
        return used_tail_entity_id

    def sample_by_entity_ids(self, head_entity_ids, num=1):#每个head_entity_id采样的 entity_id都不一样，因为都是以
        #如果head_entity_ids长度为5，num=2，则entity_ids形式[[0，5,10],[1,6,11],[2,7,12],[3,8,13],[4,9,14]]
        """Sampling by head_entity_ids.

        Args:
            head_entity_ids (numpy.ndarray or list): Input head_entity_ids.
            num (int, optional): Number of sampled entity_ids for each head_entity_id. Defaults to ``1``.

        Returns:
            torch.tensor: Sampled entity_ids.
            entity_ids[0], entity_ids[len(head_entity_ids)], entity_ids[len(head_entity_ids) * 2], ...,
            entity_id[len(head_entity_ids) * (num - 1)] is sampled for head_entity_ids[0];
            entity_ids[1], entity_ids[len(head_entity_ids) + 1], entity_ids[len(head_entity_ids) * 2 + 1], ...,
            entity_id[len(head_entity_ids) * (num - 1) + 1] is sampled for head_entity_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(head_entity_ids, num)
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError(f'head_entity_id [{head_entity_id}] not exist.')


class RepeatableSampler(AbstractSampler):#主要用于seq,因为used_id的逻辑是，由于每行是一个序列并不是userid-itemid对，所以每行的负样本是所有其他itemid,除了seq的label.
    """:class:`RepeatableSampler` is used to sample negative items for each input user. The difference from
    :class:`Sampler` is it can only sampling the items that have not appeared at all phases.

    Args:
        phases (str or list of str): All the phases of input.
        dataset (Dataset): The union of all datasets for each phase.              ##与Sampler区别，Sampler是分开的datasets
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.

    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, dataset, distribution='uniform'):
        if not isinstance(phases, list):
            phases = [phases]
        self.phases = phases
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].numpy())

    def get_used_ids(self):#
        """
        Returns:
            numpy.ndarray: Used item_ids is the same as positive item_ids.
            Index is user_id, and element is a set of item_ids.
        """
        return np.array([set() for _ in range(self.user_num)])

    def sample_by_user_ids(self, user_ids, item_ids, num):#user_ids, item_ids 并非唯一，所以注意self.used_ids的构建都是本行itemid,所以每行负样本除了本行itemid其他都可以，适合用于seq序列，其他的模型需要另外在看前情况
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            self.used_ids = np.array([{i} for i in item_ids])
            return self.sample_by_key_ids(np.arange(len(user_ids)), num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')

    def set_phase(self, phase):
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, and :attr:`phase` is set the same as input phase.
        """
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        return new_sampler


class SeqSampler(AbstractSampler):#目前看和RepeatableSampler逻辑基本一样，只是这里只采样1个
    """:class:`SeqSampler` is used to sample negative item sequence.

        Args:
            datasets (Dataset or list of Dataset): All the dataset for each phase.
            distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.
    """

    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def get_used_ids(self):
        pass

    def sample_neg_sequence(self, pos_sequence):#pos_sequence每个序列要预测的item组成的序列，shape of `(N, )
        """For each moment, sampling one item from all the items except the one the user clicked on at that moment.

        Args:
            pos_sequence (torch.Tensor):  all users' item history sequence, with the shape of `(N, )`.

        Returns:
            torch.tensor : all users' negative item history sequence.

        """
        total_num = len(pos_sequence)#等于序列总数
        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        while len(check_list) > 0:
            value_ids[check_list] = self.sampling(len(check_list))
            check_index = np.where(value_ids[check_list] == pos_sequence[check_list])
            check_list = check_list[check_index]

        return torch.tensor(value_ids)
