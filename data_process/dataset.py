from os.path import join as joinpath
import re
from common.utlis import iterline
from common.constants import const_dict, TYPE_SET
from common.predicate import pred_register, Predicate
from copy import deepcopy
from common import constants
import os
import math
import torch
from tqdm import tqdm
import itertools
from common.cmd_args import cmd_args
from random import shuffle
import random



class DomainDataset:

    def __init__(self, dataset_path):

        pred_path = joinpath(dataset_path, 'pred.txt')

        fact_path_ls = [joinpath(dataset_path, 'fact.txt') if cmd_args.kb else joinpath(dataset_path, 'fact_domains')]
        if os.path.isfile(joinpath(dataset_path, 'train.txt')):
            fact_path_ls.append(joinpath(dataset_path, 'train.txt'))
        valid_path_ls = [joinpath(dataset_path, 'valid.txt') if cmd_args.kb else joinpath(dataset_path, 'valid_domains')]
        test_path_ls = [joinpath(dataset_path, 'test.txt') if cmd_args.kb else joinpath(dataset_path, 'test_domains')]

        if os.path.isfile(joinpath(dataset_path, 'ent.txt')):
            ent_path_ls = [joinpath(dataset_path, 'ent.txt')]
        else:
            ent_path_ls = None

        self.fact_pred2domain_dict, self.fact_domain_set = preprocess_withDomain(pred_path, fact_path_ls, ent_path_ls)
        self.valid_pred2domain_dict, self.valid_domain_set = preprocess_withDomain(pred_path, valid_path_ls, ent_path_ls)
        self.test_pred2domain_dict, self.test_domain_set = preprocess_withDomain(pred_path, test_path_ls, ent_path_ls)

    def sample(self, pred2domain_dict, batch_size, allow_recursion, rotate, keep_array, tgt_pred_ls, bg_domain):

        tgt_pred_ls = [pn for pn in tgt_pred_ls if len(pred2domain_dict[pn]) > 0]

        iter_ls = [self.batch_iter(pred2domain_dict, batch_size, pn, allow_recursion, keep_array, bg_domain) for pn in tgt_pred_ls]

        num_batch_ls = [self.get_numBatches(pred2domain_dict, pn, batch_size) for pn in tgt_pred_ls]
        num_batches = sum(num_batch_ls)

        def epoch_iterator():
            if rotate:
                for rotate_ls in itertools.zip_longest(*iter_ls):  # [(p1,q1), (p2,q2), (p3,None), ...]
                    for batch in rotate_ls:
                        if batch is not None:
                            yield batch
            else:
                for it in iter_ls:
                    for batch in it:
                        yield batch

        return num_batches, epoch_iterator()

    def get_numBatches(self, pred2domain_dict, pred_name, bsize):
        cnt = 0
        for domain in pred2domain_dict[pred_name]:
            cnt += math.ceil(len(domain.fact_dict[pred_name]) / bsize)
        return cnt

    def get_numSamples(self, pred2domain_dict, pred_name):
        cnt = 0
        for domain in pred2domain_dict[pred_name]:
            cnt += len(domain.fact_dict[pred_name])
        return cnt

    def batch_iter(self, pred2domain_dict, batch_size, p_star, allow_recursion, keep_array, bg_domain):
        """
            iterator that enumerates batches for p_star in all domains. No cross-domain batches

        :param batch_size:
        :param p_star:
        :param allow_recursion:
        :param keep_array:
        :return:
             unp_arr_ls: [num_unp, (b, num_ent, 1)], same data expanded to bsize samples
             bip_arr_ls: [num_bip+1, (1, num_ent, num_ent)], +1 for ident_phi
             arg_input: (bsize, num_args, num_ents)
             input_y: (bsize)
        """

        assert p_star != constants.IDENT_PHI

        is_unp = pred_register.is_unp(p_star)

        for domain in pred2domain_dict[p_star]:
            num_unp, num_bip, num_ents = len(domain.unp_ls), len(domain.bip_ls), len(domain.const2ind_dict)
            if bg_domain is None:
                domain_unp_arr_ls, domain_bip_arr_ls = domain.toArray(keep_array=keep_array)
            else:
                domain_unp_arr_ls, domain_bip_arr_ls = bg_domain.toArray(keep_array=keep_array)
            # if num_bip == bip_arr.size(0): # concat Ident matrix
            #     bip_arr = torch.cat([bip_arr, torch.eye(num_ents).unsqueeze(0).to(cmd_args.device)], dim=0)
            train_pair_ls = domain.fact_dict[p_star]
            domain_numSamples = len(train_pair_ls)

            cnt = 0  # bsize can be smaller than batch_size if domain is small
            while cnt * batch_size < domain_numSamples:
                arg_input_ls = []
                input_y = []
                batch_pair_ls = train_pair_ls[cnt * batch_size:(cnt + 1) * batch_size]
                for ind, pair in enumerate(batch_pair_ls):
                    arg_input = torch.zeros(1 if is_unp else 2, num_ents)
                    val, consts = pair
                    for c_ind, const in enumerate(consts):
                        arg_input[c_ind, domain.const2ind_dict[const]] = 1
                    input_y.append(val)
                    arg_input_ls.append(arg_input)

                if cmd_args.sample_neg:
                    neg_y, neg_arg_input_ls = [], []
                    for ind in range(len(batch_pair_ls)):
                        arg_input = torch.zeros(1 if is_unp else 2, num_ents)
                        for c_ind in ([0] + ([] if is_unp else [1])):
                            arg_input[c_ind, random.randint(0, num_ents-1)] = 1
                        neg_y.append(0)
                        neg_arg_input_ls.append(arg_input)

                    mix_choice = [random.randint(0, 2) for _ in range(len(batch_pair_ls))]
                    input_y = [[input_y, neg_y, neg_y][c][ind] for ind, c in enumerate(mix_choice)]
                    arg_input_ls = [[arg_input_ls, neg_arg_input_ls, neg_arg_input_ls][c][ind] for ind, c in enumerate(mix_choice)]

                # bsize = len(input_y)
                if bg_domain is None:
                    unp_ls, bip_ls = deepcopy(domain.unp_ls), deepcopy(domain.bip_ls)
                else:
                    unp_ls, bip_ls = deepcopy(bg_domain.unp_ls), deepcopy(bg_domain.bip_ls)
                unp_arr_ls, bip_arr_ls = [unp_arr for unp_arr in domain_unp_arr_ls], \
                                         [bip_arr for bip_arr in domain_bip_arr_ls]

                if not allow_recursion:
                    if is_unp:
                        ind = unp_ls.index(p_star)
                        unp_ls.pop(ind)
                        unp_arr_ls.pop(ind)
                    else:
                        ind = bip_ls.index(p_star)
                        bip_ls.pop(ind)
                        bip_arr_ls.pop(ind)

                cnt += 1
                input_y = torch.tensor(input_y, dtype=torch.float32, device=cmd_args.device)
                arg_input = torch.stack(arg_input_ls, dim=0).to(cmd_args.device)

                # the case where there's only one channel which is trivial
                if is_unp and (len(unp_ls) == 0):
                    continue

                yield [[unp_arr_ls, bip_arr_ls], arg_input], input_y, unp_ls, bip_ls, p_star


class Domain:

    def __init__(self, unp_ls, bip_ls, const2ind_dict, ind2const_dict, fact_dict):

        self.unp_ls = unp_ls
        # manually put Ident predicate into the list, though it's in pred_register
        self.bip_ls = bip_ls + [constants.IDENT_PHI]
        self.const2ind_dict = const2ind_dict
        self.ind2const_dict = ind2const_dict
        self.fact_dict = fact_dict
        self.name = None
        self.has_neg_sample = False

        self.unp_arr_ls, self.bip_arr_ls = None, None

    def __eq__(self, other):
        if isinstance(other, Domain):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def toArray(self, update=False, keep_array=False):
        """

        :param update:
            set to true for re-computing the array
        :param keep_array:
            keep the generated array, useful if in single domain env, i.e. non-GQA tasks
        :return:
            unp_array of (num_unp, num_const, 1)
            bip_array of (num_bip, num_const, num_const)
        """

        if (self.unp_arr_ls is not None) and (self.bip_arr_ls is not None) and (not update):
            return self.unp_arr_ls, self.bip_arr_ls

        num_unp, num_bip, num_const = len(self.unp_ls), len(self.bip_ls), len(self.const2ind_dict)

        unp_arr_ls = [torch.zeros(num_const, 1, device=cmd_args.device) for _ in range(num_unp)]

        for ind, unp in enumerate(self.unp_ls):
            for val, consts in self.fact_dict[unp]:
                entry_inds = tuple([self.const2ind_dict[const] for const in consts])
                unp_arr_ls[ind][entry_inds] = val

        if cmd_args.sparse_mat:
            bip_arr_ls = []

            for _, bip in enumerate(self.bip_ls[:-1]):
                ind_ls, val_ls = [], []
                for val, consts in self.fact_dict[bip]:
                    ind_ls.append([self.const2ind_dict[const] for const in consts])
                    val_ls.append(val)
                ind_arr = torch.LongTensor(ind_ls).transpose(0, 1)
                val_arr = torch.FloatTensor(val_ls)
                bip_arr_ls.append(
                    torch.sparse_coo_tensor(ind_arr, val_arr, [num_const, num_const], device=cmd_args.device))

            # ident a placeholder
            bip_arr_ls.append(torch.sparse_coo_tensor([[], []], [], [num_const, num_const], device=cmd_args.device))

        else:
            bip_arr_ls = [torch.zeros(num_const, num_const, device=cmd_args.device) for _ in range(num_bip - 1)] + \
                         [torch.eye(num_const, device=cmd_args.device)]

            for ind, bip in enumerate(self.bip_ls[:-1]):
                for val, consts in self.fact_dict[bip]:
                    entry_inds = tuple([self.const2ind_dict[const] for const in consts])
                    bip_arr_ls[ind][entry_inds] = val

        if keep_array:
            self.unp_arr_ls = unp_arr_ls
            self.bip_arr_ls = bip_arr_ls

        return unp_arr_ls, bip_arr_ls


def gen_neg_samples(domain, unp_ls, bip_ls, sample_ratio=1, global_unp_arr=None, global_bip_arr=None):
    """

    :param domain:
    :param unp_ls:
    :param bip_ls:
    :param sample_ratio:
    :param global_unp_arr:
        used for single domain case
    :param global_bip_arr:
        used for single domain case
    :return:

    """

    domain_unp2ind_dict = dict([(pn, ind) for ind, pn in enumerate(domain.unp_ls)])
    domain_bip2ind_dict = dict([(pn, ind) for ind, pn in enumerate(domain.bip_ls)])

    unp_arr_ls, bip_arr_ls = domain.toArray() if global_unp_arr is None else (global_unp_arr, global_bip_arr)
    num_const = bip_arr_ls[0].size(0)

    for pn in unp_ls:
        if pn not in domain_unp2ind_dict:
            continue  # '%s not in domain %s' % (pn, domain.name)

        if len(domain.fact_dict[pn]) == 0:
            continue

        unp_ind = domain_unp2ind_dict[pn]
        unp_neg_indices = (1 - unp_arr_ls[unp_ind]).nonzero()[:, 0]
        num_pos_samples = num_const - unp_neg_indices.size(0)

        perm_indices = torch.randperm(unp_neg_indices.size(0))[:num_pos_samples * sample_ratio]  # permute neg samples
        unp_neg_indices = unp_neg_indices[perm_indices]

        for neg_ind in unp_neg_indices:
            consts = tuple([domain.ind2const_dict[int(neg_ind)]])
            domain.fact_dict[pn].append((0, consts))

        shuffle(domain.fact_dict[pn])

    for pn in bip_ls:
        if pn not in domain_bip2ind_dict:
            continue  # '%s not in domain %s' % (pn, domain.name)

        if len(domain.fact_dict[pn]) == 0:
            continue

        bip_ind = domain_bip2ind_dict[pn]
        bip_neg_indices = (1 - bip_arr_ls[bip_ind]).nonzero()
        num_pos_samples = num_const ** 2 - bip_neg_indices.size(0)

        if num_pos_samples == 0:
            continue

        # multiply by 2 for 2 slots
        perm_indices = torch.randperm(bip_neg_indices.size(0))[
                       :num_pos_samples * sample_ratio * 2]  # permute neg samples
        bip_neg_indices = bip_neg_indices[perm_indices]

        for neg_ind in bip_neg_indices:
            consts = tuple([domain.ind2const_dict[int(e)] for e in neg_ind])
            domain.fact_dict[pn].append((0, consts))

        shuffle(domain.fact_dict[pn])

    domain.has_neg_sample = True


def preprocess_withDomain(pred_path, fact_path_ls, ent_path_ls=None):
    pred_reg = re.compile(r'([\w-]+)\(([^)]+)\)')

    for line in iterline(pred_path):
        m = pred_reg.match(line)

        # TensorLog data
        if m is None:
            pred_name = line
            pred_name = pred_name.replace('.', 'DoT') # deal with fb15k
            var_types = ['type', 'type']
        else:
            pred_name = m.group(1)
            var_types = m.group(2).split(',')

        if pred_name in pred_register.pred_dict:
            continue

        pred_register.add(Predicate(pred_name, var_types))
        TYPE_SET.update(var_types)

    if constants.IDENT_PHI not in pred_register.pred_dict:
        pred_register.add(Predicate(constants.IDENT_PHI, ['type', 'type']))

    if ent_path_ls is not None:
        global_const2ind, global_ind2const = {}, {}
        for fp in ent_path_ls:
            for line in iterline(fp):
                if line not in global_const2ind:
                    global_ind2const[len(global_const2ind)] = line
                    global_const2ind[line] = len(global_const2ind)
    else:
        global_const2ind, global_ind2const = None, None

    def parse_fact(fp_ls, const2ind_dict, ind2const_dict, verbose=False, keep_empty=False):
        unp_set, bip_set = set(), set()
        const2ind_dict = {} if const2ind_dict is None else const2ind_dict
        ind2const_dict = {} if ind2const_dict is None else ind2const_dict
        fact_dict = dict([(pn, []) for pn in pred_register.pred_dict.keys()]) if keep_empty else {}

        if verbose:
            v = lambda x: tqdm(x)
        else:
            v = lambda x: x

        for fp in fp_ls:
            for line in v(iterline(fp)):
                parts = line.split('\t')

                # TensorLog case
                if len(parts) == 3:
                    val = 1
                    e1, pred_name, e2 = parts
                    pred_name = pred_name.replace('.', 'DoT')  # deal with fb15k
                    consts = [e1, e2]
                else:
                    val = int(parts[0])
                    m = pred_reg.match(parts[1])
                    assert m is not None

                    pred_name = m.group(1)
                    consts = m.group(2).split(',')

                if pred_name not in pred_register.pred_dict:
                    continue

                for const in consts:
                    if const not in const2ind_dict:
                        ind2const_dict[len(const2ind_dict)] = const
                        const2ind_dict[const] = len(const2ind_dict)

                fact = (val, tuple(consts))
                if pred_name in fact_dict:
                    fact_dict[pred_name].append(fact)
                else:
                    fact_dict[pred_name] = [fact]

                if pred_register.is_unp(pred_name):
                    unp_set.add(pred_name)
                else:
                    bip_set.add(pred_name)

        if keep_empty:
            pn_ls = list(pred_register.pred_dict.keys())
            unp_ls = [pn for pn in pn_ls if pred_register.is_unp(pn)]
            bip_ls = [pn for pn in pn_ls if not pred_register.is_unp(pn)]
        else:
            unp_ls = list(sorted(unp_set))
            bip_ls = list(sorted(bip_set))

        return Domain(unp_ls, bip_ls, const2ind_dict, ind2const_dict, fact_dict)

    domain_set = set()
    pred2domain_dict = dict((pred_name, []) for pred_name in pred_register.pred_dict)
    # a single file containing all facts, e.g. FB15K
    if os.path.isfile(fact_path_ls[0]):
        tqdm.write('Processing Single Domain..')
        d = parse_fact(fact_path_ls, global_const2ind, global_ind2const, verbose=True)
        d.name = 'default'
        for pn in d.unp_ls + d.bip_ls:
            pred2domain_dict[pn].append(d)
            domain_set.add(d)

    # a folder containing fact files named with unique ids, e.g. GQA images
    elif os.path.isdir(fact_path_ls[0]):
        assert len(fact_path_ls) == 1
        tqdm.write('Processing Multiple Domains..')
        for fn in tqdm(os.listdir(fact_path_ls[0])):
            d = parse_fact([joinpath(fact_path_ls[0], fn)], global_const2ind, global_ind2const,
                           keep_empty=cmd_args.keep_empty)
            d.name = fn
            if (len(d.unp_ls) == 0) or (len(d.bip_ls) == 0):
                tqdm.write('skip %s for zero-length unp or bip ls' % fn)
                continue
            for pn in d.unp_ls + d.bip_ls:
                pred2domain_dict[pn].append(d)
                domain_set.add(d)

    else:
        raise ValueError

    return pred2domain_dict, domain_set


def preprocess(pred_path, fact_path):
    pred_reg = re.compile(r'([\w]+)\(([^)]+)\)')

    for line in iterline(pred_path):
        m = pred_reg.match(line)
        assert m is not None

        pred_name = m.group(1)
        var_types = m.group(2).split(',')

        pred_register.add(Predicate(pred_name, var_types))
        TYPE_SET.update(var_types)

    fact_dict = dict((pred_name, set()) for pred_name in pred_register.pred_dict)
    pred2domain_dict = dict((pred_name, []) for pred_name in pred_register.pred_dict)

    def parse_fact(fp):
        avail_pred_ls = []

        for line in iterline(fp):
            parts = line.split('\t')
            val = int(parts[0])
            m = pred_reg.match(parts[1])
            assert m is not None

            pred_name = m.group(1)
            consts = m.group(2).split(',')

            if pred_name not in fact_dict:
                continue

            for ind, var_type in enumerate(pred_register.pred_dict[pred_name].var_types):
                const_dict.add_const(var_type, consts[ind])

            fact = (val, tuple(consts))
            fact_dict[pred_name].add(fact)
            avail_pred_ls.append(pred_name)

        return avail_pred_ls

    # a single file containing all facts, e.g. FB15K
    if os.path.isfile(fact_path):
        parse_fact(fact_path)

    # a folder containing fact files named with unique ids, e.g. GQA images
    elif os.path.isdir(fact_path):
        for fn in os.listdir(fact_path):
            pred_ls = parse_fact(joinpath(fact_path, fn))
            for pn in pred_ls:
                pred2domain_dict[pn].append(fn)

    else:
        raise ValueError

    fact_dict_sort = dict((pred_name, sorted(list(fact_dict[pred_name]), key=lambda x: x[1]))
                          for pred_name in fact_dict.keys())
    const_dict_sort = dict([(type_name, sorted(list(const_dict[type_name])))
                            for type_name in const_dict.constants.keys()])

    return fact_dict, fact_dict_sort, const_dict_sort, pred2domain_dict
