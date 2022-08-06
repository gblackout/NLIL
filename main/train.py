import os, sys

sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__)))

from common.cmd_args import cmd_args
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import torch.optim as optim
from itertools import chain
from common.utlis import makedir, get_output_folder, EarlyStopMonitor
from os.path import join as joinpath
from common.predicate import pred_register
from model.Models import RuleLearner, PhiList, TabularPhiList
from model.gqa_freq import GQAFreq
import common.phi as phi
from data_process.dataset import DomainDataset
from data_process.dataset import gen_neg_samples
from common import constants
from common.utlis import iterline
import math
import time
import sys


def tensorlog_dataset(root_path):
    name_ls = ['family', 'fb15k-237', 'wn18']
    return any(n in root_path for n in name_ls)


def init_model2():
    unp_ls = [(pn, phi.TabularPhi(pn)) for pn in pred_register.pred_dict if pred_register.is_unp(pn)]
    bip_ls = [(pn, phi.TabularPhi(pn)) for pn in pred_register.pred_dict if not pred_register.is_unp(pn)]

    phi_list = TabularPhiList(unp_ls, bip_ls).to(cmd_args.device)
    rule_learner = RuleLearner(pred_register.num_pred, phi_list).to(cmd_args.device)

    return rule_learner


def prep_dataset():
    dataset = DomainDataset(cmd_args.data_root)
    bg_domain = None
    fbd = None

    if tensorlog_dataset(cmd_args.data_root):
        # for KG datasets, there is only one domain (i.e., the entire KG itself)
        # Therefore, we construct and keep the adjacency arrays throughout the experiments
        fact_domain = list(dataset.fact_domain_set)[0]
        test_domain = list(dataset.test_domain_set)[0]

        fact_domain.toArray(update=False, keep_array=True)

        tgt_pred_ls = test_domain.bip_ls[:-1]

        bg_domain = fact_domain
        if 'fb15k' in cmd_args.data_root:
            fbd = FBDomain(10)

    elif 'gqa' in cmd_args.data_root:

        tgt_pred_ls = [line for line in iterline('freq_gqa.txt')]

    elif 'evensucc' in cmd_args.data_root:
        tqdm.write('running even-odd..')
        tgt_pred_ls = ['even']

    else:
        raise ValueError

    return dataset, tgt_pred_ls, bg_domain, fbd


def train():
    output_path = get_output_folder(cmd_args.output_root, run_name=cmd_args.run_name)
    makedir(output_path, remove_old=False)
    model_path = joinpath(output_path, 'best_model')
    makedir(model_path,  remove_old=False)
    with open(joinpath(output_path, 'run_log.txt'), 'w') as f:
        for arg in vars(cmd_args):
            f.write('%s %s\n' % (str(arg), str(getattr(cmd_args, arg))))

    dataset, tgt_pred_ls, bg_domain, fbd = prep_dataset()

    skip_prob_dict = dict((pn, 0) for pn in tgt_pred_ls)

    model_dict = dict([(pn, init_model2()) for pn in tgt_pred_ls])
    opt_dict = {}
    for pn in tgt_pred_ls:
        model = model_dict[pn]
        params = [model.parameters()]
        optimizer = optim.Adam(chain.from_iterable(params),
                               lr=cmd_args.learning_rate, weight_decay=cmd_args.l2_coef)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=cmd_args.lr_decay_factor,
                                                         patience=cmd_args.lr_decay_patience,
                                                         min_lr=cmd_args.lr_decay_min)
        opt_dict[pn] = [optimizer, scheduler]

    if cmd_args.model_load_path:
        tqdm.write('Load model from %s' % cmd_args.model_load_path)
        for pn, model in model_dict.items():
            model.load_state_dict(torch.load(joinpath(cmd_args.model_load_path, str(pred_register.pred2ind[pn]))))

    monitor = EarlyStopMonitor(cmd_args.patience)

    if cmd_args.hard_sample:
        tqdm.write('Doing hard sample')
    else:
        cmd_args.num_samples = 1
        tqdm.write('Doing averaging')

    shouldstop = False
    st = time.time()
    total_batch = 0
    for cur_epoch in range(cmd_args.num_epochs):

        if shouldstop:
            break

        num_batches, epoch_iterator = dataset.sample(dataset.fact_pred2domain_dict,
                                                     cmd_args.batch_size,
                                                     allow_recursion=cmd_args.allow_recursion,
                                                     rotate=cmd_args.rotate,
                                                     keep_array=cmd_args.kb,
                                                     tgt_pred_ls=tgt_pred_ls,
                                                     bg_domain=None)

        pbar = tqdm(total=num_batches)
        acc_loss = 0.0
        cur_batch = 0
        for input_x, input_y, unp_ls, bip_ls, p_star in epoch_iterator:

            if shouldstop:
                break

            skip = False
            if cmd_args.skip_trained and ((skip_prob_dict[p_star] > 0.5) and
                                          (random.random() < skip_prob_dict[p_star])):
                skip = True

            if 'fb15k' in cmd_args.data_root:
                bip_ls, input_x = fbd.filter(bip_ls, input_x, p_star)
            if bip_ls is None:
                skip = True

            if not skip:
                # [num_sample] loss
                sample_val_ls, mask_ls, name_ls = model_dict[p_star](input_x, input_y, pred_register.pred2ind[p_star], unp_ls, bip_ls,
                                                 cmd_args.gumbelmax_temp, cmd_args.num_samples)

                tqdm.write('%s <- %s' % (p_star, name_ls[0][-1][-1][0]))

                cand_val = torch.cat([val_ls[-1] for val_ls in sample_val_ls], dim=-1)
                input_y = input_y.to(cmd_args.device)

                num_cand = cand_val.size(-1)
                succ_loss = F.binary_cross_entropy(cand_val.clamp(max=1), input_y.view(-1, 1).expand(-1, num_cand))

                loss = succ_loss

                optimizer, scheduler = opt_dict[p_star]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(succ_loss)

            pbar.update()
            acc_loss += (acc_loss / (cur_batch+1e-6)) if skip else loss.item()
            cur_batch += 1
            total_batch += 1
            pbar.set_description('Epoch %i, avg loss: %.4f' % (cur_epoch, acc_loss / cur_batch))

            if total_batch % cmd_args.num_batches_per_valid == 0:

                tqdm.write('validating..')
                with torch.no_grad():
                    rule_ls, global_prf, global_f = inference(model_dict, dataset,
                                                                          dataset.valid_pred2domain_dict,
                                                                          tgt_pred_ls, bg_domain, fbd, skip_prob_dict)

                update_rule = -global_f < monitor.cur_best
                shouldstop = monitor.update(-global_f)
                if update_rule:
                    best_rule_ls = rule_ls
                    for pn, model in model_dict.items():
                        torch.save(model.state_dict(), joinpath(model_path, str(pred_register.pred2ind[pn])))
                    with open(joinpath(output_path, 'best_rule.txt'), 'w') as f:
                        f.write('Epoch %i batch %i time %.4f %s\n' %
                                (cur_epoch, cur_batch, time.time()-st, global_prf))
                        for rule_str in best_rule_ls:
                            f.write('%s\n' % rule_str)


        pbar.close()

    tqdm.write('testing..')
    with torch.no_grad():
        for pn, model in model_dict.items():
            model_path = joinpath(model_path, str(pred_register.pred2ind[pn]))
            if not os.path.isfile(model_path):
                tqdm.write('No saved model for %s ' % pn)
                continue
            model.load_state_dict(torch.load(model_path))
        _, global_prf, global_f = inference(model_dict, dataset, dataset.test_pred2domain_dict,
                                                       tgt_pred_ls, bg_domain, fbd, skip_prob_dict)
        with open(joinpath(output_path, 'test_loss.txt'), 'w') as f:
            f.write('%s\n' % (global_prf))

        if cmd_args.kb:
            tqdm.write('evaluating mrr and hits..')
            mh_dict = mrr_and_hits(model_dict, dataset, dataset.test_pred2domain_dict, tgt_pred_ls, bg_domain, fbd)
            with open(joinpath(output_path, 'test_mh.txt'), 'w') as f:
                for pn, mh in mh_dict.items():
                    mh_str = 'mrr %.4f hits@10 %.4f %s' % (mh[0], mh[1], pn)
                    tqdm.write(mh_str)
                    f.write('%s\n' % mh_str)
                mrr = sum(e[0] for e in mh_dict.values()) / len(mh_dict)
                hits = sum(e[1] for e in mh_dict.values()) / len(mh_dict)
                mh_str = 'global mrr %.4f hits@10 %.4f' % (mrr, hits)
                tqdm.write(mh_str)
                f.write('%s\n' % mh_str)


class FBDomain:

    def __init__(self, filter_under):
        self.filter_under = filter_under
        pstar_domain_dict = {}
        pn_freq_dict = {}

        for line in iterline(joinpath(cmd_args.data_root, 'train_cnt.txt')):
            parts = line.split(' ')
            cnt, pn = int(parts[0]), parts[1]
            pn_freq_dict[pn] = cnt

        for line in iterline(joinpath(cmd_args.data_root, 'rules_by_q')):
            parts = line.split(' ')
            pstar_domain_dict[parts[0]] = set(parts[1:])

        self.pstar_domain_dict = pstar_domain_dict
        self.pn_freq_dict = pn_freq_dict

    def filter(self, bip_ls, input_x, p_star, evaluate=False):
        pstar_d = p_star.split('/')[1]
        d_set = self.pstar_domain_dict[pstar_d]
        if cmd_args.allow_recursion:
            try:
                pstar_ind = bip_ls.index(p_star)
            except:
                pstar_ind = None
        else:
            pstar_ind = None

        filter_ind_ls = sorted([(ind, self.pn_freq_dict[pn]) for ind, pn in
                                enumerate(bip_ls[:-1]) if pn.split('/')[1] in d_set],
                               key=lambda x:x[1], reverse=True)[:self.filter_under]
        filter_ind_ls = [e[0] for e in filter_ind_ls]

        if (len(filter_ind_ls) == 1) and (filter_ind_ls[0] == pstar_ind):
            filter_ind_ls = []

        if evaluate:
            if len(filter_ind_ls) == 0:
                filter_ind_ls = [3]

        if len(filter_ind_ls) == 0:
            return None, None

        input_x[0][1] = [input_x[0][1][i] for i in filter_ind_ls] + [input_x[0][1][-1]]
        return [bip_ls[i] for i in filter_ind_ls] + [bip_ls[-1]], input_x


def inference(model_dict, dataset, domain_dict, tgt_pred_ls, bg_domain, fbd, skip_prob_dict):
    # get one hard sample to get the learned formulas
    infer_temp, infer_numSample = 0.001, 1
    bak1, bak2, bak3 = cmd_args.hard_sample, cmd_args.attn_temp, cmd_args.sample_neg
    if not cmd_args.kb:
        cmd_args.hard_sample = True
    cmd_args.attn_temp = infer_temp
    cmd_args.sample_neg = False

    acc_loss = 0.0
    cur_batch = 0
    rule_ls = []
    g_pred_cnt, g_inter_cnt, g_y_cnt = 0, 0, 0
    rule_dict = dict()
    pbar = tqdm(tgt_pred_ls)
    for tgt_p in pbar:
        pbar.set_description('Inferring facts for %s' % (tgt_p))
        # rotate over data w.r.t each p_star
        num_batches, epoch_iterator = dataset.sample(domain_dict,
                                                     cmd_args.batch_size,
                                                     allow_recursion=cmd_args.allow_recursion,
                                                     rotate=False,
                                                     keep_array=cmd_args.kb,
                                                     tgt_pred_ls=[tgt_p],
                                                     bg_domain=bg_domain)

        name_ls = None
        apply_cnt, pred_cnt, inter_cnt, y_cnt = None, None, None, None
        joint_pred_cnt, joint_inter_cnt = 0, 0
        for input_x, input_y, unp_ls, bip_ls, p_star in epoch_iterator:

            if 'fb15k' in cmd_args.data_root:
                bip_ls, input_x = fbd.filter(bip_ls, input_x, p_star, evaluate=True)
            assert bip_ls is not None


            # [num_sample] loss

            sample_val_ls, mask_ls, name_ls = model_dict[p_star](input_x, input_y, pred_register.pred2ind[p_star], unp_ls, bip_ls,
                                             infer_temp, infer_numSample)


            cand_val = torch.cat([val_ls[-1] for val_ls in sample_val_ls], dim=-1)
            # cand_mask = torch.cat(mask_ls, dim=-1)
            num_cand = cand_val.size(-1)
            input_y = input_y.to(cmd_args.device).view(-1, 1).expand(-1, num_cand)

            prediction = (cand_val >= 0.5).type(torch.float)#(cand_val * cand_mask >= 0.5).type(torch.float)
            intersect = ((prediction > 0) * (input_y > 0)).type(torch.float)
            if apply_cnt is None:
                apply_cnt = input_y.sum(dim=0)#cand_mask.sum(dim=0)
                pred_cnt = prediction.sum(dim=0)
                inter_cnt = intersect.sum(dim=0)
                y_cnt = input_y.sum(dim=0)
                joint_pred_cnt = (prediction.max(dim=-1)[0]).sum()
                joint_inter_cnt = (intersect.max(dim=-1)[0]).sum()
            else:
                apply_cnt += input_y.sum(dim=0)#cand_mask.sum(dim=0)
                pred_cnt += prediction.sum(dim=0)
                inter_cnt += intersect.sum(dim=0)
                y_cnt += input_y.sum(dim=0)
                joint_pred_cnt += (prediction.max(dim=-1)[0]).sum()
                joint_inter_cnt += (intersect.max(dim=-1)[0]).sum()

            # acc_loss += loss
            cur_batch += 1

        if y_cnt is None:
            tqdm.write('skipping target %s as it does not exist in the domain' % (tgt_p))
            continue

        y_cnt += 1e-8  # prevent div by 0
        pred_cnt += 1e-8
        joint_pred_cnt += 1e-8

        p, r = inter_cnt / pred_cnt, inter_cnt / y_cnt
        f = 2 * p * r / (p + r + 1e-8)

        joint_p, joint_r = joint_inter_cnt / joint_pred_cnt, joint_inter_cnt / y_cnt[0]
        joint_f = 2 * joint_p * joint_r / (joint_p + joint_r + 1e-8)

        g_pred_cnt += joint_pred_cnt
        g_inter_cnt += joint_inter_cnt
        g_y_cnt += y_cnt[0]

        tgt_joint_str = 'tgt pred joint p %.4f r %.4f f %.4f' % (joint_p.item(), joint_r.item(), joint_f.item())
        rule_ls.append(tgt_joint_str)
        for ind, cand in enumerate(name_ls[0][-1][-1]):
            tgt_p_rule = 'cnt %.4f p %.4f r %.4f f %.4f\t: %s <- %s' % (apply_cnt[ind].item(),
                                                                        p[ind].item(),
                                                                        r[ind].item(),
                                                                        f[ind].item(), tgt_p, cand)
            rule_ls.append(tgt_p_rule)

        if skip_prob_dict[tgt_p] < joint_f.item():
            skip_prob_dict[tgt_p] = joint_f.item()

    global_p, global_r = g_inter_cnt / g_pred_cnt, g_inter_cnt / g_y_cnt
    global_f = 2 * global_p * global_r / (global_p + global_r + 1e-8)

    global_prf = 'global p %.4f r %.4f f %.4f' % (global_p.item(), global_r.item(), global_f.item())
    tqdm.write(global_prf)
    cmd_args.hard_sample, cmd_args.attn_temp, cmd_args.sample_neg = bak1, bak2, bak3

    return rule_ls, global_prf, global_f


def kb_test(run_log_path):
    assert os.path.isdir(run_log_path)
    model_path = joinpath(run_log_path, 'best_model')
    assert os.path.isdir(model_path)

    dataset, tgt_pred_ls, bg_domain, fbd = prep_dataset()
    model_dict = dict([(pn, init_model2()) for pn in tgt_pred_ls])
    with torch.no_grad():
        for pn, model in model_dict.items():
            model.load_state_dict(torch.load(joinpath(model_path, str(pred_register.pred2ind[pn]))))
        mh_dict = mrr_and_hits(model_dict, dataset, dataset.test_pred2domain_dict, tgt_pred_ls, bg_domain, fbd)

    with open(joinpath(run_log_path, 'test_mh.txt'), 'w') as f:
        for pn, mh in mh_dict.items():
            mh_str = 'mrr %.4f hits@10 %.4f cnt %i %s' % (mh[0], mh[1], int(mh[2]), pn)
            tqdm.write(mh_str)
            f.write('%s\n' % mh_str)
        wsum = sum(float(e[2]) for e in mh_dict.values())
        mrr = sum(e[0] * e[2] / wsum for e in mh_dict.values())
        hits = sum(e[1] * e[2] / wsum for e in mh_dict.values())
        mh_str = 'global mrr %.4f hits@10 %.4f' % (mrr, hits)
        tqdm.write(mh_str)
        f.write('%s\n' % mh_str)


def mrr_and_hits(model_dict, dataset, domain_dict, tgt_pred_ls, bg_domain, fbd):
    # get one hard sample to get the learned formulas
    infer_temp, infer_numSample = 0.001, 1
    num_const = len(bg_domain.const2ind_dict)
    n_eval = 20

    def const_iter(nc, nb, is_unp):
        cnt = 0
        l = []
        for i in range(nc):
            k = [i] if is_unp else range(nc)
            for j in k:
                l.append((i, j))
                cnt += 1
                if cnt == nb:
                    yield l
                    l = []
                    cnt = 0

    def rand_iter(nc, nb, is_unp):
        num_eval = n_eval
        cnt, t_cnt = 0, 0
        l = []
        while t_cnt < num_eval:
            i = random.randint(0, nc-1)
            j = i if is_unp else random.randint(0, nc-1)
            l.append((i, j))
            cnt += 1
            if cnt == nb:
                yield l
                l = []
                cnt = 0
                t_cnt += 1
        if len(l) > 0:
            yield l

    mh_dict = {}
    for tgt_p in tgt_pred_ls:
        # rotate over data w.r.t each p_star
        num_batches, epoch_iterator = dataset.sample(domain_dict,
                                                     cmd_args.batch_size,
                                                     allow_recursion=cmd_args.allow_recursion,
                                                     rotate=False,
                                                     keep_array=cmd_args.kb,
                                                     tgt_pred_ls=[tgt_p],
                                                     bg_domain=bg_domain)

        cand_val_ls = []
        pstar_bip_ls, pstar_unp_ls, pstar_unp_arr_ls, pstar_bip_arr_ls = [], [], [], []
        for input_x, input_y, unp_ls, bip_ls, p_star in epoch_iterator:

            if 'fb15k' in cmd_args.data_root:
                bip_ls, input_x = fbd.filter(bip_ls, input_x, p_star, evaluate=True)
            assert bip_ls is not None

            sample_val_ls, mask_ls, name_ls = model_dict[p_star](input_x, input_y, pred_register.pred2ind[p_star],
                                                                 unp_ls, bip_ls, infer_temp, infer_numSample)

            cand_val = torch.cat([val_ls[-1] for val_ls in sample_val_ls], dim=-1)
            cand_val_ls.append(cand_val)
            pstar_bip_ls = bip_ls
            pstar_unp_ls = unp_ls
            pstar_unp_arr_ls = input_x[0][0]
            pstar_bip_arr_ls = input_x[0][1]

        cand_val = torch.cat(cand_val_ls, dim=0)
        is_unp = pred_register.is_unp(tgt_p)
        full_batch_size = 100
        c_iter = rand_iter(num_const, full_batch_size, is_unp) if cmd_args.rand_eval else \
            const_iter(num_const, full_batch_size, is_unp)
        num_full_batches = math.ceil(num_const * (1 if is_unp else num_const) / full_batch_size)
        num_full_batches = n_eval if cmd_args.rand_eval else num_full_batches
        query_rank = torch.zeros_like(cand_val).squeeze(-1)
        p_bar = tqdm(total=num_full_batches)
        p_bar.set_description(tgt_p)
        for ind_ls in c_iter:
            ind_arr = torch.tensor(ind_ls, dtype=torch.int64, device=cmd_args.device) # (b,2)
            if is_unp:
                ind_arr = ind_arr[:, 0].unsqueeze(1) # (b,1)
            arg_input = F.one_hot(ind_arr, num_classes=num_const) # (b,num_arg,num_const)
            arg_input = arg_input.type(torch.float32)
            input_x = [[pstar_unp_arr_ls, pstar_bip_arr_ls], arg_input]
            sample_val_ls, _, _ = model_dict[tgt_p](input_x, None, pred_register.pred2ind[tgt_p],
                                                                 pstar_unp_ls, pstar_bip_ls,
                                                                infer_temp, infer_numSample)
            p_bar.update()
            p_val = sample_val_ls[-1][-1]
            query_rank += (cand_val <= p_val.transpose(0, 1)).sum(dim=-1).type(torch.float32) # worst rank

        query_rank -= (cand_val <= cand_val.transpose(0, 1)).sum(dim=-1).type(torch.float32)
        query_rank = query_rank.clamp(min=0)
        query_rank += 1
        mrr = (1 / query_rank).mean().item()
        hits = ((query_rank <= 10).sum().type(torch.float32) / query_rank.size(0)).item()
        mh_dict[tgt_p] = [mrr, hits, query_rank.size(0)]
        tqdm.write('mrr %.4f hits@10 %.4f cnt %i %s' % (mrr, hits, query_rank.size(0), tgt_p))
        p_bar.close()

    return mh_dict


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # train()
    if cmd_args.test_only:
        kb_test(cmd_args.output_root)
    else:
        train()
