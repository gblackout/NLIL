from common.predicate import pred_register
from collections import Counter
import pickle
from common.utlis import flatten


class GQAFreq:

    def __init__(self, ht_dict_path, th_dict_path):

        with open(ht_dict_path, 'rb') as f:
            self.ht_dict = pickle.load(f)

        with open(th_dict_path, 'rb') as f:
            self.th_dict = pickle.load(f)

    def predict(self, pred2domain_dict, tgt_pred_ls):

        def get_rel_ls(c, fd):
            id2obj = {}
            rh_ls, rt_ls = [], []
            for pn in fd:
                if pred_register.is_unp(pn):
                    for val, consts in fd[pn]:
                        id2obj[consts[0]] = pn
            for pn in fd:
                if pred_register.is_unp(pn):
                    continue
                for val, consts in fd[pn]:
                    c1, c2 = consts
                    if c1 == c:
                        rt_ls.append([pn, id2obj[c2]])
                    elif c2 == c:
                        rh_ls.append([pn, id2obj[c1]])

            return rh_ls, rt_ls

        tgt_pred_ls = [pn for pn in pred_register.pred_dict if pred_register.is_unp(pn)]

        past_d_set = set()
        tgt_pred_dict = dict((pn, []) for pn in tgt_pred_ls)
        last_rank = len([pn for pn in pred_register.pred_dict if pred_register.is_unp(pn)])
        for domain_ls in pred2domain_dict.values():
            for domain in domain_ls:
                if domain.name in past_d_set:
                    continue
                else:
                    past_d_set.add(domain.name)

                for pn in domain.fact_dict:
                    if pn not in tgt_pred_dict:
                        continue

                    cand_cnt_dict = {}
                    for val, consts in domain.fact_dict[pn]:
                        rel_head_ls, rel_tail_ls = get_rel_ls(consts[0], domain.fact_dict)

                        for rel, head in rel_head_ls:
                            if rel not in self.ht_dict:
                                continue
                            if head not in self.ht_dict[rel]:
                                continue
                            obj_name, cnt = self.ht_dict[rel][head][0]
                            if obj_name in cand_cnt_dict:
                                cand_cnt_dict[obj_name] += cnt
                            else:
                                cand_cnt_dict[obj_name] = cnt

                        for rel, tail in rel_tail_ls:
                            if rel not in self.th_dict:
                                continue
                            if tail not in self.th_dict[rel]:
                                continue
                            obj_name, cnt = self.th_dict[rel][tail][0]
                            if obj_name in cand_cnt_dict:
                                cand_cnt_dict[obj_name] += cnt
                            else:
                                cand_cnt_dict[obj_name] = cnt

                    rank_ls = [e[0] for e in sorted(cand_cnt_dict.items(), key=lambda x:x[1], reverse=True)
                               if e[0] in tgt_pred_dict]

                    q_rank = last_rank
                    for ind, obj_name in enumerate(rank_ls):
                        if obj_name == pn:
                            q_rank = ind+1
                            break

                    tgt_pred_dict[pn].append(q_rank)

        tgt_ls = sorted([(pn, rank_ls) for pn, rank_ls in tgt_pred_dict.items()], key= lambda x:len(x[1]), reverse=True)[:150]

        tgt_pred_r1 = dict((pn, sum([r == 1 for r in rank_ls]) / len(rank_ls)) for pn, rank_ls in tgt_ls)
        tgt_pred_r5 = dict((pn, sum([r <= 5 for r in rank_ls]) / len(rank_ls)) for pn, rank_ls in tgt_ls)

        flat_rank_ls = flatten([e[1] for e in tgt_ls])
        r1 = sum([r == 1 for r in flat_rank_ls]) / len(flat_rank_ls)
        r5 = sum([r <= 5 for r in flat_rank_ls]) / len(flat_rank_ls)

        with open('freq_gqa_log.txt', 'w') as f:
            f.write('r1 %.4f r5 %.4f\n' % (r1, r5))
            for e in tgt_ls:
                f.write('r1 %.4f r5 %.4f %s\n' % (tgt_pred_r1[e[0]], tgt_pred_r5[e[0]], e[0]))