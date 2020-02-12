import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.Layers import EncoderLayer, DecoderLayer
from common.cmd_args import cmd_args
from model.SubLayers import FeedForwardConcat
from model.embedding import EmbeddingTable
from common.predicate import pred_register
from common import constants
from common.utlis import mask_select, is_empty, flatten
import time


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout):
        super().__init__()

        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                                          for _ in range(n_layers)])

    def forward(self, enc_input, enc_mask=None, return_attns=False, norm=True):

        enc_slf_attn_list = []
        slf_attn_mask, non_pad_mask = None, None

        # -- Prepare masks
        if enc_mask is not None:
            slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_mask, seq_q=enc_mask)
            non_pad_mask = get_non_pad_mask(enc_mask)

        # -- Forward
        enc_output = enc_input

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output,
                                                 non_pad_mask=non_pad_mask,
                                                 slf_attn_mask=slf_attn_mask,
                                                 norm=norm)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout):
        super().__init__()

        assert n_layers >= 1
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                                          for _ in range(n_layers - 1)])

        # TODO last layer only single head maybe improve later?
        self.layer_stack.append(DecoderLayer(d_model, d_inner, 1, d_k, d_v, dropout=dropout))

    def forward(self, dec_input, enc_output, dec_mask=None, return_attns=False, norm=True):

        last_sfm_attn, last_dec_enc_attn = None, None
        non_pad_mask, slf_attn_mask = None, None

        # -- Forward
        dec_output = dec_input

        for dec_layer in self.layer_stack:
            dec_output, sfm_attn, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                         non_pad_mask=non_pad_mask,
                                                                         slf_attn_mask=slf_attn_mask,
                                                                         dec_enc_attn_mask=dec_mask,
                                                                         norm=norm)

            if return_attns:
                last_sfm_attn = sfm_attn
                last_dec_enc_attn = dec_enc_attn

        if return_attns:
            return dec_output, last_sfm_attn, last_dec_enc_attn

        return dec_output


class BSTransformer(nn.Module):
    def __init__(self, pos_emb_table, pred_emb_table):
        super(BSTransformer, self).__init__()

        self.num_bs_instances = cmd_args.num_bs_instances
        self.num_nested_calls = cmd_args.num_nested_calls
        # self.bs_temp = cmd_args.bs_temp

        self.encoder = Encoder(n_layers=cmd_args.n_layer,
                               n_head=cmd_args.n_head,
                               d_k=cmd_args.embedding_size,
                               d_v=cmd_args.embedding_size,
                               d_model=cmd_args.embedding_size,
                               d_inner=cmd_args.embedding_size,
                               dropout=cmd_args.dropout_rate)

        self.decoder = Decoder(n_layers=cmd_args.n_layer,
                               n_head=cmd_args.n_head,
                               d_k=cmd_args.embedding_size,
                               d_v=cmd_args.embedding_size,
                               d_model=cmd_args.embedding_size,
                               d_inner=cmd_args.embedding_size,
                               dropout=cmd_args.dropout_rate)

        self.pos_emb_table = pos_emb_table
        self.pred_emb_table = pred_emb_table
        self.phi_emb_table = EmbeddingTable(1, cmd_args.embedding_size)
        # self.dummyarg_emb_table = EmbeddingTable(1, cmd_args.embedding_size)
        self.xy_emb_table = EmbeddingTable(2, cmd_args.embedding_size)
        self.bs_query_table = EmbeddingTable(self.num_bs_instances, cmd_args.embedding_size)

        self.ffn = FeedForwardConcat(cmd_args.embedding_size, cmd_args.embedding_size,
                                     dropout=cmd_args.dropout_rate)
        self.ffn2 = FeedForwardConcat(cmd_args.embedding_size, cmd_args.embedding_size,
                                      dropout=cmd_args.dropout_rate, residual=False)
        self.ffn3 = FeedForwardConcat(cmd_args.embedding_size, cmd_args.embedding_size,
                                      dropout=cmd_args.dropout_rate, residual=False)

    def forward(self, p_star, pstar_ind, unp_ls, bip_ls):

        num_noarg_phi, num_arg_phi = len(unp_ls), len(bip_ls)
        num_args = pred_register.get_numargs(p_star)

        # =========================== init phi emb ===========================
        p_star_emb = self.pred_emb_table([p_star])
        unp_emb, bip_emb = self.pred_emb_table(unp_ls), self.pred_emb_table(bip_ls)
        noarg_phi_emb = self.ffn(unp_emb + self.phi_emb_table([0]), p_star_emb.expand(num_noarg_phi, -1))
        # arg_phi_emb = self.ffn(bip_emb + self.phi_emb_table([0]), p_star_emb.expand(num_arg_phi, -1))

        init_input_emb = [self.xy_emb_table([arg_ind]) for arg_ind in range(num_args)]

        # all arg_phi except ident
        nest_query = self.ffn3(bip_emb[:-1] + self.phi_emb_table([0]), p_star_emb.expand(num_arg_phi-1, -1))
        nest_sfm_attn_ls, nest_attn_ls, nest_emb_ls = [], [], [([] if len(unp_ls) == 0 else [noarg_phi_emb]) +
                                                               init_input_emb]
        num_channel = len(nest_emb_ls[0])
        xchannel_ind = 0 if num_noarg_phi == 0 else 1
        for nest_ind in range(self.num_nested_calls):
            nest_input_ls = nest_emb_ls[-1]

            sfm_attn_channel_ls, attn_channel_ls, emb_channel_ls = [], [], []
            for channel_ind, nest_input in enumerate(nest_input_ls):

                # TODO debug
                n_query = nest_query
                if (nest_ind == 0) and (channel_ind == xchannel_ind):
                    if pstar_ind is not None:
                        n_query = torch.cat([nest_query[:pstar_ind], nest_query[pstar_ind + 1:]], dim=0)


                _, nest_emb, sfm_nest_attn, nest_attn, _ = self.encode_decode(n_query, nest_input)
                sfm_attn_channel_ls.append(sfm_nest_attn)
                attn_channel_ls.append(nest_attn)
                emb_channel_ls.append(nest_emb)

            nest_sfm_attn_ls.append(sfm_attn_channel_ls)
            nest_attn_ls.append(attn_channel_ls)
            nest_emb_ls.append(emb_channel_ls)

        all_channel_emb_ls = [torch.cat([emb_channel_ls[ch_ind] for emb_channel_ls in nest_emb_ls], dim=0)
                              for ch_ind in range(num_channel)]
        bs_query = self.bs_query_table(list(range(self.num_bs_instances)))
        bs_query = self.ffn(bs_query, p_star_emb.expand(self.num_bs_instances, -1))
        src_sfm_attn_ls, src_attn_ls, src_emb_ls = [], [], []
        for channel_emb in all_channel_emb_ls:
            _, src_emb, sfm_src_attn, src_attn, _ = self.encode_decode(bs_query, channel_emb)
            src_sfm_attn_ls.append(sfm_src_attn)
            src_attn_ls.append(src_attn)
            src_emb_ls.append(src_emb)

        bs_emb_ls = []
        for i in range(len(src_emb_ls)):
            for j in range(i+1, len(src_emb_ls)):
                bs_emb_ls.append(self.ffn2(src_emb_ls[i], src_emb_ls[j]))

        bs_emb = torch.cat(bs_emb_ls, dim=0)
        # TODO
        ident_ind_ls = None

        return bs_emb, src_sfm_attn_ls, src_attn_ls, nest_sfm_attn_ls, nest_attn_ls, ident_ind_ls

    def encode_decode(self, query, phi_embedding, noarg_prob=None,
                      tgt_noarg_prob=None, enc_output=None, pstar_mask=None):
        """

        :param query:
            (K, latent_dim)
        :param phi_embedding:
            (N, latent_dim)
        :param noarg_prob:
            (N)
        :param rand_mask:
            (K)
        :return:
        """

        # encode
        enc_output = self.encoder(phi_embedding.unsqueeze(0)) if enc_output is None else enc_output.unsqueeze(0)

        # decode
        choice, sfm_attn, attn = self.decoder(query.unsqueeze(0), enc_output,
                                              dec_mask=pstar_mask, return_attns=True)

        if noarg_prob is not None:
            noarg_prob = sfm_attn.squeeze(0).matmul(noarg_prob)  # (K)

        return enc_output.squeeze(0), choice.squeeze(0), sfm_attn.squeeze(0), attn.squeeze(0), noarg_prob


class LogicTransformer(nn.Module):

    def __init__(self, pos_emb_table, pred_emb_table):
        super(LogicTransformer, self).__init__()

        self.pos_emb_table = pos_emb_table
        self.pred_emb_table = pred_emb_table

        self.channels = cmd_args.channels
        assert all(e > 0 for e in self.channels)
        assert self.channels[-1] == 1

        self.encoder = Encoder(n_layers=cmd_args.n_layer,
                               n_head=cmd_args.n_head,
                               d_k=cmd_args.embedding_size,
                               d_v=cmd_args.embedding_size,
                               d_model=cmd_args.embedding_size,
                               d_inner=cmd_args.embedding_size,
                               dropout=cmd_args.dropout_rate)

        self.decoder = Decoder(n_layers=cmd_args.n_layer,
                               n_head=cmd_args.n_head,
                               d_k=cmd_args.embedding_size,
                               d_v=cmd_args.embedding_size,
                               d_model=cmd_args.embedding_size,
                               d_inner=cmd_args.embedding_size,
                               dropout=cmd_args.dropout_rate)

        self.ffn = FeedForwardConcat(cmd_args.embedding_size, cmd_args.embedding_size,
                                     dropout=cmd_args.dropout_rate)
        self.ffn2 = FeedForwardConcat(cmd_args.embedding_size, cmd_args.embedding_size,
                                     dropout=cmd_args.dropout_rate, residual=False)
        self.ffn3 = FeedForwardConcat(cmd_args.embedding_size, cmd_args.embedding_size,
                                      dropout=cmd_args.dropout_rate)
        self.candidate_query_table_ls = nn.ModuleList([EmbeddingTable(e, cmd_args.embedding_size)
                                                       for e in self.channels])

        self.sign_emb_tab = EmbeddingTable(2, cmd_args.embedding_size)

        self.layer_norm1 = nn.LayerNorm(cmd_args.embedding_size)
        self.layer_norm2 = nn.LayerNorm(cmd_args.embedding_size)
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, p_star, bs_emb):

        p_star_emb = self.pred_emb_table([p_star])

        attn_ls, sfm_attn_ls = [], []
        last_level = len(self.channels) - 1
        init_bs_emb = torch.cat([bs_emb + self.sign_emb_tab([ind]) for ind in [0, 1]], dim=0)
        candidate_emb_ls = [init_bs_emb]
        for depth, num_channel in enumerate(self.channels):
            if depth == last_level:
                candidate_emb = torch.cat(candidate_emb_ls, dim=0)
            else:
                candidate_emb = candidate_emb_ls[-1]

            candidate_emb = self.encoder(candidate_emb.unsqueeze(0)).squeeze(0)  # (N, latent_dim)

            cand_query = self.candidate_query_table_ls[depth](list(range(num_channel)))
            cand_query = self.ffn3(cand_query, p_star_emb.expand(num_channel, -1))

            left_emb, sfm_left_attn, left_attn = self.decoder(cand_query.unsqueeze(0), candidate_emb.unsqueeze(0),
                                                         return_attns=True, norm=False)
            left_emb, sfm_left_attn, left_attn = left_emb.squeeze(0), sfm_left_attn.squeeze(0), left_attn.squeeze(0)
            sfm_right_attn, right_attn = None, None
            if depth != last_level:
                # choose right
                right_query = self.ffn(cand_query, self.layer_norm1(left_emb))
                right_emb, sfm_right_attn, right_attn = self.decoder(right_query.unsqueeze(0), candidate_emb.unsqueeze(0),
                                                               return_attns=True, norm=False)
                right_emb, sfm_right_attn, right_attn = right_emb.squeeze(0), sfm_right_attn.squeeze(0), right_attn.squeeze(0)
                candidate_emb = self.ffn2(left_emb, right_emb)

                candidate_emb = torch.cat([candidate_emb + self.sign_emb_tab([ind]) for ind in [0, 1]], dim=0)
                candidate_emb_ls.append(candidate_emb)

            sfm_attn_ls.append([sfm_left_attn, sfm_right_attn])
            attn_ls.append([left_attn, right_attn])

            return sfm_attn_ls, attn_ls


class RuleLearner(nn.Module):

    def __init__(self, num_pred, phi_list):
        super(RuleLearner, self).__init__()

        self.pos_emb_table = EmbeddingTable(2, cmd_args.embedding_size)
        self.pred_emb_table = EmbeddingTable(num_pred, cmd_args.embedding_size)
        self.bbs_trans = BSTransformer(self.pos_emb_table, self.pred_emb_table)
        self.logic_trans = LogicTransformer(self.pos_emb_table, self.pred_emb_table)

        self.phi_list = phi_list
        # self.num_inst_per = cmd_args.num_phi_instances
        self.bs_thres = cmd_args.succ_threshold
        self.recall_penalty = cmd_args.recall_penalty

    def forward(self, input_x, input_y, p_star, unp_name_ls, bip_name_ls, tau, num_sample, visualize=True):
        # Ident must be the last pred in bip_ls
        assert pred_register.get_class(bip_name_ls[-1]).name == constants.IDENT_PHI

        unp_ind_ls = [pred_register.pred2ind[pn] for pn in unp_name_ls]
        bip_ind_ls = [pred_register.pred2ind[pn] for pn in bip_name_ls]
        num_args = pred_register.get_numargs(p_star)
        num_unp, num_bip = len(unp_name_ls), len(bip_name_ls)
        if cmd_args.allow_recursion:
            try:
                pstar_ind = [unp_ind_ls, bip_ind_ls][num_args - 1].index(p_star)
            except:
                pstar_ind = None
        else:
            pstar_ind = None

        # name ls for later visualization
        init_name_ls = ([] if num_unp == 0 else [['phi_%s()' % p_name for p_name in unp_name_ls]])\
                       + [['x'], ['y']][:num_args]
        # for var_name in ['x', 'y'][:num_args]:
        #     init_name_ls.append(['phi_%s(%s)' % (p_name, var_name) for p_name in bip_name_ls])
        nest_name_template = ['phi_%s' % p_name for p_name in bip_name_ls[:-1]]
        # bs_name_template = unp_name_ls + bip_name_ls[:-1]

        # =========================== generate attention ===========================

        bs_emb, \
        src_sfm_attn_ls, src_attn_ls, \
        nest_sfm_attn_ls, nest_attn_ls, \
        ident_ind_ls = self.bbs_trans(p_star, pstar_ind, unp_ind_ls, bip_ind_ls)

        # transformer on logic
        logic_sfm_attn_ls, logic_attn_ls = self.logic_trans(p_star, bs_emb)

        # =========================== gumble max sampling and eval ===========================
        sample_val_ls, mask_ls, name_ls = [], [], []
        for sample_ind in range(num_sample):

            if cmd_args.hard_sample:
                nest_sample_ls, src_sample_ls, logic_sample_ls = \
                    self.gumbel_max_sample(unp_name_ls, bip_name_ls, nest_attn_ls, src_attn_ls,
                                           logic_attn_ls, tau, ident_ind_ls, pstar_ind)


            else:
                nest_sample_ls = nest_sfm_attn_ls
                src_sample_ls = src_sfm_attn_ls
                logic_sample_ls = logic_sfm_attn_ls

            # visualize formulas
            nest_name_ls, bs_name_ls, rule_name_ls = None, None, None
            if visualize:
                nest_name_ls, bs_name_ls, rule_name_ls = self.visualize_rule_sample(init_name_ls, nest_name_template,
                                                                   nest_sample_ls, src_sample_ls, logic_sample_ls,
                                                                   pstar_ind)

            # =========================== solving base statement ===========================
            bs_val = self.eval_base_statement2(input_x, unp_name_ls, bip_name_ls,
                                               nest_sample_ls, src_sample_ls, pstar_ind)

            # =========================== solving candidate ===========================
            val_ls = self.eval_rule_candidate(input_y, bs_val, logic_sample_ls)


            sample_val_ls.append(val_ls)
            # mask_ls.extend(sample_mask_ls)
            name_ls.append([init_name_ls, nest_name_ls, rule_name_ls])

        return sample_val_ls, mask_ls, name_ls

    def eval_base_statement2(self, input_x, unp_ls, bip_ls, nest_sample_ls, src_sample_ls, pstar_ind):
        """
        :param input_x:
            unp_arr_ls: [num_unp, (1, num_ent, 1)]
            bip_arr_ls: [num_bip, (1, num_ent, num_ent)]
            arg_input: (bsize, num_args, num_ents)
        :param unp_ls:
        :param bip_ls:
        :param nested_phi_sample:
            (num_arg_phi, (num_load_noarg_phi+num_load_arg_phi))
        :param bs_sample_ls:
            [(num_un_pred, 1, num_inst), (num_bi_pred, 2, num_inst)]
        :return:
        """

        noarg_input, arg_input = input_x
        filtered_bip_ls = bip_ls[:-1]
        bsize, num_arg, num_const = arg_input.size(0), arg_input.size(1), arg_input.size(2)
        finalvec = lambda x: x if x is None else torch.max(x, dim=-1)[0]

        def apply_attn(attn, mat):
            num_q = attn.size(0)
            mat_numArg = mat.size(1)
            if num_q == 0:
                return mat
            attn = attn.unsqueeze(0).unsqueeze(2)
            mat = mat.view(bsize, mat_numArg, -1)
            res = attn.matmul(mat.unsqueeze(1)).squeeze(2).view(bsize, num_q, num_const, -1)
            return res

        un_src = self.phi_list(noarg_input, unp_ls)
        if un_src is not None:
            un_src = un_src.unsqueeze(0).expand(bsize, -1, -1, -1)

        init_input_ls = ([] if un_src is None else [un_src]) + \
                        [arg_input[:, i, :].unsqueeze(-1).unsqueeze(1) for i in range(num_arg)]
        num_channel = len(init_input_ls)
        nest_input = init_input_ls
        phi_output = [nest_input]
        xchannel_ind = 0 if un_src is None else 1
        for nest_ind, sample_channel_ls in enumerate(nest_sample_ls):
            tmp = []
            for channel_ind, nest_sample in enumerate(sample_channel_ls):

                # TODO debug
                f_bip_ls = filtered_bip_ls
                na_input = noarg_input
                if (nest_ind == 0) and (channel_ind == xchannel_ind):
                    if pstar_ind is not None:
                        f_bip_ls = [e for ind,e in enumerate(filtered_bip_ls) if ind!=pstar_ind]
                        na_input = [noarg_input[0], [e for ind,e in enumerate(noarg_input[1]) if ind!=pstar_ind]]

                _, bi_src, _, _ = self.apply_phi2(na_input, nest_input[channel_ind], None, f_bip_ls,
                                                  bi_src_attn=nest_sample)
                tmp.append(bi_src)
            phi_output.append(tmp)
            nest_input = tmp

        all_channel_output = [torch.cat([out[channel_ind] for out in phi_output], dim=1)
                              for channel_ind in range(num_channel)]

        src_val_ls = []
        for channel_ind, src_sample in enumerate(src_sample_ls):
            src_val_ls.append(apply_attn(src_sample, all_channel_output[channel_ind]))

        src_val_ls = [finalvec(e) for e in src_val_ls]
        bs_val_ls = []
        for i in range(num_channel):
            for j in range(i+1, num_channel):
                bs_val_ls.append(torch.sum(src_val_ls[i] * src_val_ls[j], dim=-1).clamp(min=0, max=1))

        val = torch.cat(bs_val_ls, dim=-1)
        return val


    def apply_phi2(self, noarg_input, arg_input, unp_ls, bip_ls, bi_src_attn=None, un_tgt_attn=None, bi_tgt_attn=None):
        """

        :param noarg_input:
            unp_arr_ls: [num_unp, (1, num_const, 1)]
            bip_arr_ls: [num_bip, (1, num_const, num_const)]
        :param arg_input:
            (b, num_arg, num_const, num_inst)
        :param unp_ls:
        :param bip_ls:
        :param bi_src_attn:
            (num_bip, num_args)
        :param un_tgt_attn:
            (num_unp, num_args)
        :param bi_tgt_attn:
            (num_bip, num_args)
        :return:
        """

        bsize, num_arg, num_const = arg_input.size(0), arg_input.size(1), arg_input.size(2)
        un_tgt, bi_tgt = None, None

        if unp_ls is None:
            un_src = noarg_input
        else:
            # (num_unp, num_const, num_inst)
            un_src = self.phi_list(noarg_input, unp_ls)
            if un_src is None:
                un_src = None #torch.ones(bsize, 0, num_const, 1, device=cmd_args.device)
            else:
                un_src = un_src.unsqueeze(0).expand(bsize, -1, -1, -1)

        def apply_attn(attn, mat):
            num_q = attn.size(0)
            if num_q == 0:
                return mat
            attn = attn.unsqueeze(0).unsqueeze(2)
            mat = mat.view(bsize, num_arg, -1)
            res = attn.matmul(mat.unsqueeze(1)).squeeze(2).view(bsize, num_q, num_const, -1)
            return res

        if bi_src_attn is None:
            init_input = arg_input.unsqueeze(1).expand(-1, len(bip_ls), -1, -1, -1)
            init_res = [self.phi_list(noarg_input, bip_ls, init_input[:, :, ind, :, :]) for ind in range(num_arg)]
            bi_src = torch.cat(init_res, dim=1) # (b, num_bip*num_arg, num_const, num_inst)
        else:
            bi_src_input = apply_attn(bi_src_attn, arg_input) # (b, num_bip, num_const, num_inst)
            bi_src = self.phi_list(noarg_input, bip_ls, bi_src_input)

        if un_tgt_attn is not None:
            un_tgt = apply_attn(un_tgt_attn, arg_input)

        if bi_tgt_attn is not None:
            bi_tgt = apply_attn(bi_tgt_attn, arg_input)

        return un_src, bi_src, un_tgt, bi_tgt


    def visualize_rule_sample(self, init_name_ls, nest_name_template, nest_sample_ls,
                              src_sample_ls, logic_sample_ls, pstar_ind):

        # visualize formulas
        nest_name_ls = [init_name_ls]
        nest_input_name_ls = init_name_ls
        num_channel = len(init_name_ls)
        xchannel_ind = 0 if init_name_ls[0][0] == 'x' else 1
        for nest_ind, sample_channel_ls in enumerate(nest_sample_ls):

            channel_name_ls = []
            for channel_ind, nest_sample in enumerate(sample_channel_ls):
                n_name_template = nest_name_template
                if (nest_ind == 0) and (channel_ind == xchannel_ind):
                    if pstar_ind is not None:
                        n_name_template = [e for ind,e in enumerate(nest_name_template) if ind!=pstar_ind]

                tmp = []
                for ind, p_ind in enumerate(nest_sample.max(dim=-1)[-1]):
                    tmp.append('%s(%s)' % (n_name_template[ind], nest_input_name_ls[channel_ind][p_ind]))
                channel_name_ls.append(tmp)

            nest_input_name_ls = channel_name_ls
            nest_name_ls.append(channel_name_ls)

        all_channel_name_ls = []
        for channel_ind in range(num_channel):
            tmp = []
            for channel_name_ls in nest_name_ls:
                tmp.extend(channel_name_ls[channel_ind])
            all_channel_name_ls.append(tmp)

        src_name_ls = []
        for channel_ind, src_sample in enumerate(src_sample_ls):
            tmp = []
            for src_ind in src_sample.max(dim=-1)[-1]:
                tmp.append(all_channel_name_ls[channel_ind][src_ind])
            src_name_ls.append(tmp)

        bs_name_ls = []
        for i in range(num_channel):
            for j in range(i+1, num_channel):
                bs_name_ls.extend(['%s=%s' % (src_name, src_name_ls[j][bs_ind])
                                   for bs_ind, src_name in enumerate(src_name_ls[i])])

        rule_name_ls = [['(%s)' % e for e in bs_name_ls] + [('(%s)' if cmd_args.no_negate else '!(%s)') % e
                                                            for e in bs_name_ls]]
        last_level = len(logic_sample_ls) - 1
        for level_ind, level in enumerate(logic_sample_ls):
            if level_ind == last_level:
                rule_name = flatten(rule_name_ls)
            else:
                rule_name = rule_name_ls[-1]
            left_sample, right_sample = level
            left_choice_ind = left_sample.max(dim=-1)[-1]
            right_choice_ind = None if right_sample is None else (right_sample.max(dim=-1)[-1])
            tmp_ls = []

            for left_ind, left_choice in enumerate(left_choice_ind):
                if level_ind == last_level:
                    tmp_ls.append('%s' % (rule_name[left_choice]))
                else:
                    tmp_ls.append('(%s ^ %s)' % (rule_name[left_choice], rule_name[right_choice_ind[left_ind]]))

            tmp_ls = tmp_ls + ([] if (level_ind == last_level) else ['!%s' % e for e in tmp_ls])
            rule_name_ls.append(tmp_ls)


        return nest_name_ls, bs_name_ls, rule_name_ls

    def eval_rule_candidate(self, input_y, bs_val, logic_sample_ls):
        """

        :param bs_val:
            (b, num_bs)
        :param logic_sample_ls:
            [L, 2, (C_l, C_l-1)]
        :return:

        """

        bs_succ_mask = (bs_val > self.bs_thres).type(torch.float)

        loss = 0.0
        # cand_val, cand_mask = bs_val, bs_succ_mask
        # cand_ls, mask_ls = [], []
        if cmd_args.no_negate:
            init_val = torch.cat([bs_val, bs_val], dim=1)
        else:
            init_val = torch.cat([bs_val, 1 - bs_val], dim=1)
        val_ls = [init_val]
        last_level = len(logic_sample_ls) - 1
        for level_ind, level in enumerate(logic_sample_ls):
            left_sample, right_sample = level
            if last_level == level_ind:
                rule_val = torch.cat(val_ls, dim=1)
            else:
                rule_val = val_ls[-1]

            left_val = rule_val.matmul(left_sample.transpose(0, 1))

            if last_level == level_ind:
                rule_val = left_val
                val_ls.append(rule_val)

            else:
                right_val = rule_val.matmul(right_sample.transpose(0, 1))
                rule_val = left_val * right_val
                val_ls.append(torch.cat([rule_val, 1 - rule_val], dim=1))


        loss /= len(logic_sample_ls) + 1

        return val_ls


    def gumbel_max_sample(self, unp_ls, bip_ls, nested_attn_ls, src_attn_ls, logic_attn_ls, tau,
                          ident_ind_ls, pstar_ind):
        """
        :param nested_attn:
             (1, num_arg_phi, (num_load_noarg_phi+num_load_arg_phi))
        :param slot_attn_ls:
            [(num_un_pred, 1, num_inst), (num_bi_pred, 2, num_inst)]
        :param logic_attn_ls:
            [L, 2, (C_l, C_l-1)]
        :return:

        """

        get_sample = lambda x: None if x is None else F.gumbel_softmax(x / cmd_args.attn_temp, tau=tau, hard=True, dim=-1)
        num_args = None if ident_ind_ls is None else len(ident_ind_ls)
        num_noarg_phi, num_arg_phi = len(unp_ls), len(bip_ls)

        nest_sample_ls = []
        for attn_channel_ls in nested_attn_ls:
            sample_channel_ls = []
            for nest_attn in attn_channel_ls:
                sample_channel_ls.append(get_sample(nest_attn))
            nest_sample_ls.append(sample_channel_ls)

        src_sample_ls = [get_sample(src_attn) for src_attn in src_attn_ls]

        logic_sample_ls = []
        for level_ind, level in enumerate(logic_attn_ls):
            left_attn, right_attn = level
            left_sample, right_sample = get_sample(left_attn), (None if right_attn is None else get_sample(right_attn))
            logic_sample_ls.append([left_sample, right_sample])

        return nest_sample_ls, src_sample_ls, logic_sample_ls


class PhiList(nn.Module):

    def __init__(self, full_unp_ls, full_bip_ls):
        super(PhiList, self).__init__()

        self.full_unp_ls = nn.ModuleDict(full_unp_ls)
        self.full_bip_ls = nn.ModuleDict(full_bip_ls)

    def forward(self, noarg_input, p_ls, arg_input=None):
        """

        :param noarg_input:
            (b, dx1, *)
        :param p_ls:
            [len_p]
        :param arg_input:
            (b, num_args, dx1, *)
        :return:
            (b, num_args or 1, len_p, dx1, *)
        """
        if arg_input is None:
            un_res = torch.stack([self.full_unp_ls[str(phi_ind)](noarg_input, arg_input)
                                  for phi_ind in p_ls], dim=1).unsqueeze(dim=1)  # (b, 1, len_p, dx)
            return un_res
        else:
            bi_res = torch.stack([self.full_bip_ls[str(phi_ind)](noarg_input, arg_input)
                                  for phi_ind in p_ls], dim=2)  # (b, num_arg, len_p, dx)
            return bi_res


class TabularPhiList(PhiList):

    def __init__(self, full_unp_ls, full_bip_ls):

        full_unp_ls = [('phi_' + pn, phi) for pn, phi in full_unp_ls]
        full_bip_ls = [('phi_' + pn, phi) for pn, phi in full_bip_ls]
        super(TabularPhiList, self).__init__(full_unp_ls, full_bip_ls)

    def forward(self, noarg_input, p_ls, arg_input=None):
        unp_arr_ls, bip_arr_ls = noarg_input
        p_ls = ['phi_'+pn for pn in p_ls]

        if arg_input is None:
            # [num_unp, (num_const, num_inst)]
            un_res_ls = [self.full_unp_ls[pn](unp_arr_ls[ind]) for ind, pn in enumerate(p_ls)]

            if len(un_res_ls) == 0:
                return None
            return torch.stack(un_res_ls, dim=0) # (num_unp, num_const, num_inst)

        else:
            # [num_bip, [(1, num_const, num_const), (b, num_const, num_inst)]]
            bi_res_ls = [self.full_bip_ls[pn](bip_arr_ls[ind], arg_input[:, ind, :, :]) for ind, pn in enumerate(p_ls)]
            return torch.stack(bi_res_ls, dim=1) # (b, num_bip, num_const, num_inst)


def ind2mask(total_len, ind_ls):
    mask = torch.zeros(total_len).scatter_add(0, torch.tensor(ind_ls), torch.ones(len(ind_ls)))
    return mask.eq(1).view(-1, 1).to(cmd_args.device)
