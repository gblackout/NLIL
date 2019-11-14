from common.cmd_args import cmd_args
from common.utlis import get_output_folder, makedir
from os.path import join as joinpath
import math
import random


def generate_evenodd_data(tlog=False, seq_len=10, name='evenodd'):

    output_path = joinpath(cmd_args.data_root, name)
    makedir(output_path, remove_old=True)

    ent_path = joinpath(output_path, 'entities.txt') if tlog else joinpath(output_path, 'ent.txt')
    fact_path = joinpath(output_path, 'facts.txt') if tlog else joinpath(output_path, 'fact.txt')
    pred_path = joinpath(output_path, 'relations.txt') if tlog else joinpath(output_path, 'pred.txt')
    train_path = joinpath(output_path, 'train.txt')
    valid_path = joinpath(output_path, 'valid.txt')
    test_path = joinpath(output_path, 'test.txt')

    if tlog:
        with open(pred_path, 'w') as f:
            f.write('zero\n')
            f.write('even\n')
            f.write('succ\n')
    else:
        with open(pred_path, 'w') as f:
            f.write('zero(int)\n')
            f.write('even(int)\n')
            f.write('succ(int,int)\n')

    num_seq = list(range(seq_len))

    if tlog:

        with open(ent_path, 'w') as f:
            for n in num_seq:
                f.write('%i\n' % n)

        random.shuffle(num_seq)
        split_cnt = math.ceil(len(num_seq) / 10)
        inds = [[fact_path, 0, split_cnt*5],
                [train_path, split_cnt*5, split_cnt*7],
                [valid_path, split_cnt*7, split_cnt*8],
                [test_path, split_cnt*8, len(num_seq)]]

        for ind, parts in enumerate(inds):
            fp, s, e = parts
            with open(fp, 'w') as f:
                if ind == 0:
                    f.write('%i\tzero\t%i\n' % (0, 0))

                    for num in num_seq:
                        if num > 0:
                            f.write('%i\tsucc\t%i\n' % (num - 1, num))

                for i, num in enumerate(num_seq[s:e]):

                    if num % 2 == 0:
                        f.write('%i\teven\t%i\n' % (num, num))

    else:
        for fp in [fact_path, valid_path, test_path]:
            with open(fp, 'w') as f:
                for i, num in enumerate(num_seq):

                    zero_val = int(num == 0)
                    even_val = int(num % 2 == 0)

                    f.write('%i\tzero(%i)\n' % (zero_val, num))
                    f.write('%i\teven(%i)\n' % (even_val, num))

                    for j in range(seq_len):
                        succ_val = int(num+1 == num_seq[j])
                        f.write('%i\tsucc(%i,%i)\n' % (succ_val, num, num_seq[j]))


def generate_kniship_data():
    pass







if __name__ == '__main__':
    random.seed(10)

    cmd_args.data_root = '../data'
    generate_evenodd_data(tlog=False, seq_len=10, name='evensucc10')