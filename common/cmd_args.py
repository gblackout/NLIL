import argparse


cmd_opt = argparse.ArgumentParser(description='argparser')


cmd_opt.add_argument('-data_root', default='../data/wn18', help='root folder to load data from')
cmd_opt.add_argument('-output_root', default='../run_logs', help='root folder to save run outputs')
cmd_opt.add_argument('-run_name', default='default', type=str, help='folder name for saving a run output')
cmd_opt.add_argument('-model_load_path', default=None, type=str, help='path to load pre-trained model')


cmd_opt.add_argument('-embedding_size', default=32, type=int, help='embedding size')
cmd_opt.add_argument('-n_head', default=4, type=int, help='number of attention heads')
cmd_opt.add_argument('-n_layer', default=3, type=int, help='number of encoder decoder block')
cmd_opt.add_argument('-channels', default=[1], type=int, nargs='*', help='num of combinations channels per stack')
cmd_opt.add_argument('-num_bs_instances', default=2, type=int, help='num of primitive statements')
cmd_opt.add_argument('-num_nested_calls', default=1, type=int, help='num of nested phi to be searched')
cmd_opt.add_argument('-no_recursion', default=True, action='store_false', dest='allow_recursion',
                     help='flag for disabling recursion')

cmd_opt.add_argument('-num_epochs', default=10, type=int, help='num epochs')
cmd_opt.add_argument('-num_batches_per_valid', default=100, type=int, help='num training batches before validation')
cmd_opt.add_argument('-batch_size', default=32, type=int, help='batch size for training')
cmd_opt.add_argument('-num_samples', default=1, type=int, help='num batches per epoch')
# cmd_opt.add_argument('-test_size', default=0.2, type=float, help='ratio of the size of the test set')
cmd_opt.add_argument('-no_rotate', default=True, action='store_false', dest='rotate',
                     help='flag for disabling rotation over tgt_pred_ls')

cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')
cmd_opt.add_argument('-lr_decay_factor', default=0.5, type=float, help='learning rate decay factor')
cmd_opt.add_argument('-lr_decay_patience', default=200, type=float, help='learning rate decay patience')
cmd_opt.add_argument('-lr_decay_min', default=0.00001, type=float, help='learning rate decay min')
cmd_opt.add_argument('-patience', default=20, type=int, help='patience for early stopping')
cmd_opt.add_argument('-l2_coef', default=0.0, type=float, help='L2 coefficient for weight decay')
cmd_opt.add_argument('-dropout_rate', default=0.0, type=int, help='rate of dropout')
cmd_opt.add_argument('-succ_threshold', default=0.5, type=float, help='threshold of accepting a base statement')
cmd_opt.add_argument('-recall_penalty', default=0.01, type=float, help='penalty for formula that does not apply')
cmd_opt.add_argument('-gumbelmax_temp', default=10, type=float, help='init temperature of gumbel max')
cmd_opt.add_argument('-attn_temp', default=1, type=float, help='temperature for transformer attention')
cmd_opt.add_argument('-neg_sample_ratio', default=1, type=int, help='ratio between pos and neg samples')
# cmd_opt.add_argument('-bs_temp', default=4, type=float, help='temperature of selecting base statement')
cmd_opt.add_argument('-avg_sample', default=True, action='store_false', dest='hard_sample',
                     help='flag for using averaging instead of hard samples')
cmd_opt.add_argument('-default_matmul', default=True, action='store_false', dest='const_matmul',
                     help='flag for using default matmul which is faster but will exceeds memory on FB dataset')
cmd_opt.add_argument('-dense_mat', default=True, action='store_false', dest='sparse_mat',
                     help='flag for using dense mat which is faster but will exceeds memory on FB dataset')
cmd_opt.add_argument('-no_sample_neg', default=True, action='store_false', dest='sample_neg',
                     help='')
cmd_opt.add_argument('-gqa', default=True, action='store_false', dest='kb',
                     help='')
cmd_opt.add_argument('-keep_empty', default=False, action='store_true', dest='keep_empty', help='')
cmd_opt.add_argument('-no_negate', default=False, action='store_true', dest='no_negate', help='')
cmd_opt.add_argument('-skip_trained', default=False, action='store_true', dest='skip_trained', help='')
cmd_opt.add_argument('-test_only', default=False, action='store_true', dest='test_only', help='')
cmd_opt.add_argument('-rand_eval', default=False, action='store_true', dest='rand_eval', help='')



cmd_opt.add_argument('-seed', default=10, type=int, help='random seed')
cmd_opt.add_argument('-device', default='cuda', type=str, help='run on cpu or cuda')


cmd_args, _ = cmd_opt.parse_known_args()

# wn18
cmd_args.hard_sample = False
cmd_args.allow_recursion = True
cmd_args.data_root = '../data/wn18'
cmd_args.embedding_size = 32
cmd_args.num_bs_instances = 1
cmd_args.num_nested_calls = 1
cmd_args.batch_size = 32
cmd_args.rotate = True
cmd_args.no_negate = True
cmd_args.sample_neg = False
cmd_args.skip_trained = True
cmd_args.patience = 20
cmd_args.num_epochs = 10
cmd_args.num_batches_per_valid = 800
cmd_args.lr_decay_patience = 800

# # fb15k
# cmd_args.hard_sample = False
# cmd_args.allow_recursion = True
# cmd_args.data_root = '../data/fb15k-237'
# cmd_args.num_batches_per_valid = 900
# cmd_args.lr_decay_patience = 700
# cmd_args.embedding_size = 32
# cmd_args.num_bs_instances = 1
# cmd_args.num_nested_calls = 2
# cmd_args.batch_size = 32
# cmd_args.rotate = False
# cmd_args.patience = 100
# cmd_args.num_epochs = 50
# cmd_args.no_negate = True
# cmd_args.sample_neg = False
# cmd_args.skip_trained = True

# # gqa
# cmd_args.hard_sample = False
# cmd_args.allow_recursion = False
# cmd_args.sample_neg = False
# cmd_args.sparse_mat = False
# cmd_args.data_root = '../data/gqa'
# cmd_args.kb = False
# cmd_args.num_batches_per_valid = 1
# cmd_args.lr_decay_patience = 9000
# cmd_args.num_bs_instances = 16
# cmd_args.num_nested_calls = 2
# cmd_args.patience = 20
# cmd_args.num_epochs = 50
# cmd_args.embedding_size = 32
# cmd_args.no_negate = True
# cmd_args.skip_trained = True
# cmd_args.rotate = True
# cmd_args.batch_size = 128

# # evensucc
# cmd_args.hard_sample = False
# cmd_args.allow_recursion = True
# cmd_args.data_root = '../data/evensucc10'
# cmd_args.embedding_size = 32
# cmd_args.num_bs_instances = 1
# cmd_args.num_nested_calls = 2
# cmd_args.batch_size = 32
# cmd_args.rotate = True
# cmd_args.no_negate = True
# cmd_args.sample_neg = False
# cmd_args.skip_trained = True
# cmd_args.patience = 20
# cmd_args.num_epochs = 10
# cmd_args.num_batches_per_valid = 5
# cmd_args.lr_decay_patience = 800