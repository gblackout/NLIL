cd main

python train.py \
        -run_name gqa \
        -data_root ../data/gqa \
        -embedding_size 32 \
        -num_bs_instances 16 \
        -num_nested_calls 2 \
        -batch_size 128 \
        -gqa \
        -dense_mat \
        -avg_sample \
        -no_negate \
        -no_sample_neg \
        -skip_trained \
        -patience 20 \
        -num_epochs 50 \
        -num_batches_per_valid 1 \
        -lr_decay_patience 9000