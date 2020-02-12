cd main

python train.py \
        -run_name wn \
        -data_root ../data/wn18 \
        -embedding_size 32 \
        -num_bs_instances 1 \
        -num_nested_calls 1 \
        -batch_size 32 \
        -avg_sample \
        -no_negate \
        -no_sample_neg \
        -skip_trained \
        -patience 20 \
        -num_epochs 10 \
        -num_batches_per_valid 800 \
        -lr_decay_patience 800