cd main

python train.py \
        -run_name fb \
        -data_root ../data/fb15k-237 \
        -embedding_size 32 \
        -num_bs_instances 1 \
        -num_nested_calls 2 \
        -batch_size 32 \
        -avg_sample \
        -no_negate \
        -no_rotate \
        -no_sample_neg \
        -skip_trained \
        -patience 100 \
        -num_epochs 50 \
        -num_batches_per_valid 900 \
        -lr_decay_patience 700