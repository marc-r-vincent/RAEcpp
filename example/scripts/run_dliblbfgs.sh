./run_rae \
    --labels \
        "./example/data/panglee.tgt" \
    --phrases \
        "./example/data/panglee.dat" \
    --train \
        "./example/data/panglee_train.idx" \
    --val \
        "./example/data/panglee_val.idx" \
    --test \
        "./example/data/panglee_test.idx" \
    --models_dir \
        "./example/models" \
    --predictions_dir \
        "./example/preds" \
    --representations_dir \
        "./example/representations" \
    --alpha "0.02" \
    --b_size 1024 \
    --opt_batch_num 3000 \
    --opt_type "dliblbfgs_batch" \
    --w_length 50 \
    --wait_min 1024 \
    --perf_type "accuracy" \
    --thread_num 8
