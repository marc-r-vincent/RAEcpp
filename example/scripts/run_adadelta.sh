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
    --alpha "0.2" \
    --b_size 64 \
    --lambda_reg_L 0.001 \
    --lambda_reg_c 0.001 \
    --lambda_reg_h 0.001 \
    --lambda_reg_r 0.001 \
    --learning_rate "0.1" \
    --opt_batch_num 3000 \
    --opt_type "adadelta_minibatch" \
    --perf_type "accuracy" \
    --rho "0.9" \
    --thread_num 8 \
    --w_length 50 \
    --wait_increase 2 \
    --wait_min 1024
