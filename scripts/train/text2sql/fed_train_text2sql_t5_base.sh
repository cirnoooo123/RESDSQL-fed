set -e

# fed train text2sql-t5-base model
python -u fed_train.py \
    --client_num 5 \
    --batch_size 16 \
    --gradient_descent_step 2 \
    --device "0" \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 42 \
    --save_path "./models/text2sql-t5-base" \
    --tensorboard_save_path "./tensorboard_log/text2sql-t5-base" \
    --model_name_or_path "t5-base" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_spider.json"
