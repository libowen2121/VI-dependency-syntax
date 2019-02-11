CODE_ROOT="PATH_TO_THIS_CODE_REPO/"
RESULT_DIR="PATH_TO_RESULT_DIR/"
WORD_VEC_DIR="PATH_TO_WORD_VECTOR_DIR/"
LANGUAGE="fr"
source /PATH_TO_PYTHON_ENV/activate &&
python ../nvil_pre_ud.py \
    --data_path $CODE_ROOT"data/ud" \
    --train_fname $LANGUAGE"_train_enhanced" \
    --valid_fname $LANGUAGE"_valid_enhanced" \
    --test_fname $LANGUAGE"_test_enhanced" \
    --cluster \
    --cluster_fname $CODE_ROOT"data/cluster/"$LANGUAGE"_cluster" \
    --word_vector_cache $WORD_VEC_DIR \
    --result_dir $RESULT_DIR \
    --log_name $LANGUAGE"_LOG_NAME.log" \
    --pr_fname $CODE_ROOT"data/pr_rules/ud_c/"$LANGUAGE"_10_gt.txt" \
    --word_dim 0 \
    --language $LANGUAGE \
    --gpu_id -1 \
    --seed -1