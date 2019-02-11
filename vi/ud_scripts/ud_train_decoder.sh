CODE_ROOT="PATH_TO_THIS_CODE_REPO/"
RESULT_DIR="PATH_TO_RESULT_DIR/"
WORD_VEC_DIR="PATH_TO_WORD_VECTOR_DIR/"
LANGUAGE="fr"
source /PATH_TO_PYTHON_ENV/activate &&
CUDA_VISIBLE_DEVICES=0  python ../train_decoder.py \
    --data_path $CODE_ROOT"data/ud" \
    --train_fname $LANGUAGE"_train_enhanced" \
    --valid_fname $LANGUAGE"_valid_enhanced" \
    --test_fname $LANGUAGE"_test_enhanced" \
    --cluster \
    --cluster_fname $CODE_ROOT"data/cluster/"$LANGUAGE"_cluster" \
    --word_vector_cache $WORD_VEC_DIR \
    --result_dir $RESULT_DIR \
    --log_name $LANGUAGE"_LOG_NAME.log" \
    --language $LANGUAGE \
    --seed -1 \
    --gpu_id 0