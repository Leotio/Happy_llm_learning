CUDA_VISIBLE_DEVICES=0

deepspeed s4_train_withDeepspeed.py \
    --config_name Happy_llm_learning/models/Qwen_2.5_1.5B \
    --tokenizer_name Happy_llm_learning/models/Qwen_2.5_1.5B \
    --train_files datasets/mobvoi_seq_monkey_general_open_c/mobvoi_seq_monkey_general_open_corpus.jsonl \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir Happy_llm_learning/Chapert6/PreTrain_withHF \
    --evaluation_strategy  no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir Happy_llm_learning/Chapert6/PreTrain_withHF/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \

    # ds_config_zero2.json 是 DeepSpeed 的核心配置文件。
    # 它的作用是告诉 DeepSpeed 如何在多张显卡之间分配内存、如何做并行计算以及如何优化显存。