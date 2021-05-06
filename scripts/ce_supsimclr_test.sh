
python main.py --module ce_supsim \
               --max_epochs 10 \
               --accelerator ddp \
               --gpu 1 \
               --batch_size 128 \
               --num_workers 48 \
               --lr 0.001 \
               --test \
               --path "./log/ce_supsim/lightning_logs/success_0/checkpoints/epoch=9-step=299.ckpt"