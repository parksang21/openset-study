
python main.py --module ce_supsim \
               --max_epochs 10 \
               --accelerator ddp \
               --gpu 1 \
               --batch_size 512 \
               --num_workers 48 \
               --lr 0.001 \
               --train \
               --test \
               --path "./log/supsimclr/lightning_logs/success_0/checkpoints/epoch=999-step=58999.ckpt"