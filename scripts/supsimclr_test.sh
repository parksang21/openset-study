
python main.py --module supsimclr \
               --max_epochs 1000 \
               --accelerator ddp \
               --gpu 1 \
               --batch_size 1024 \
               --num_workers 48 \
               --lr 0.8 \
               --temperature 0.1 \
               --test \
               --path "./log/supsimclr/lightning_logs/success_0/checkpoints/epoch=999-step=58999.ckpt"