
python main.py --module ce_supsim \
               --max_epochs 10 \
               --accelerator ddp \
               --gpu 2 \
               --batch_size 1024 \
               --num_workers 48 \
               --lr 0.001 \
               --train \
               --path lightning_logs/success_0/checkpoints/epoch=999-step=14999.ckpt