
python main.py --module supsimclr \
               --max_epochs 1000 \
               --accelerator ddp \
               --gpu 1 \
               --batch_size 256 \
               --num_workers 48 \
               --lr 0.8 \
               --temperature 0.1 \
               --train --test