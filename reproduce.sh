python3 train.py --embedding_dim 256 --dropout 0.4 --batch_size 1024 --window 10 --weight_decay 1e-4 --lr 2e-4 --epochs 25 --classifier ensemble --load_path .
python3 train.py --embedding_dim 256 --dropout 0.4 --batch_size 1024 --window 10 --weight_decay 1e-4 --lr 2e-4 --epochs 25 --classifier KNN_js --load_path .
python3 train.py --embedding_dim 256 --dropout 0.4 --batch_size 1024 --window 10 --weight_decay 1e-4 --lr 2e-4 --epochs 25 --classifier KNN_l1 --load_path .
python3 train.py --embedding_dim 256 --dropout 0.4 --batch_size 1024 --window 10 --weight_decay 1e-4 --lr 2e-4 --epochs 25 --classifier KNN_l2 --load_path .
python3 train.py --embedding_dim 256 --dropout 0.4 --batch_size 1024 --window 10 --weight_decay 1e-4 --lr 2e-4 --epochs 25 --classifier chain --load_path .

python3 train.py --model_type LSTM --dropout 0.4 --weight_decay 5e-5 --load_path .
