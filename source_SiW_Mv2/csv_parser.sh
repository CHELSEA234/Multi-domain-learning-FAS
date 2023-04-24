source ~/.bashrc
conda activate anti_spoofing
CUDA_NUM=0
python csv_parser.py --pro=1 --log_dir=../train_log
python csv_parser.py --pro=2 --log_dir=../train_log
python csv_parser.py --pro=3 --log_dir=../train_log