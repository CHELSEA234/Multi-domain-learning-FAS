source ~/.bashrc
conda activate anti_spoofing
CUDA_NUM=0
# python inference.py --cuda=$CUDA_NUM --pro=1 --dir=./demo/live/ --overwrite --weight_dir=../saved_model
python inference.py --cuda=$CUDA_NUM --pro=1 --img=./demo/1.png --overwrite --weight_dir=../saved_model
# python inference.py --cuda=$CUDA_NUM --pro=1 --img=./demo/1.png --overwrite