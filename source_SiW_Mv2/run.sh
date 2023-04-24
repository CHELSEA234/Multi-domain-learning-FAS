source ~/.bashrc
conda activate anti_spoofing
CUDA_NUM=0
python train.py --cuda=$CUDA_NUM --pro=1
python test.py --cuda=$CUDA_NUM --pro=1
python train.py --cuda=$CUDA_NUM --pro=2 --unknown=Co
python test.py --cuda=$CUDA_NUM --pro=2 --unknown=Co
python train.py --cuda=$CUDA_NUM --pro=2 --unknown=Eye
python test.py --cuda=$CUDA_NUM --pro=2 --unknown=Eye
python train.py --cuda=$CUDA_NUM --pro=2 --unknown=Funnyeye
python test.py --cuda=$CUDA_NUM --pro=2 --unknown=Funnyeye
python train.py --cuda=$CUDA_NUM --pro=2 --unknown=Half
python test.py --cuda=$CUDA_NUM --pro=2 --unknown=Half
python train.py --cuda=$CUDA_NUM --pro=2 --unknown=Im
python test.py --cuda=$CUDA_NUM --pro=2 --unknown=Im
python train.py --cuda=$CUDA_NUM --pro=2 --unknown=Mann
python test.py --cuda=$CUDA_NUM --pro=2 --unknown=Mann
python train.py --cuda=$CUDA_NUM --pro=2 --unknown=Mouth
python test.py --cuda=$CUDA_NUM --pro=2 --unknown=Mouth
python train.py --cuda=$CUDA_NUM --pro=3
python test.py --cuda=$CUDA_NUM --pro=3