# UMT-MVSNet

## Using
1. prepare image: e.g. /home/hadoop/data/data/scx/Zone19
2. colmap reconstrucion: e.g. output in /home/data/data/scx/col_Zone19
3. colmap2mvsnet:
 ```
 python colmap2mvsnet.py --dense_folder /home/hadoop/data/data/col_Zone19/dense
 ```
 you can use whitelist to define a part of images to reconstruct,just as
  ```
 python colmap2mvsnet.py --dense_folder /home/hadoop/data/data/col_Zone19/dense --whitlist /home/data/data/scx/col_Zone19/dense/stereo/fusion.cfg
 ```
4. eval: 
```
python eval.py --dateset=data_eval_transform --max_h=360 --max_w=480 --image_scale=1.0 --test_path=/home/hadoop/data/data/col_Zone19/dense \
              --testlist=lists/dtu/test.txt --batch_size=1 --interval_scale=0.4 --numdepth=512 --pyramid=0 --loadckpt=./checkpoints/model_blended.ckpt \
              --outdir=/home/hadoop/data/data/col_Zone19/dense/refine
```

## Train:
1. prepare your training data: e.g. dtu in  /home/hadoop/scx/trainingdata/dtu_training.  
we also support unsupervised training just ignore this step，but to implement the your own data sample py in datasets dir, just as dtu_yao.py.
2. ./train.sh
