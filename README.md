0. prepare image: 
	e.g. /home/hadoop/data/data/scx/Zone19/

1. colmap reconstruction:
	e.g. /home/hadoop/data/data/scx/col_Zone19/

2. model_converter: bin -> txt :
	 ./colmap model_converter --input_path /home/hadoop/data/data/scx/ori_col_zone19/0/sparse --output_path /home/hadoop/data/data/scx/ori_col_zone19/0/convert --output_type TXT

3. colmap2mvsnet:python /home/hadoop/scx/mvsnet/mvsnet/D2HC-RMVSNet-master/colmap2mvsnet.py --dense_folder /home/hadoop/data/data/scx/ori_col_zone19/0/mvsnet_data/


4. eval:
python eval.py --dataset=data_eval_transform --max_h=360 --max_w=480 --image_scale=1.0 --testpath=/home/hadoop/data/data/0529/Colmap_In_0/ --testlist=lists/dtu/test.txt --batch_size=1 --interval_scale=0.4 --numdepth=512 --pyramid=0 --loadckpt=/home/hadoop/scx/mvsnet/mvsnet/UMT-MVSNet/checkpoints_backup/model_blended.ckpt --outdir=/home/hadoop/data/data/0529/Colmap_In_0/dense/refine/


6. For training: ./train.sh

python eval.py --dataset=data_eval_transform --max_h=360 --max_w=480 --image_scale=1.0 --testpath=//home/hadoop/data/data/scx/ori_col_zone19/0/mvsnet_data --testlist=lists/dtu/test.txt --batch_size=1 --interval_scale=0.4 --numdepth=512 --pyramid=0 --loadckpt=/home/hadoop/scx/mvsnet/mvsnet/UMT-MVSNet/checkpoints_backup/model_blended.ckpt --outdir=/home/hadoop/data/data/scx/ori_col_zone19/0/mvsnet_data/refine/

