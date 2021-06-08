0. prepare image: 
	e.g. /home/hadoop/data/data/scx/Zone19/

1. colmap reconstruction:
	e.g. /home/hadoop/data/data/scx/col_Zone19/

2. model_converter: bin -> txt :
	 ./colmap model_converter --input_path /home/hadoop/data/data/scx/ori_col_zone19/0 --output_path /home/hadoop/data/data/scx/ori_col_zone19/0/convert --output_type TXT

3. colmap2mvsnet:python /home/hadoop/scx/mvsnet/mvsnet/D2HC-RMVSNet-master/colmap2mvsnet.py --dense_folder /home/hadoop/data/data/scx/ori_col_zone19/0/mvsnet_data/


4. eval:
	python eval.py --dataset=data_eval_transform --model=drmvsnet --syncbn=False --batch_size=1 --inverse_cost_volume --inverse_depth=False --origin_size=False --gn=True --refine=False --save_depth=True --fusion=True --ngpu=1 --fea_net=FeatNet --cost_net=UNetConvLSTM --numdepth=512 --interval_scale=0.4 --max_h=360 --max_w=480 --image_scale=1.0 --pyramid=0 --testpath=/home/hadoop/data/data/scx/ori_col_zone19/0/ --testlist=lists/dtu/test.txt --loadckpt=/home/hadoop/scx/mvsnet/mvsnet/D2HC-RMVSNet-master/checkpoints_backup/model_blended.ckptmodel_blended.ckpt --outdir=/home/hadoop/data/data/scx/ori_col_zone19/0/outputs_Zone19

5. fusion:
	python fusion.py --testpath=/home/hadoop/data/data/scx/ori_col_zone19/0/ --testlist=lists/dtu/test.txt --outdir=/home/hadoop/data/data/scx/ori_col_zone19/0/outputs_Zone19/checkpoints_backup_model_blended.ckptmodel_blended.ckpt/ --test_dataset=dtu


6. For training: ./train.sh



