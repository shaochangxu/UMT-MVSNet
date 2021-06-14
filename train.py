import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
import ast
from datasets.data_io import *

from third_party.sync_batchnorm import patch_replication_callback
from third_party.sync_batchnorm import convert_model
from third_party.radam import RAdam

cudnn.benchmark = True
#torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='A Official PyTorch Codebase of PVA-MVSNet')
parser.add_argument('--mode', default='train', help='train, val or test', choices=['train', 'test', 'val', 'evaluate', 'profile'])
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--loss', default='mvsnet_loss', help='select loss', choices=['mvsnet_loss', 'mvsnet_loss_l1norm', 
                            'mvsnet_loss_divby_interval', 'mvsnet_cls_loss', 'mvsnet_cls_loss_ori', 'unsup_loss'])

parser.add_argument('--refine', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--dp_ratio', type=float, default=0.0, help='learning rate')

parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

##### Distributed Sync BN
parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')

##### for dsrmvsnet
parser.add_argument('--reg_loss', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--max_h', type=int, default=512, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width when training.')
##### end dsrmvsnet

parser.add_argument('--local_rank', type=int, default=0, help='training view num setting')

parser.add_argument('--view_num', type=int, default=3, help='training view num setting')

parser.add_argument('--image_scale', type=float, default=0.25, help='pred depth map scale') # 0.5

parser.add_argument('--ngpu', type=int, default=4, help='gpu size')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--vallist', help='val list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--loss_w', type=int, default=4, help='number of epochs to train')

parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')

parser.add_argument('--lr_scheduler', default='multistep', help='lr_scheduler')
parser.add_argument('--optimizer', default='Adam', help='optimizer')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values') # 1.01

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')

parser.add_argument('--logdir', default='./logdir', help='the directory to save checkpoints/logs')
parser.add_argument('--save_dir', default=None, help='the directory to save checkpoints/logs')

# parse arguments and check
args = parser.parse_args()

if args.testpath is None:
    args.testpath = args.trainpath

set_random_seed(1)
device = torch.device(args.device)

#using sync_bn by using nvidia-apex, need to install apex. 半精度运算库
if args.sync_bn:
    assert args.using_apex, "must set using apex and install nvidia-apex"
if args.using_apex:
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

is_distributed = args.ngpu > 1

if is_distributed:
    print('start distributed ************\n')
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

if (not is_distributed) or (dist.get_rank() == 0):
    # create logger for mode "train" and "testall"
    if args.mode == "train":
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir)

        current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        print("current time", current_time_str)

        print("creating new summary file")
        logger = SummaryWriter(args.logdir)

    print("argv:", sys.argv[1:])
    print_args(args)

# model, optimizer
model = DrMVSNet(refine=args.refine, dp_ratio=args.dp_ratio, image_scale=args.image_scale, max_h=args.max_h, max_w=args.max_w, reg_loss=args.reg_loss)

model.to(device)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
print('Model define:')
print(model)
print('**********************\n')
if args.sync_bn:
    import apex
    print("using apex synced BN")
    model = apex.parallel.convert_syncbn_model(model)

##### LOSS
loss_dict = {'mvsnet_loss':mvsnet_loss, 'mvsnet_cls_loss': mvsnet_cls_loss, 'unsup_loss': unsup_loss}
try:
    model_loss = loss_dict[args.loss]
except KeyError:
    raise ValueError('invalid loss func key')

##### OPTIMIZER
if args.optimizer == 'Adam':
    print('optimizer: Adam \n')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
elif args.optimizer == 'RAdam':
    print('optimizer: RAdam !!!! \n')
    optimizer = RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

# load parameters
start_epoch = 0
if args.loadckpt:
    # load checkpoint file specified by args.loadckpt when eval
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)

if args.using_apex:
    # Initialize Amp
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level="O0",
                                      keep_batchnorm_fp32=None,
                                      loss_scale=None
                                      )

#conver model to dist
if is_distributed:
    print("Dist Train, Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )
else:
    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

# dataset, dataloader
# args.origin_size only load origin size depth, not modify Camera.txt
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.view_num, args.numdepth, args.interval_scale, args.inverse_depth, -1, args.image_scale, have_depth=(args.loss != 'unsup_loss')) # Training with False, Test with inverse_depth

val_dataset = MVSDataset(args.trainpath, args.vallist, "val", 5, args.numdepth, args.interval_scale, args.inverse_depth, 3, args.image_scale, reverse=False, both=False) #view_num = 5, light_idx = 3
test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, 1.06, args.inverse_depth, 3, args.image_scale, reverse=False, both=False)
reverse_test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, 1.06, args.inverse_depth, 3, args.image_scale, reverse=True, both=False)

if is_distributed:
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                        rank=dist.get_rank())
    TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=8,
                                    drop_last=True,
                                    pin_memory=True)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=4, drop_last=False,
                                    pin_memory=True)  
    ResTestImgLoader = DataLoader(reverse_test_dataset, args.batch_size, sampler=test_sampler, num_workers=4, drop_last=False,
                                    pin_memory=True)                                                       
else:
    TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=12, drop_last=True)
    ValImgLoader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    ResTestImgLoader = DataLoader(reverse_test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)


# main function
def train():
    print('run train()')
    if args.lr_scheduler == 'multistep':
        print('lr scheduler: multistep')
        milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
        lr_gamma = 1 / float(args.lrepochs.split(':')[1])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                            last_epoch=start_epoch - 1)
        ## get intermediate learning rate
        for _ in range(start_epoch):
            lr_scheduler.step()
    elif args.lr_scheduler == 'cosinedecay':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=4e-05)
        ## get intermediate learning rate
        for _ in range(start_epoch):
            lr_scheduler.step()
    elif args.lr_scheduler == 'warmupmultisteplr':
        milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
        lr_gamma = 1 / float(args.lrepochs.split(':')[1])
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                            last_epoch=len(TrainImgLoader) * start_epoch - 1)
    
    for epoch_idx in range(start_epoch, args.epochs):
        
        print('Epoch {}/{}:'.format(epoch_idx, args.epochs))

        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx
        print('Start Training')
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % 20 == 0

            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    logger.add_scalar('train/lr', lr, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                del scalar_outputs, image_outputs
                print(
                    'Epoch {}/{}, Iter {}/{}, LR {}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                        len(TrainImgLoader), lr, loss,
                                                                                        time.time() - start_time))

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % 1 == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.save_dir, epoch_idx),
                    _use_new_zipfile_serialization=False)
        gc.collect()

        # on test dataset
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % 20 == 0

            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
 
            if loss == 0:
                print('Loss is zero, no valid point')
                continue
            
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    save_images(logger, 'test', image_outputs, global_step)
                    print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(
                                    epoch_idx, args.epochs, batch_idx,
                                    len(TestImgLoader), loss,
                                    time.time() - start_time))

                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                
        if (not is_distributed) or (dist.get_rank() == 0):
            save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
            print("avg_test_scalars:", avg_test_scalars.mean())
        gc.collect()

        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(ResTestImgLoader):
            start_time = time.time()
            global_step = len(ResTestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % 20 == 0
            
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            
            if loss == 0:
                print('Loss is zero, no valid point')
                continue
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'test_reverser', scalar_outputs, global_step)
                    save_images(logger, 'test_reverse', image_outputs, global_step)
                    print('Epoch {}/{}, Iter {}/{}, reverse test loss = {:.3f}, time = {:3f}'.format(
                                    epoch_idx, args.epochs, batch_idx,
                                    len(ResTestImgLoader), loss,
                                    time.time() - start_time))

                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                
        if (not is_distributed) or (dist.get_rank() == 0):
            save_scalars(logger, 'fulltest_reverse', avg_test_scalars.mean(), global_step)
            print("avg_test_scalars_reverse:", avg_test_scalars.mean())
        gc.collect()
        

def forward_hook(module, input, output):
        print(module)
        print('input', input)
        print('output', output)


def val():
    global save_dir
    print('Phase: test \n')


    avg_test_scalars = DictAverageMeter()
    if args.mode == 'test':
        ImgLoader = TestImgLoader
    elif args.mode == 'val':
        ImgLoader = ValImgLoader
        
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(ImgLoader):
        start_time = time.time()
        
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
       
        if loss == 0:
            print('Loss is zero, no valid point')
            continue

        avg_test_scalars.update(scalar_outputs)

        if (not is_distributed) or (dist.get_rank() == 0):      
            print('Iter {}/{}, val loss = {:.3f}, time = {:3f}'.format(batch_idx, len(ImgLoader), loss,
                                                                    time.time() - start_time))
            del scalar_outputs, image_outputs

            if batch_idx % 100 == 0:
                print("Iter {}/{}, val results = {}".format(batch_idx, len(ImgLoader), avg_test_scalars.mean()))

    if (not is_distributed) or (dist.get_rank() == 0):
        print("avg_{}_scalars:".format(args.mode), avg_test_scalars.mean())

def train_sample(sample, detailed_summary=False, refine=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    depth_value = sample_cuda["depth_values"]
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

    if args.loss == 'unsup_loss':
        depth_est = outputs["depth"]
        semantic_mask = outputs["semantic_mask"]
        loss = model_loss(sample_cuda["imgs"], sample_cuda["proj_matrices"], depth_est, semantic_mask)
    else:
        depth_gt = sample_cuda["depth"]
        depth_est = outputs["depth"]
        semantic_mask = outputs["semantic_mask"]
        loss = model_loss(sample_cuda["imgs"], depth_est, depth_gt, mask, semantic_mask)

    if is_distributed and args.using_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    # gradient clip
    #torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
    optimizer.step()
    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask,  
                     "ref_img": sample["imgs"][:, 0],
                    }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True, refine=False):
    model.eval()
    sample_cuda = tocuda(sample)
    
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    depth_value = sample_cuda["depth_values"]
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    #print(depth_value.type(), depth_interval.type(), depth_gt.type())
    
    if args.loss == 'unsup_loss':
        depth_est = outputs["depth"]
        semantic_mask = outputs["semantic_mask"]
        photometric_confidence = outputs['photometric_confidence']
        loss = model_loss(sample_cuda["imgs"], sample_cuda["proj_matrices"], depth_est, semantic_mask)
    else:
        depth_gt = sample_cuda["depth"]
        depth_est = outputs["depth"]
        photometric_confidence = outputs['photometric_confidence']
        semantic_mask = outputs["semantic_mask"]
        loss = model_loss(sample_cuda["imgs"], depth_est, depth_gt, mask, semantic_mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask,
                     "photometric_confidence": photometric_confidence * mask, 
                     "ref_img": sample["imgs"][:, 0]}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test" or args.mode == "val":
        val()
