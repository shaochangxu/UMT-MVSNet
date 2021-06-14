import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import sys
from copy import deepcopy
import torchvision
import math

from .submodule import volumegatelight, volumegatelightgn
# Recurrent Multi-scale Module
from .rnnmodule import *

class DrMVSNet(nn.Module):
    def __init__(self, refine=True, dp_ratio=0.0, image_scale=0.25, max_h=960, max_w=480,
                 reg_loss=False, return_depth=False, gn=True, pyramid=-1, predict=False):
        super(DrMVSNet, self).__init__() # parent init
        
        self.gn = gn
        self.feature = FeatNet(gn=self.gn)

        ## 4 LSTM layers
        if pyramid == -1:
            input_size = (int(max_h*image_scale), int(max_w*image_scale)) #height, width
        elif pyramid == 0:
            input_size = (296, 400)
        elif pyramid == 1:
            input_size = (144, 200)
        elif pyramid == 2:
            input_size = (72, 96)   
        #print('input UNetConvLSTM H,W: {}, {}'.format(input_size[0], input_size[1]))
        input_dim = [32, 16, 16, 32, 32]
        hidden_dim = [ 16, 16, 16, 16, 8]
        num_layers = 5
        kernel_size = [(3, 3) for i in range(num_layers)]
        
        self.cost_regularization = UNetConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers,
             batch_first=False, bias=True, return_all_layers=False, gn=self.gn)
    
        # Cost Aggregation
        self.gatenet = gatenet(self.gn, 32)

        self.reg_loss = reg_loss
        self.return_depth = return_depth
        self.predict = predict
        self.mask_net = SemanticNet()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # process DrMVSNet
        if not self.predict:
            semantic_mask = self.mask_net(imgs[0])

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        
        # Recurrent process i-th depth layer
        # initialization for drmvsnet # recurrent module
        cost_reg_list = []
        hidden_state = None
        if not self.return_depth: # Training Phase;
            for d in range(num_depth):
                # step 2. differentiable homograph, build cost volume

                ref_volume = ref_feature
                warped_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                        warped_volume = homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_values[:, d])
                        warped_volume = (warped_volume - ref_volume).pow_(2)
                        reweight = self.gatenet(warped_volume) 
                        if warped_volumes is None:
                            warped_volumes = (reweight + 1) * warped_volume
                        else:
                            warped_volumes = warped_volumes + (reweight + 1) * warped_volume
                volume_variance = warped_volumes / len(src_features)
                
                # step 3. cost volume regularization
                cost_reg, hidden_state= self.cost_regularization(-1 * volume_variance, hidden_state, d)
                cost_reg_list.append(cost_reg)
            
            
            prob_volume = torch.stack(cost_reg_list, dim=1).squeeze(2)
            prob_volume = F.softmax(prob_volume, dim=1) # get prob volume use for recurrent to decrease memory consumption

            if not self.reg_loss:
                depth = depth_regression(prob_volume, depth_values=depth_values)
                if self.predict:
                    with torch.no_grad():
                        # photometric confidence
                        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                        depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                        photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
                    semantic_mask = photometric_confidence
                # print(self.reg_loss)
                return {"depth": depth, 'prob_volume': prob_volume, "semantic_mask":semantic_mask, "photometric_confidence": photometric_confidence}
            else:
                depth = depth_regression(prob_volume, depth_values=depth_values)
                # print(self.reg_loss)
                with torch.no_grad():
                    # photometric confidence
                    prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                    depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                    photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
                
                if self.predict:
                    semantic_mask = photometric_confidence
                #return {'prob_volume': prob_volume, "depth": depth, "photometric_confidence": photometric_confidence, "semantic_mask":semantic_mask}
                return {"depth": depth, "photometric_confidence": photometric_confidence, "semantic_mask":semantic_mask}
        else:
            shape = ref_feature.shape
            depth_image = torch.zeros(shape[0], shape[2], shape[3]).cuda() #B X H X w
            max_prob_image = torch.zeros(shape[0], shape[2], shape[3]).cuda()
            exp_sum = torch.zeros(shape[0], shape[2], shape[3]).cuda()

            for d in range(num_depth):
                # step 2. differentiable homograph, build cost volume
               

                ref_volume = ref_feature
                warped_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                        warped_volume = homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_values[:, d])
                        warped_volume = (warped_volume - ref_volume).pow_(2)
                        reweight = self.gatenet(warped_volume) # saliency 
                        if warped_volumes is None:
                            warped_volumes = (reweight + 1) * warped_volume
                        else:
                            warped_volumes = warped_volumes + (reweight + 1) * warped_volume
                volume_variance = warped_volumes / len(src_features)
                
                # step 3. cost volume regularization
                cost_reg, hidden_state= self.cost_regularization(-1 * volume_variance, hidden_state, d)

                # Start to caculate depth index
                #print('cost_reg: ', cost_reg.shape())
                prob = torch.exp(cost_reg.squeeze(1))

                d_idx = d
                depth = depth_values[:, d] # B 
                temp_depth_image = depth.view(shape[0], 1, 1).repeat(1, shape[2], shape[3])
                update_flag_image = (max_prob_image < prob).type(torch.float)
                #print('update num: ', torch.sum(update_flag_image))
                new_max_prob_image = torch.mul(update_flag_image, prob) + torch.mul(1-update_flag_image, max_prob_image)
                new_depth_image = torch.mul(update_flag_image, temp_depth_image) + torch.mul(1-update_flag_image, depth_image)
                max_prob_image = new_max_prob_image
                depth_image = new_depth_image
                exp_sum = exp_sum + prob
            
            forward_exp_sum = exp_sum
            forward_depth_map = depth_image
            
            return {"depth": forward_depth_map, "photometric_confidence": max_prob_image / forward_exp_sum}


def get_propability_map(prob_volume, depth, depth_values):
    # depth_values: B,D
    shape = prob_volume.shape
    batch_size = shape[0]
    depth_num = shape[1]
    height = shape[2]
    width = shape[3]

    depth_delta = torch.abs(depth.unsqueeze(1).repeat(1, depth_num, 1, 1) - depth_values.repeat(height, width, 1, 1).permute(2, 3, 0, 1))
    _, index = torch.min(depth_delta, dim=1) # index: B, H, W
    index = index.unsqueeze(1).repeat(1, depth_num, 1, 1)
    index_left0 = index 
    index_left1 = torch.clamp(index - 1, 0, depth_num - 1)
    index_right0 = torch.clamp(index + 1, 0, depth_num - 1)
    index_right1 = torch.clamp(index + 2, 0, depth_num - 1)

    prob_map_left0 = torch.mean(torch.gather(prob_volume, 1, index_left0), dim=1)
    prob_map_left1 = torch.mean(torch.gather(prob_volume, 1, index_left1), dim=1)
    prob_map_right0 = torch.mean(torch.gather(prob_volume, 1, index_right0), dim=1)
    prob_map_right1 = torch.mean(torch.gather(prob_volume, 1, index_right1), dim=1)
    prob_map = torch.clamp(prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1, 0, 0.9999)
    return prob_map

def mvsnet_loss(imgs, depth_est, depth_gt, mask, semantic_mask):
    mask = mask > 0.5
    mask1 = semantic_mask[:, 0, :, :]
    step_mask = semantic_mask[:, 1, :, :]
    ref_features = torch.unbind(imgs, 1)[0]
    simplify_loss = simplifyDis(ref_features, mask1, step_mask)
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True) + simplify_loss

def mvsnet_cls_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False): 
    # depth_value: B * NUM
    # get depth mask
    mask_true = mask 
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape 

    depth_num = depth_value.shape[-1]
    depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
   
    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1) # round; B, H, W

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))# - mask_true # remove mask=0 pixel
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    
    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1).squeeze(1) # B, 1, H, W

    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map


def gradient(pred):
    #print(pred.shape)
    D_dy = pred[:, :, 1:, :] - pred[:,:,:-1,:]
    D_dx = pred[:, :, :, 1:] - pred[:,:,:,:-1]
    return D_dx, D_dy 

def gradient_image(img):
	#print(img.shape)
	w = img.shape[3]
	h = img.shape[2]

	r = F.pad(img, (0,1,0,0))[:,:,:,1:]
	l = F.pad(img, (1,0,0,0))[:,:,:,:w]
	t = F.pad(img, (0,0,1,0))[:,:,:h,:]
	b = F.pad(img, (0,0,0,1))[:,:,1:,:]
	dx = torch.abs(r - l)
	dy = torch.abs(b - t)
	dx[:,:,:,-1] = 0
	dy[:,:,-1,:] = 0
	return dx, dy	

def compute_reconstr_loss_map(warped,ref,mask):
	alpha = 0.5
	mask = mask.unsqueeze(1).repeat(1,3,1,1)

	photo_loss = F.smooth_l1_loss(warped * mask, ref * mask,  reduce='none')

	ref_dx, ref_dy = gradient_image(ref * mask)
	warpped_dx, warpped_dy = gradient_image(warped * mask)
	#print(ref_dx.shape)
	#grad_loss = F.smooth_l1_loss(warpped_dx, ref_dx,  size_average=True) + F.smooth_l1_loss(warpped_dy, ref_dy,  size_average=True)
	grad_loss = torch.abs(warpped_dx - ref_dx) + torch.abs(warpped_dy - ref_dy)
	#print(grad_loss)
	return (1 - alpha)*photo_loss + alpha*grad_loss

def ssim(x, y, mask):
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)
    sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
    
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    
    ssim = ssim_n / ssim_d
    mask = mask.unsqueeze(1).repeat(1,3,1,1)
    ssim_mask = F.avg_pool2d(mask, 3, 1)
    #print(ssim.shape)

    return ssim_mask * torch.clamp((1 - ssim) / 2, 0, 1)

def depth_smoothness(depth, img, lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    img_dx, img_dy = gradient(img)
    depth_dx, depth_dy = gradient(depth)

    weights_x = torch.exp(- (lambda_wt * torch.mean(torch.abs(img_dx), 3, True)))
    weights_y = torch.exp(- (lambda_wt * torch.mean(torch.abs(img_dy), 3, True)))
    
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y

    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

def calV(src_fea, step):
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    max_R = int(min(height, width) / 20)
    step = step * max_R

    step_max = torch.exp(-step * step / 0.003)
    #print(step_max.shape)
    step_max.view(batch, height * width)

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                           torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xy = torch.stack((x, y)).repeat(batch, 1, 1)  # [2, H*W]
    #print((max_R * 2 + 1) * (max_R * 2 + 1))

    warped_src_fea = src_fea.unsqueeze(2).repeat(1, 1, (max_R * 2 + 1) * (max_R * 2 + 1), 1, 1)#[B,C,N,H,W]
    W = step_max.unsqueeze(1).repeat(1, (max_R * 2 + 1) * (max_R * 2 + 1), 1, 1)#[B,N,H,W]
    step_max = step_max.unsqueeze(1).repeat(1, (max_R * 2 + 1) * (max_R * 2 + 1), 1, 1)
    i = 0
    for s_y in range(-max_R, max_R + 1):
        for s_x in range(-max_R, max_R + 1):
            trans_x = xy[:,0, :] + s_x
            trans_y = xy[:, 1, :] + s_y
            trans_x_normalized = trans_x / ((width - 1) / 2) - 1
            trans_y_normalized = trans_y / ((height - 1) / 2) - 1
            grid = torch.stack((trans_x_normalized, trans_y_normalized), dim=-1)  # [B, H*W, 2]
            warped_src_fea[:,:,i,:,:] = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear', padding_mode='zeros')
            W[:,i,:,:] = math.exp((-s_y*s_y - s_x * s_x) / 0.003)
            i = i + 1
        #print(i)
    #print(step_max.shape)
    #print(W.shape)
    W = torch.where(W > step_max, torch.zeros_like(W), W)
    W = F.normalize(W, p=2, dim=1)
    W = W.unsqueeze(1)
    WF = W * warped_src_fea
    SWF = torch.sum(WF, dim=2)
    V = src_fea - SWF

    return V

def simplifyDis(src_fea, mask, step):
    V_before = calV(src_fea, step)
    mask = mask.unsqueeze(1).repeat(1,3,1,1)
    V_after = calV(src_fea * mask, step)
    loss_fn = nn.MSELoss(reduce=True, size_average=True)
    return loss_fn(V_before, V_after)

def unsup_loss(imgs, proj_matrices, depth_est, semantic_mask):
	#print("unsup")
	#print(mask.shape)
	imgs = torch.unbind(imgs, 1)
	#print(len(imgs))
	proj_matrices = torch.unbind(proj_matrices, 1)
	assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
	img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
	#num_depth = depth_value.shape[-1]
	num_views = len(imgs)
	mask = semantic_mask[:, 0, :, :]
	step_mask = semantic_mask[:, 1, :, :]
	
	features = [img for img in imgs] #ref_num * 3
	ref_features, src_features = features[0], features[1:]
	
	ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
	batch, channels = ref_features.shape[0], ref_features.shape[1]

	b = True

	ssim_loss = 0
	for src_fea, src_proj in zip(src_features, src_projs):
		#print(src_fea.shape)
		warped_volume, vis_mask = homo_warping(src_fea, src_proj, ref_proj, depth_est) 

		recon_loss = compute_reconstr_loss_map(warped_volume, imgs[0].view(batch, 3, img_height, img_width), vis_mask)
		ssim_loss += torch.mean(ssim(warped_volume, imgs[0].view(batch, 3, img_height, img_width), vis_mask))
	
		recon_loss = torch.mean(recon_loss, 1, keepdim = True)
		#print(recon_loss.shape)
		vis_mask = 1 - vis_mask		
		res_map = recon_loss + 1e4 * vis_mask.unsqueeze(1)

		if(b):
			reproj_volume =  res_map
			b = False		
		else:																																																										
			reproj_volume = torch.cat((reproj_volume, res_map), 1) #[B, N, H, W]
	
	smooth_loss = depth_smoothness(depth_est.view(batch, 1, img_height, img_width), imgs[0])
		
	top_vals, _ = torch.topk(reproj_volume, 3, dim = 1, largest = False, sorted = False)

	top_masks = torch.where(top_vals < 1e4, torch.ones_like(top_vals), torch.zeros_like(top_vals))
	top_vals = top_vals * top_masks

	reconstr_loss = torch.mean(torch.mean(top_vals, 1))

	simplify_loss = simplifyDis(ref_features, mask, step_mask)

	loss = 12 * reconstr_loss + 6 * ssim_loss / (num_views - 1) + 0.18 * smooth_loss + simplify_loss

	return loss

            








