import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = nn.SyncBatchNorm(out_channels)
    def forward(self, x):
        return self.bn(self.conv(x))

class deConvBnReLU(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=1
            ):
        super(deConvBnReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, output_padding=output_padding, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            #nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class ConvGnReLU(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            group_channel=8 # channel number in each group
            ):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_channel=group_channel
        G = int(max(1, out_channels / self.group_channel))
        #print(G , out_channels)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)

class ConvGn(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            group_channel=8 # channel number in each group
            ):
        super(ConvGn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.group_channel=group_channel
        G = int(max(1, out_channels / self.group_channel))
        #print(G , out_channels)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return self.gn(self.conv(x))

class deConvGnReLU(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            output_padding=1,
            group_channel=8 # channel number in each group
            ):
        super(deConvGnReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, output_padding=output_padding, stride=stride, bias=bias)
        self.group_channel=group_channel
        G = int(max(1, out_channels / self.group_channel))
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)


class ConvGnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, group_channel=8):
        super(ConvGnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        #self.bn = nn.BatchNorm3d(out_channels)
        self.group_channel=group_channel
        G = int(max(1, out_channels / self.group_channel))
        #print(G , out_channels)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1

class UResVCG16(nn.Module): 
    def __init__(self):
        super(UResVCG16, self).__init__()
        base_filter = 8
        self.conv1_0 = ConvGnReLU(3              , base_filter * 2, stride=2)
        self.conv2_0 = ConvGnReLU(base_filter * 2, base_filter * 4, stride=2)
        self.conv3_0 = ConvGnReLU(base_filter * 4, base_filter * 8, stride=2)
        self.conv4_0 = ConvGnReLU(base_filter * 8, base_filter * 16, stride=2)

        self.conv0_1 = ConvGnReLU(3              , base_filter * 1)
        self.conv0_2 = ConvGnReLU(base_filter * 1, base_filter * 1)

        self.conv1_1 = ConvGnReLU(base_filter * 2, base_filter * 2)
        self.conv1_2 = ConvGnReLU(base_filter * 2, base_filter * 2)

        self.conv2_1 = ConvGnReLU(base_filter * 4, base_filter * 4)
        self.conv2_2 = ConvGnReLU(base_filter * 4, base_filter * 4)

        self.conv3_1 = ConvGnReLU(base_filter * 8, base_filter * 8)
        self.conv3_2 = ConvGnReLU(base_filter * 8, base_filter * 8)

        self.conv4_1 = ConvGnReLU(base_filter * 16, base_filter * 16)
        self.conv4_2 = ConvGnReLU(base_filter * 16, base_filter * 16)
        self.conv5_0 = deConvGnReLU(base_filter * 16, base_filter * 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv5_1 = ConvGnReLU(base_filter * 16, base_filter * 8)
        self.conv5_2 = ConvGnReLU(base_filter * 8, base_filter * 8)
        self.conv6_0 = deConvGnReLU(base_filter * 8, base_filter * 4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv6_1 = ConvGnReLU(base_filter * 8, base_filter * 4)
        self.conv6_2 = ConvGnReLU(base_filter * 4, base_filter * 4)
        self.conv7_0 = deConvGnReLU(base_filter * 4, base_filter * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv7_1 = ConvGnReLU(base_filter * 4, base_filter * 2)
        self.conv7_2 = ConvGnReLU(base_filter * 2, base_filter * 2)
        self.conv8_0 = deConvGnReLU(base_filter * 2, base_filter * 1, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.conv8_1 = ConvGnReLU(base_filter * 2, base_filter * 1)
        self.conv8_2 = ConvGnReLU(base_filter * 1, base_filter * 1) # end of UNet
        # self.conv9_0 = ConvGnReLU(base_filter * 1, base_filter * 2, 5, 2, 2)
        # self.conv9_1 = ConvGnReLU(base_filter * 2, base_filter * 2)
        # self.conv9_2 = ConvGnReLU(base_filter * 2, base_filter * 2)
        # self.conv10_0 = ConvGnReLU(base_filter * 2, base_filter * 4, 5, 2, 2)
        # self.conv10_1 = ConvGnReLU(base_filter * 4, base_filter * 4)
        # self.conv10_2 = nn.Conv2d(base_filter * 4, base_filter * 4, 1, bias=False)
        self.conv9_0 = nn.Conv2d(base_filter * 1, 2, kernel_size=1, stride=1)
        self.conv9_1 = nn.Conv2d(2, 2, kernel_size=1, stride=1)
        self.conv9_2 = nn.Conv2d(2, 2, kernel_size=1, stride=1)
        self.activate_fn = nn.Sigmoid()

    def forward(self, x):
        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)
        conv4_0 = self.conv4_0(conv3_0)

        conv0_2 = self.conv0_2(self.conv0_1(x))

        conv1_2 = self.conv1_2(self.conv1_1(conv1_0))

        conv2_2 = self.conv2_2(self.conv2_1(conv2_0))

        conv3_2 = self.conv3_2(self.conv3_1(conv3_0))

        conv4_2 = self.conv4_2(self.conv4_1(conv4_0))
        conv5_0 = self.conv5_0(conv4_2)

        x = torch.cat((conv5_0, conv3_2), dim=1)
        conv5_2 = self.conv5_2(self.conv5_1(x))
        conv6_0 = self.conv6_0(conv5_2)

        x = torch.cat((conv6_0, conv2_2), dim=1)
        conv6_2 = self.conv6_2(self.conv6_1(x))
        conv7_0 = self.conv7_0(conv6_2)
        
        x = torch.cat((conv7_0, conv1_2), dim=1)
        conv7_2 = self.conv7_2(self.conv7_1(x))
        conv8_0 = self.conv8_0(conv7_2)

        x = torch.cat((conv8_0, conv0_2), dim=1)
        conv8_2 = self.conv8_2(self.conv8_1(x))
        #print(conv8_2.shape)
        conv9_2 = self.conv9_2(self.conv9_1(self.conv9_0(conv8_2)))
        #conv10_2 = self.conv10_2(self.conv10_1(self.conv10_0(conv9_2)))
        out = self.activate_fn(conv9_2)
        return out


def homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_value):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_value: [B] # TODO: B, 1
    # out: [B, C, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.repeat(1, 1, 1) * depth_value.view(batch, 1, 1)
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz[:,2:3,:][proj_xyz[:, 2:3,:] == 0] += 0.0001 # WHY BUG
        proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
        
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, 1 * height, width, 2), mode='bilinear',
                                   padding_mode='zeros').type(torch.float32)
    #warped_src_fea = warped_src_fea.view(batch, channels, height, width) # B, C, H, W
    #warped_src_fea = warped_src_fea.type(torch.float32)
    return warped_src_fea

def homo_warping_depthwise_ori(src_fea, src_proj, ref_proj, depth_value):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_value: [B] # TODO: B, 1
    # out: [B, C, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, 1, 1) * depth_value.view(batch, 1, 1,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz[:,2:3,:,:][proj_xyz[:, 2:3, :, :] == 0] += 0.0001 # WHY BUG
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, 1 * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, height, width) # B, C, H, W
    warped_src_fea = warped_src_fea.type(torch.float32)
    return warped_src_fea


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] # TODO: B, 1
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz[:,2:3,:,:][proj_xyz[:, 2:3, :, :] == 0] += 0.0001 # WHY BUG
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    
    warped_src_fea = warped_src_fea.type(torch.float32)
    return warped_src_fea


# Without gradient for Testing to save some memory
def homo_warping2(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    warped_src_fea = src_fea.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        for i in range(num_depth):
            rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, 1, 1) * depth_values[:,i].view(batch, 1, 1, 1)  # [B, 3, 1, H*W]
            # print('rot_depth_xyz: ', np.shape(rot_depth_xyz))
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, 1, H*W]
            # print('proj_xyz: ', np.shape(proj_xyz))
            proj_xy = proj_xyz[:, :2, 0, :] / proj_xyz[:, 2:3, 0, :]  # [B, 2, H*W]
            # print('proj_xy: ', np.shape(proj_xy))
            proj_x_normalized = proj_xy[:, 0, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :] / ((height - 1) / 2) - 1
            grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=-1)  # [B, H*W, 2]

            warped_src_fea[:,:,i,:,:] = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear',
                                        padding_mode='zeros')
    #print('in homo_warping2 require_grad return', warped_src_fea.requires_grad)
    return warped_src_fea

def homo_warping3(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, 1]
    # out: [B, C, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    #warped_src_fea = torch.zeros(batch, channels, num_depth, height, width).cuda() # [B, C, Ndepth, H, W] 

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, 1, 1) * (depth_values.view(batch, 1, 1, 1))  # [B, 3, 1, H*W]
        # print('rot_depth_xyz: ', np.shape(rot_depth_xyz))
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, 1, H*W]
        # print('proj_xyz: ', np.shape(proj_xyz))
        proj_xy = proj_xyz[:, :2, 0, :] / proj_xyz[:, 2:3, 0, :]  # [B, 2, H*W]
        # print('proj_xy: ', np.shape(proj_xy))
        proj_x_normalized = proj_xy[:, 0, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :] / ((height - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=-1)  # [B, H*W, 2]

        warped_src_fea = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear',
                                    padding_mode='zeros')

    return warped_src_fea



def homo_warping4(src_fea, src_proj, ref_proj, depth_maps):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_maps: [B, H, W]
    # out: [B, C, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    #warped_src_fea = torch.zeros(batch, channels, num_depth, height, width).cuda() # [B, C, Ndepth, H, W] 

    #with torch.no_grad():
    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                           torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    #rot_xyz = xyz
    t = rot_xyz.unsqueeze(2).repeat(1, 1, 1, 1)
    #print(xyz)
    rot_depth_xyz =  t * (depth_maps.view(-1).view(batch, 1, 1, height * width))  # [B, 3, 1, H*W]
    #print(rot_depth_xyz)
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, 1, H*W]
    # print('proj_xyz: ', np.shape(proj_xyz))
    proj_xy = proj_xyz[:, :2, 0, :] / proj_xyz[:, 2:3, 0, :]  # [B, 2, H*W]
    #print("proj_xy")
    
    #print(px)
    px = proj_xy[:, 0, :]
    py = proj_xy[:, 1, :]
    zeros = torch.zeros_like(px)
    ones = torch.ones_like(px)
    torch.set_printoptions(profile="full")
    #print(px.view(batch, height, width))
    px_mask_high = torch.where(px > width - 1, zeros, ones).view(batch, height, width)
    px_mask_low = torch.where(px < 0, zeros, ones).view(batch, height, width)
    py_mask_high = torch.where(py > height - 1, zeros, ones).view(batch, height, width)
    py_mask_low = torch.where(py < 0, zeros, ones).view(batch, height, width)
    mask = px_mask_high * px_mask_low * py_mask_high * py_mask_low

    #print(mask)

    # print('proj_xy: ', np.shape(proj_xy))
    proj_x_normalized = proj_xy[:, 0, :] / ((width - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1, :] / ((height - 1) / 2) - 1
    grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=-1)  # [B, H*W, 2]


    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear',
                                padding_mode='zeros')

    return warped_src_fea, mask


# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth


if __name__ == "__main__":
    # some testing code, just IGNORE it
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    MVSDataset = find_dataset_def("dtu_yao")
    dataset = MVSDataset("/data1/Dataset/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 256)
    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4].cuda()
    proj_matrices = item["proj_matrices"].cuda()
    mask = item["mask"].cuda()
    depth = item["depth"].cuda()
    depth_values = item["depth_values"].cuda()

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warping2(src_imgs[0], src_projs[0], ref_proj, depth_values)
    
    cv2.imwrite('../tmp/ref.png', ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite('../tmp/tmp{}.png'.format(i), img_np[:, :, ::-1] * 255)


    # generate gt
    def tocpu(x):
        return x.detach().cpu().numpy().copy()


    ref_img = tocpu(ref_img)[0].transpose([1, 2, 0])
    src_imgs = [tocpu(x)[0].transpose([1, 2, 0]) for x in src_imgs]
    ref_proj_mat = tocpu(ref_proj)[0]
    src_proj_mats = [tocpu(x)[0] for x in src_projs]
    mask = tocpu(mask)[0]
    depth = tocpu(depth)[0]
    depth_values = tocpu(depth_values)[0]

    for i, D in enumerate(depth_values):
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # print("yy", yy.max(), yy.min())
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        X = np.vstack((xx, yy, np.ones_like(xx)))
        # D = depth.reshape([-1])
        # print("X", "D", X.shape, D.shape)

        X = np.vstack((X * D, np.ones_like(xx)))
        X = np.matmul(np.linalg.inv(ref_proj_mat), X)
        X = np.matmul(src_proj_mats[0], X)
        X /= X[2]
        X = X[:2]

        yy = X[0].reshape([height, width]).astype(np.float32)
        xx = X[1].reshape([height, width]).astype(np.float32)

        warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
        # warped[mask[:, :] < 0.5] = 0

        cv2.imwrite('../tmp/tmp{}_gt.png'.format(i), warped[:, :, ::-1] * 255)
