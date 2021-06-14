import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SemanticNet(nn.Module): 
    def __init__(self):
        super(SemanticNet, self).__init__()
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

def homo_warping(src_fea, src_proj, ref_proj, depth_maps):
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