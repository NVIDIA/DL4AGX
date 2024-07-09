import math
import torch
import torch.nn as nn
from packnet_sfm.networks.DEST.DEST_EncDec import DEST_Pose, SimpleTR_B0, SimpleTR_B1, SimpleTR_B2, SimpleTR_B3, SimpleTR_B4, SimpleTR_B5


class InvDepth(nn.Module):
    """Inverse depth layer"""
    def __init__(self, in_channels, out_channels=1, min_depth=0.5):
        """
        Initializes an InvDepth object.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        min_depth : float
            Minimum depth value to calculate
        """
        super().__init__()
        self.min_depth = min_depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.pad = nn.ConstantPad2d([1] * 4, value=0)
        self.activ = nn.Sigmoid()
        self.scale = torch.tensor([self.min_depth])
    def forward(self, x):
        """Runs the InvDepth layer."""
        x = self.conv1(self.pad(x))
        return self.activ(x) / self.min_depth


class DESTNet(nn.Module):
    def __init__(self, model='B3', nb_ref_imgs=2, img_size=(192, 640)):
        """
        Defines the size of DEST model

        Parameters
        ----------
        model : string
            The size of DEST can be selected: 'B0' | 'B1' | 'B2' | 'B3' | 'B4' | 'B5' 
        nb_ref_imgs : int
            Number of reference images for Pose-Net
        img_size : tuple
            Input image size (H, W)
        """
        super().__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.connectivity = True

        if model == 'B0':
            self.num_out_ch, self.dest = SimpleTR_B0(img_size=img_size)
        elif model == 'B1':
            self.num_out_ch, self.dest = SimpleTR_B1(img_size=img_size)
        elif model == 'B2':
            self.num_out_ch, self.dest = SimpleTR_B2(img_size=img_size)
        elif model == 'B3':
            self.num_out_ch, self.dest = SimpleTR_B3(img_size=img_size)
        elif model == 'B4':
            self.num_out_ch, self.dest = SimpleTR_B4(img_size=img_size)
        elif model == 'B5':
            self.num_out_ch, self.dest = SimpleTR_B5(img_size=img_size)
        
        self.disp1_layer = InvDepth(self.dest.dims[-4])
        self.disp2_layer = InvDepth(self.dest.dims[-4])
        self.disp3_layer = InvDepth(self.dest.dims[-3])
        self.disp4_layer = InvDepth(self.dest.dims[-2])
        
        num_out_ch, self.dest_pose = DEST_Pose(dims=self.dest.dest_encoder.embed_dims, channels=16,
                                               num_layers=self.dest.dest_encoder.depths,
                                               reduction_ratio=self.dest.dest_encoder.sr_ratios,
                                               connectivity=self.connectivity)
                                               
        self.pose_pred = nn.Sequential(nn.Conv2d(self.dest.dest_encoder.embed_dims[3], 6 * self.nb_ref_imgs, kernel_size=1, padding=0))
        self.channel_reduction_pose = nn.Sequential(nn.Conv2d(9, 16, kernel_size=3, padding=0),
                                                        nn.BatchNorm2d(16),
                                                        nn.Tanh())

    def measure_Complexity(self, input_size=(3, 192, 640), mode='Depth'):
        input_shape = input_size

        if mode == 'Depth':
            model = Dummy_net_depth(self.dest, self.disp1_layer).eval()
            macs, params = get_model_complexity_info(model.cpu(),
                                                     input_shape, as_strings=True,
                                                     print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('%sNet Computational complexity: ' % mode, macs))
        print('{:<30}  {:<8}'.format('%sNet Number of parameters: ' % mode, params))


    def forward(self, rgb):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        out, _, self.ref_feat = self.dest(rgb)

        x = self.disp1_layer(out[0])
        if self.training:
            x2 = self.disp2_layer(out[1])
            x3 = self.disp3_layer(out[2])
            x4 = self.disp4_layer(out[3])
        
        if self.training:
            return {'inv_depths': [x, x2, x3, x4]}
        else:
            return {'inv_depths': x }

    def pose(self, image, context):
        assert (len(context) == self.nb_ref_imgs)
        input_ = [image]
        input_.extend(context)
        input_ = torch.cat(input_, 1)

        return self._poseNet(input_)

    def _poseNet(self, x_src):
        x_src = self.channel_reduction_pose(x_src)

        if self.connectivity:
            x = self.dest_pose(self.ref_feat, x_src)
        else :
            x = self.dest_pose(x_src)

        pose = self.pose_pred(x)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)
        return pose


    def reshape(self, ref_featss, b_):
        for i, e_dim in enumerate(self.dest.dest_encoder.embed_dims):
            ref_featss[i] = ref_featss[i].reshape(b_, -1, e_dim).repeat((1+self.nb_ref_imgs), 1, 1)
        return ref_featss



