import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from nnunetv2.training.nnUNetTrainer.blocks import *
import os
import sys
from torch.cuda.amp import autocast as autocast
class MedNeXt(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='3d',  # 2d or 3d
                 grn=False
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[0])]
                                         )
        ###############
        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=np.array([1,3,3]),
            stride=(1,2,2),
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[1])]
                                         )
        ###############
        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=np.array([1,3,3]),
            stride=(1,2,2),
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[3])]
                                         )
        ############### n_channels * 16
        if n_channels<32:
          lowest_channels=n_channels*16
        else:
          lowest_channels = 320
        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=lowest_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        ############### n_channels * 16
        self.enc_block_4 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=lowest_channels,
                out_channels=lowest_channels,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[4])]
                                         )
        ############### n_channels * 16
        self.down_4 = MedNeXtDownBlock(
            in_channels=lowest_channels,
            out_channels=lowest_channels,
            exp_r=exp_r[4],
            kernel_size=np.array([3, 3, 3]),
            stride=(1, 2, 2),
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_5 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=lowest_channels,
                out_channels=lowest_channels,
                exp_r=exp_r[4],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                         )
        ############### n_channels * 16
        self.down_5 = MedNeXtDownBlock(
            in_channels=lowest_channels,
            out_channels=lowest_channels,
            exp_r=exp_r[4],
            kernel_size=np.array([3, 3, 3]),
            stride=(1, 2, 2),
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=lowest_channels,
                out_channels=lowest_channels,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                        )

        ############### n_channels * 16
        self.up_5 = MedNeXtUpBlock(
            in_channels=lowest_channels,
            out_channels=lowest_channels,
            exp_r=exp_r[5],
            kernel_size=np.array([3, 3, 3]),
            stride=(1, 2, 2),
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        ############### n_channels * 16
        self.dec_block_5 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=lowest_channels,
                out_channels=lowest_channels,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                         )

        ############### n_channels * 16
        self.up_4 = MedNeXtUpBlock(
            in_channels=lowest_channels,
            out_channels=lowest_channels,
            exp_r=exp_r[5],
            kernel_size=np.array([3, 3, 3]),
            stride=(1, 2, 2),
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        ############### n_channels * 16
        self.dec_block_4 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=lowest_channels,
                out_channels=lowest_channels,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                         )

        self.up_3 = MedNeXtUpBlock(
            in_channels=lowest_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[6])]
                                         )
        ###############
        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=np.array([1,3,3]),
            stride=(1,2,2),
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[7])]
                                         )
        ###############
        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=np.array([1,3,3]),
            stride=(1,2,2),
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[8])]
                                         )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=lowest_channels, n_classes=n_classes, dim=dim)
            self.out_5 = OutBlock(in_channels=lowest_channels, n_classes=n_classes, dim=dim)
            self.out_6 = OutBlock(in_channels=lowest_channels, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x
    #@autocast()
    def forward(self, x):
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)
            x_res_4 = self.iterative_checkpoint(self.enc_block_4, x)
            x = checkpoint.checkpoint(self.down_4, x_res_4, self.dummy_tensor)
            x_res_5 = self.iterative_checkpoint(self.enc_block_5, x)
            x = checkpoint.checkpoint(self.down_5, x_res_5, self.dummy_tensor)
            
            x = self.iterative_checkpoint(self.bottleneck, x)
            if torch.any(torch.isnan(x)):
              print('sth. NAN in down block')
            if self.do_ds:
                x_ds_6 = checkpoint.checkpoint(self.out_6, x, self.dummy_tensor)

            x_up_5 = checkpoint.checkpoint(self.up_5, x, self.dummy_tensor)
            if torch.any(torch.isnan(x_up_5)):
              print('sth. NAN in x_up_5')
            dec_x = x_res_5 + x_up_5
            x = self.iterative_checkpoint(self.dec_block_5, dec_x)
            if self.do_ds:
                x_ds_5 = checkpoint.checkpoint(self.out_5, x, self.dummy_tensor)

            if torch.any(torch.isnan(x)):
              print('sth. NAN in up block5')
              
            x_up_4 = checkpoint.checkpoint(self.up_4, x, self.dummy_tensor)
            if torch.any(torch.isnan(x_up_4)):
              print('sth. NAN in x_up_4')
            dec_x = x_res_4 + x_up_4
            x = self.iterative_checkpoint(self.dec_block_4, dec_x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            if torch.any(torch.isnan(x)):
              print('sth. NAN in up block4')
              
            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            if torch.any(torch.isnan(x_up_3)):
              print('sth. NAN in x_up_3')
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)

            if torch.any(torch.isnan(x)):
              print('sth. NAN in up block3')
              
            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            if torch.any(torch.isnan(x_up_2)):
              print('sth. NAN in x_up_2')
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)

            if torch.any(torch.isnan(x)):
              print('sth. NAN in up block2')
              
            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            if torch.any(torch.isnan(x_up_1)):
              print('sth. NAN in x_up_1')
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)

            if torch.any(torch.isnan(x)):
              print('sth. NAN in up block1')
              
            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            if torch.any(torch.isnan(x_up_0)):
              print('sth. NAN in x_up_0')
            dec_x = x_res_0 + x_up_0
            if torch.any(torch.isnan(dec_x)):
              print('sth. NAN in up add')
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)

            if torch.any(torch.isnan(x)): 
              print('sth. NAN in up block0')
            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)
            if torch.any(torch.isnan(x)):
              print('sth. NAN in Final block')
              
        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)
            x_res_4 = self.enc_block_4(x)
            x = self.down_4(x_res_4)
            x_res_5 = self.enc_block_5(x)
            x = self.down_5(x_res_5)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_6 = self.out_6(x)


            x_up_5 = self.up_5(x)
            dec_x = x_res_5 + x_up_5
            x = self.dec_block_5(dec_x)
            if self.do_ds:
                x_ds_5 = self.out_5(x)

            x_up_4 = self.up_4(x)
            dec_x = x_res_4 + x_up_4
            x = self.dec_block_4(dec_x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)
            if self.do_ds:
                x_ds_3 = self.out_3(x)

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4, x_ds_5, x_ds_6]
        else:
            return x


if __name__ == "__main__":

    network = MedNeXt(
        in_channels=1,
        n_channels=32,
        n_classes=2,
        # exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
        exp_r = 2,
        kernel_size=3,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=True,
        block_counts = [2,2,2,2,2,2,2,2,2],
        # block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style=None,
        dim='3d',
        grn=True

    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

