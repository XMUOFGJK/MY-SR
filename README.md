# MY-SR


# # #my train
# CUDA_VISIBLE_DEVICES=6,7 python main.py --n_GPUs 2 --model resDBPN_2 --scale 2 --save resDBPN2_v5_x2_bs16_120_ps112_k6_res3_L1_RDB_SE --ext sep --base_filter 120 --n_resblock 3 \
# --lr 1e-4 --batch_size 16 --patch_size 112 --epochs 1000 --loss 1*L1 --print_every 20 --test_every 300 --data_test Set5 --save_results #--reset --weight_decay 1e-4 --chop


## small_model #my train
# CUDA_VISIBLE_DEVICES=0,1 python main.py --n_GPUs 2 --model resDBPN_2 --scale 2 --save resDBPN2_small_x2_bs24_64_ps128_k6_res3_L1_SE_stage4_dilated_2 --ext sep --num_stages 4 --base_filter 64 --n_resblock 3 \
# --lr 1e-4 --batch_size 24 --patch_size 128 --epochs 1000 --loss 1*L1 --print_every 20 --test_every 300 --data_test Set5 --save_results #--reset --weight_decay 1e-4 --chop

##GAN
# CUDA_VISIBLE_DEVICES=2,3,4 python main.py --n_GPUs 3  --model resDBPN_2 --scale 2 --save resDBPN2_v4_x2_bs20_96_ps128_k10_res3_GAN_2 --ext sep --base_filter 96 --n_resblock 3 \
# --lr 1e-4 --loss 6*VGG54+0.1*GAN --batch_size 16 --patch_size 128 --epochs 1000 --print_every 20 --test_every 300 --data_test Set5 --save_results #--reset --weight_decay 1e-4 --chop --loss 1*L1

#my test
# CUDA_VISIBLE_DEVICES=4 python main.py --n_GPUs 1 --model resDBPN_2 --data_test Demo --scale 4 --base_filter 96 --n_resblock 5  --pre_train ../experiment/resDBPN2_v4_x4_bs20_96_ps96_k10_res5_L1/model/model_best.pt --test_only --save_results \
# --dir_demo ../test/testsmall_jpg  #--chop

CUDA_VISIBLE_DEVICES=2 python main.py --n_GPUs 1 --model resDBPN_2 --data_test Demo --scale 2 --num_stages 4 --base_filter 64 --n_resblock 3  --pre_train ../experiment/resDBPN2_small_x2_bs24_64_ps128_k4_res3_L1_SE_stage4_dilated/model/model_best.pt --test_only --save_results \
--dir_demo ../test/testsmall_jpg  #--chop



-----------------------------------------------
import torch
import math
import torch.nn.functional as F
from model.se_module import SELayer


class sub_pixel(torch.nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(
            torch.nn.Conv2d(64, 64 * 64, 3, 1, 1)
        )
        modules.append(torch.nn.PixelShuffle(scale))
        self.body = torch.nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x


class Upsampler(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, scale=8, bn=False, activation='prelu', bias=True):
        super(Upsampler, self).__init__()
        padding = kernel_size // 2
        modules = []
        self.act = False
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        if (scale & (scale - 1)) == 0:    # Is scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(
                    torch.nn.Conv2d(num_filter, 4 * num_filter, kernel_size, stride, padding, bias=bias)
                )
                modules.append(torch.nn.PixelShuffle(2))
                if bn: modules.append(torch.nn.BatchNorm2d(num_filter))
                if self.act: modules.append(self.act)
        elif scale == 3:
            modules.append(
                torch.nn.Conv2d(num_filter, 9 * num_filter, kernel_size, stride, padding, bias=bias)
            )
            modules.append(torch.nn.PixelShuffle(3))
            if bn: modules.append(torch.nn.BatchNorm2d(num_filter))
            if self.act: modules.append(self.act)
        else:
            raise NotImplementedError

        self.body = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)



class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, dilated=False):
        super(ConvBlock, self).__init__()
        if dilated:
            self.conv = torch.nn.Conv2d(input_size, output_size, 4, 2, 3, dilation=2)
        else:
            self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        # kernel_size = 4
        # stride = 2
        # padding = 1
        # self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, dilation=1, output_padding=0, bias=bias)
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu'):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        # self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        self.res_scale = 1

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)

        return out
	#
    # def forward(self, x):
    #     residual = x
    #     out = self.conv1(x)
    #     if self.activation is not None:
    #         out = self.act(out)
    #     out = self.conv2(out)
	#
    #     out = torch.add(out.mul(self.res_scale), residual)
    #     return out


class make_dense(torch.nn.Module):
	def __init__(self, nChannels, growthRate, kernel_size=3):
		super(make_dense, self).__init__()
		self.conv = torch.nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
							  bias=False)

	def forward(self, x):
		out = F.relu(self.conv(x))
		# out = torch.cat((x, out), 1)
		return out

# Residual dense block (RDB) architecture
class RDB(torch.nn.Module):
	def __init__(self, nChannels, nDenselayer, growthRate):
		super(RDB, self).__init__()
		nChannels_ = nChannels
		modules = []
		for i in range(nDenselayer):
			nFeats = growthRate * (i + 1)
			modules.append(make_dense(nChannels_, nFeats))
			nChannels_ = nFeats
		self.dense_layers = torch.nn.Sequential(*modules)
		# self.conv_1x1 = torch.nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

	def forward(self, x):
		out = self.dense_layers(x)
		# out = self.conv_1x1(out)
		# out = out + x
		return out

#
# # Residual dense block (RDB) architecture
# class RDB(nn.Module):
# 	def __init__(self, nChannels, nDenselayer, growthRate):
# 		super(RDB, self).__init__()
# 		nChannels_ = nChannels
# 		modules = []
# 		for i in range(nDenselayer):
# 			modules.append(make_dense(nChannels_, growthRate))
# 			nChannels_ += growthRate
# 		self.dense_layers = nn.Sequential(*modules)
# 		self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
#
# 	def forward(self, x):
# 		out = self.dense_layers(x)
# 		out = self.conv_1x1(out)
# 		out = out + x
# 		return out

bool_pixshuffer = False

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', n_resblock = 3, norm=None):
        super(UpBlock, self).__init__()
        # self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        # #####body:RDB
        # nChannel = num_filter
        # nDenselayer = n_resblock
        # growthRate = num_filter // n_resblock
        # self.body = RDB(nChannel, nDenselayer, growthRate)

        ###body:without dense
        modules_body = [
            ResBlock(num_filter, 3, 1, padding=1, bias=bias, activation=activation) \
            for _ in range(n_resblock)
        ]
        self.body = torch.nn.Sequential(*modules_body)

        self.se = SELayer(num_filter, 16)

        modules_up = []
        if bool_pixshuffer:
            modules_up.append(
				Upsampler(num_filter, 3, 1, 1, scale=stride, bias=bias, activation=activation)
			)
        else:
            modules_up.append(
				# Upsampler(num_filter, 3, 1, 1, scale=stride, bias=bias, activation=activation)
				# sub_pixel(scale=stride)
				DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
			)
            modules_up.append(
                ConvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
				# DeconvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
			)
        self.up1 = torch.nn.Sequential(*modules_up)
        # self.up2 = torch.nn.Sequential(*modules_up)

    def forward(self, x):
        # res = self.body(x)
        # x = self.up1(x)
        # res_out = self.up2(res)
        # out = res_out + x
        # return out, res

        res_out = self.body(x)
        res_out = self.se(res_out)
        out = res_out + x
        out = self.up1(out)
        return out, res_out

class UpBlock_D(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', n_resblock = 3, norm=None):
        super(UpBlock_D, self).__init__()
        self.cut_channel = ConvBlock(num_filter * 2, num_filter, 1, 1, 0, activation, norm=None)

        # self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)


        # #####body:RDB
        # nChannel = num_filter
        # nDenselayer = n_resblock
        # growthRate = num_filter // n_resblock
        # self.body = RDB(nChannel, nDenselayer, growthRate)

        ###body:without dense
        modules_body = [
            ResBlock(num_filter, 3, 1, padding=1, bias=bias, activation=activation) \
            for _ in range(n_resblock)
        ]
        self.body = torch.nn.Sequential(*modules_body)

        self.se = SELayer(num_filter, 16)

        modules_up = []
        if bool_pixshuffer:
            modules_up.append(
				Upsampler(num_filter, 3, 1, 1, scale=stride, bias=bias, activation=activation)
			)
        else:
            modules_up.append(
				# Upsampler(num_filter, 3, 1, 1, scale=stride, bias=bias, activation=activation)
				# sub_pixel(scale=stride)
				DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
			)
            modules_up.append(
                ConvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
				# DeconvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
			)
        self.up1 = torch.nn.Sequential(*modules_up)
        # self.up2 = torch.nn.Sequential(*modules_up)

    def forward(self, x, res_x):
        # res = self.body(x)
        # concat_res = torch.cat((res, res_x), 1)
        # concat_res = self.cut_channel(concat_res)
        # res_out = self.up1(concat_res)
        # x = self.up2(x)
        # out = res_out + x
        # return out, res

        res_out = self.body(x)
        concat_res = torch.cat((res_out, res_x), 1)
        res = self.cut_channel(concat_res)
        res = self.se(res)
        out = res + x
        out = self.up1(out)
        return out, res_out

class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', n_resblock = 3, norm=None):
        super(D_UpBlock, self).__init__()
        self.cut_channel = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.cut_channel_res = ConvBlock(num_filter * (num_stages + 1), num_filter, 1, 1, 0, activation, norm=None)
        # self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)


        # #####body:RDB
        # nChannel = num_filter
        # nDenselayer = n_resblock
        # growthRate = num_filter // n_resblock
        # self.body = RDB(nChannel, nDenselayer, growthRate)

        modules_body = []
        for _ in range(n_resblock):
            modules_body.append(
                ResBlock(num_filter, 3, 1, 1, bias, activation)
            )
        self.body = torch.nn.Sequential(*modules_body)

        self.se = SELayer(num_filter, 16)

        modules_up = []
        if bool_pixshuffer:
            modules_up.append(
				Upsampler(num_filter, 3, 1, 1, scale=stride, bias=bias, activation=activation)
			)
        else:
            modules_up.append(
				# Upsampler(num_filter, 3, 1, 1, scale=stride, bias=bias, activation=activation)
				# sub_pixel(scale=stride)
				DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
			)
            modules_up.append(
                ConvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
				# DeconvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
			)
        self.up1 = torch.nn.Sequential(*modules_up)
        # self.up2 = torch.nn.Sequential(*modules_up)

    def forward(self, x, res_x):
        # x = self.cut_channel(x)
        # res = self.body(x)
        # concat_res = torch.cat((res, res_x), 1)
        # concat_res = self.cut_channel_res(concat_res)
        # x = self.up1(x)
        # res_out = self.up2(concat_res)
        # out = res_out + x
        # return out, res

        x = self.cut_channel(x)
        res_out = self.body(x)
        concat_res = torch.cat((res_out, res_x), 1)
        res = self.cut_channel_res(concat_res)
        res = self.se(res)
        out = res + x
        out = self.up1(out)
        return out, res_out

class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', n_resblock = 3, norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None, dilated=True)
        # self.down_conv11 = DeconvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
        # self.down_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        # self.down_conv22 = DeconvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)

        # modules_body = [
        #     ResBlock(num_filter, 3, 1, padding=1, bias=bias, activation=activation) \
        #     for _ in range(n_resblock)
        # ]
        # self.body = torch.nn.Sequential(*modules_body)

    def forward(self, x):
        # res = self.body(x)
        # x = self.down_conv1(x)
        # # x = self.down_conv11(x)
        # res_out = self.down_conv2(res)
        # # res_out = self.down_conv22(res_out)
        # out = res_out + x
        # return out, res

        # res_out = self.body(x)
        # out = res_out + x
        # out = self.down_conv1(out)

        res_out = x
        out = self.down_conv1(x)
        return out, res_out

class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', n_resblock = 3, norm=None):
        super(D_DownBlock, self).__init__()
        self.cut_channel = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.cut_channel_res = ConvBlock(num_filter * (num_stages), num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None, dilated=True)
        # self.down_conv11 = DeconvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
        # self.down_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        # self.down_conv22 = DeconvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)

        # modules_body = []
		#
        # for _ in range(n_resblock):
        #     modules_body.append(
        #         ResBlock(num_filter, 3, 1, 1, bias, activation)
        #     )
        # self.body = torch.nn.Sequential(*modules_body)

    def forward(self, x, res_x):
        # x = self.cut_channel(x)
        # res = self.body(x)
        # concat_res = torch.cat((res, res_x), 1)
        # concat_res = self.cut_channel_res(concat_res)
        # l1 = self.down_conv1(x)
        # # l1 = self.down_conv11(l1)
        # res_out = self.down_conv2(concat_res)
        # # res_out = self.down_conv22(res_out)
        # out = res_out + l1
        # return out, res

        # x = self.cut_channel(x)
        # res_out = self.body(x)
        # concat_res = torch.cat((res_out, res_x), 1)
        # res = self.cut_channel_res(concat_res)
        # out = res + x
        # out = self.down_conv1(out)

        x = self.cut_channel(x)
        res_out = x
        concat_res = torch.cat((res_out, res_x), 1)
        out = self.cut_channel_res(concat_res)
        out = self.down_conv1(out)
        return out, res_out
