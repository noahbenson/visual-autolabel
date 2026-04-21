# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/image/_model.py
# Training / validation data based on images of cortex.


#===============================================================================
# Dependencies

import math
from collections.abc import Mapping

import scipy.sparse as sps
import torch
from torch import nn
import torchvision.transforms.functional as F

from ..util import convrelu, convrelu3D


#===============================================================================
# Image-based CNN Model Code

class UNet3D(nn.Module):
    """a U-Net with a ResNet18 backbone for learning visual area labels.

    The `UNet` class implements a ["U-Net"](https://arxiv.org/abs/1505.04597)
    with a [ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/) backbone.
    The class inherits from `torch.nn.Module`. This particular UNet is designed
    for use with 3D images.
    
    Parameters
    ----------
    feature_count : int
        The number of channels (features) in the input image. When using an
        `HCPVisualDataset` object for training, this value should be set to 4
        if the dataset uses the `'anat'` or `'func'` features and 8 if it uses
        the `'both'` features.
    segment_count : int
        The number of segments (AKA classes, labels) in the output data. For
        V1-V3 this is typically either 3 (V1, V2, V3) or 6 (LV1, LV2, LV3, RV1,
        RV2, RV3).
    base_model : model name or tuple, optional
        The name of the model that is to be used as the base/backbone of the
        UNet. The default is `'resnet18'`, but `'resnet34'` is a valid option.
    logits : boolean, optional
        Whether the model should return logits (`True`) or probabilities
        (`False`). The default is `True`.

    Attributes
    ----------
    base_model : PyTorch Module
        The ResNet-18 model that is used as the backbone of the `UNet` model.
    base_layers : list of PyTorch Modules
        The ResNet-18 layers that are used in the backbone of the `UNet` model.
    feature_count : int
        The number of input channels (features) that the model expects in input
        images.
    segment_count : int
        The number of segments (labels) predicted by the model.
    logits : bool
        `True` if the output of the model is in logits and `False` if its output
        is in probabilities.
    """
    def __init__(self, feature_count, segment_count,
                 base_model='resnet18',
                 logits=True):
        # Make sure we can import the resnet3D library:
        from . import kenshohara_resnet as resnetlib
        # Initialize the super-class.
        super().__init__()
        # Store some basic attributes.
        self.feature_count = feature_count
        self.segment_count = segment_count
        self.logits = logits
        # Set up the base model and base layers for the model. Start by parsing
        # the base model parameter.
        self.base_model = base_model
        if isinstance(base_model, str):
            if base_model.startswith('resnet'):
                base_model = base_model[6:]
            base_model_n = int(base_model)
        else:
            base_model_n = base_model
        if not isinstance(base_model_n, int):
            raise ValueError(f'invalid base_model: {self.base_model}')
        base_model = resnetlib.generate_model(
            base_model_n,
            n_input_channels=feature_count,
            n_classes=segment_count,
            conv1_t_stride=2)
        # fix the base model:
        self.base_model = f'resnet{base_model_n}'
        # Because the input size may not be 3 and the output size may not be 3,
        # we want to add an additional set of layers to make all this work.
        # Adjust the first convolution's number of input channels.
        c1 = base_model.conv1
        base_model.conv1 = nn.Conv3d(
            feature_count, c1.out_channels,
            kernel_size=c1.kernel_size,
            stride=c1.stride,
            padding=c1.padding,
            bias=c1.bias)
        base_layers = list(base_model.children())
        #self.base_layers = base_layers
        # Make the U-Net layers out of the base-layers.
        # size = (N, 64, H/2, W/2)
        self.layer0 = nn.Sequential(*base_layers[:3]) 
        self.layer0_1x1 = convrelu3D(64, 64, 1, 0)
        # size = (N, 64, H/4, W/4)
        self.layer1 = nn.Sequential(*base_layers[3:5])
        self.layer1_1x1 = convrelu3D(64, 64, 1, 0)
        # size = (N, 128, H/8, W/8)        
        self.layer2 = base_layers[5]
        self.layer2_1x1 = convrelu3D(128, 128, 1, 0)  
        # size = (N, 256, H/16, W/16)
        self.layer3 = base_layers[6]  
        self.layer3_1x1 = convrelu3D(256, 256, 1, 0)  
        # size = (N, 512, H/32, W/32)
        self.layer4 = base_layers[7]
        self.layer4_1x1 = convrelu3D(512, 512, 1, 0)
        # The up-swing of the UNet; we will need to upsample the image.
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='trilinear',
                                    recompute_scale_factor=True,
                                    align_corners=True)
        self.conv_up3 = convrelu3D(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu3D(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu3D(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu3D(64 + 256, 128, 3, 1)
        self.conv_original_size0 = convrelu3D(feature_count, 64, 3, 1)
        self.conv_original_size1 = convrelu3D(64, 64, 3, 1)
        self.conv_original_size2 = convrelu3D(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv3d(64, segment_count, 1)
    def forward(self, input):
        # Do the original size convolutions.
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        # Now the front few layers, which we save for adding back in on the UNet
        # up-swing below.
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        # Now, we start the up-swing; each step must upsample the image.
        layer4 = self.layer4_1x1(layer4)
        # Up-swing Step 1
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        # Up-swing Step 2
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        # Up-swing Step 3
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        # Up-swing Step 4
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        # Up-swing Step 5
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        # And the final convolution.
        out = self.conv_last(x)
        if not self.logits:
            out = torch.sigmoid(out)
        return out

class UNet2D(torch.nn.Module):
    """A U-Net with a ResNet18 backbone for learning visual area labels.

    The `UNet` class implements a ["U-Net"](https://arxiv.org/abs/1505.04597)
    with a [ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/) bacbone.
    The class inherits from `torch.nn.Module`.
    
    The original implementation of this class was by Shaoling Chen
    (sc6995@nyu.edu), and additional modifications have been made by Noah C.
    Benson (nben@uw.edu).

    Parameters
    ----------
    feature_count : int
        The number of channels (features) in the input image. When using an
        `HCPVisualDataset` object for training, this value should be set to 4
        if the dataset uses the `'anat'` or `'func'` features and 8 if it uses
        the `'both'` features.
    segment_count : int
        The number of segments (AKA classes, labels) in the output data. For
        V1-V3 this is typically either 3 (V1, V2, V3) or 6 (LV1, LV2, LV3, RV1,
        RV2, RV3).
    base_model : model name or tuple, optional
        The name of the model that is to be used as the base/backbone of the
        UNet. The default is `'resnet18'`, but 
    pretrained : boolean, optional
        Whether to use a pretrained base model for the backbone (`True`) or not
        (`False`). The default is `False`.
    logits : boolean, optional
        Whether the model should return logits (`True`) or probabilities
        (`False`). The default is `True`.

    Attributes
    ----------
    pretrained_base : boolean
        `True` if the base model used in this `UNet` was originally pre-trained
        and `False` otherwise.
    base_model : PyTorch Module
        The ResNet-18 model that is used as the backbone of the `UNet` model.
    base_layers : list of PyTorch Modules
        The ResNet-18 layers that are used in the backbone of the `UNet` model.
    feature_count : int
        The number of input channels (features) that the model expects in input
        images.
    segment_count : int
        The number of segments (labels) predicted by the model.
    logits : bool
        `True` if the output of the model is in logits and `False` if its output
        is in probabilities.
    """
    def __init__(self, feature_count, segment_count,
                 base_model='resnet18',
                 pretrained=False,
                 logits=True):
        import torch.nn as nn
        # Initialize the super-class.
        super().__init__()
        # Store some basic attributes.
        self.feature_count = feature_count
        self.segment_count = segment_count
        self.pretrained = pretrained
        self.logits = logits
        # Set up the base model and base layers for the model.
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        import torchvision.models as mdls
        base_model = getattr(mdls, base_model)
        try:
            base_model = base_model(weights=weights,
                                    num_classes=segment_count)
        except TypeError:
            base_model = base_model(pretrained=pretrained,
                                    num_classes=segment_count)
        # Not sure we should store the base model; seems like a good idea, but
        # does it get caught up in PyTorch's Module data when we do?
        #self.base_model = resnet18(pretrained=pretrained)
        # Because the input size may not be 3 and the output size may not be 3,
        # we want to add an additional 
        if feature_count != 3:
            # Adjust the first convolution's number of input channels.
            c1 = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                self.feature_count, c1.out_channels,
                kernel_size=c1.kernel_size, stride=c1.stride,
                padding=c1.padding, bias=c1.bias)
        base_layers = list(base_model.children())
        #self.base_layers = base_layers
        # Make the U-Net layers out of the base-layers.
        # size = (N, 64, H/2, W/2)
        self.layer0 = nn.Sequential(*base_layers[:3]) 
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # size = (N, 64, H/4, W/4)
        self.layer1 = nn.Sequential(*base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        # size = (N, 128, H/8, W/8)        
        self.layer2 = base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        # size = (N, 256, H/16, W/16)
        self.layer3 = base_layers[6]  
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        # size = (N, 512, H/32, W/32)
        self.layer4 = base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        # The up-swing of the UNet; we will need to upsample the image.
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size0 = convrelu(feature_count, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, segment_count, 1)
    def forward(self, input):
        # Do the original size convolutions.
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        # Now the front few layers, which we save for adding back in on the UNet
        # up-swing below.
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        # Now, we start the up-swing; each step must upsample the image.
        layer4 = self.layer4_1x1(layer4)
        # Up-swing Step 1
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        # Up-swing Step 2
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        # Up-swing Step 3
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        # Up-swing Step 4
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        # Up-swing Step 5
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        # And the final convolution.
        out = self.conv_last(x)
        if not self.logits:
            out = torch.sigmoid(out)
        return out


'''
# The Hybrid 2d/3d CNN:
class HybridUNet(nn.Module):
    """A hybrid 2D/3D UNet CNN.
    """
    def __init__(self,
                 feature_count_3D, output_count_3D,
                 feature_count_2D, segment_count,
                 base_model='resnet18',
                 logits=True):
        nn.Module.__init__(self)
        self.unet3D = UNet3D(
            feature_count_3D,
            output_count_3D,
            base_model=base_model,
            logits=logits)
        self.unet2D = UNet2D(
            feature_count_2D + output_count_3D,
            segment_count,
            base_model=base_model,
            logits=logits)
    def forward(self, inputs3D, inputs2D, tx_3D_to_2D):
        # In the forward function, input data always has a particular shape:
        # shape is B x C x H x W [x D]
        #  Where B is batch size (number of training examples beign given)
        #  ...   C is number of image channels
        #  ...   H is height
        #  ...   W is width
        #  ...   D is depth (for a 3D CNN; no D for a 2D CNN).
        #
        # First, run the 3D CNN:
        output3D = self.unet3D(inputs3D)
        # Transform the 3D outputs into 2D images:
        shape2D = inputs2D.shape[-2:]
        flat_output3D = self.transform_3D_to_2D(output3D, tx_3D_to_2D, shape2D)
        # Combine the flattened 3D data with the 2D inputs.
        inputs_combined_2D = torch.cat(
            [inputs2D, flat_output3D],
            dim=1)
        # Run the 2D CNN:
        segments = self.unet2D(inputs_combined_2D)
        return segments
    def transform_3D_to_2D(self, data3D, tx_3D_to_2D, shape2D):
        # tx_3D_to_2D needs to be a giant sparse matrix; the shape of the matrix
        #   should be (N x M) where N is the number of cells in data3D and M is the
        #   number of pixels in the output image (shape_2D[0] * shape_2D[1]).
        # data3D has shape (B, C, Rs, Cs, Ss) where B is batches and C is channels;
        #   we want to convert it to (B, C, rows2D, columns2D).
        # shape_2D is the (rows2D, columns2D) in the 2D images we are producing and
        #   so M = shape_2D[0] * shape_2D[1]
        (rows2D, columns2D) = shape2D
        # To make the output 2D image, we iterate over the channels (dim 1) of the
        #   3D data.
        (nbatches, nchannels, nrows, ncols, nslices) = data3D.shape
        data3D_flat = torch.reshape(
            data3D,
            # single channel has dims B x H x Rs x Cs x Ss;
            # we flatten to B x H x N (N = Rs * Cs * Ss)
            (nbatches, nchannels, nrows * ncols * nslices))
        data2D_flat = [
            torch.mm(tx_item, data3D_item.T)
            for (tx_item, data3D_item) in zip(tx_3D_to_2D, data3D_flat)]
        data2D_flat = torch.stack(data2D_flat)
        # First reshape to the large (full res) size of the 2D image; these images
        # have the same aspect ratio as shape2D but may be a different number of
        # pixels.
        aspect = shape2D[1] / shape2D[0]
        full_n = data2D_flat.shape[1]
        zoom = math.sqrt(shape2D[0]*shape2D[1] / full_n)
        w = round(shape2D[0] / zoom)
        h = round(shape2D[1] / zoom)
        full_im = torch.reshape(data2D_flat, (nbatches, nchannels, w, h))
        ret = F.resize(full_im, shape2D)
        return ret

'''

class HybridUNet(nn.Module):
    def __init__(self,
                 feature_count_3D,
                 feature_count_2D,
                 segment_count,
                 base_model='resnet18',
                 logits=True):
        nn.Module.__init__(self)
        self.feature_count_3D = feature_count_3D
        self.feature_count_2D = feature_count_2D
        self.segment_count = segment_count
        self.logits = logits
        base_model_n = int(base_model.replace('resnet', ''))
        self.base_model = f'resnet{base_model_n}'
 
        #3D Encoder
        from . import kenshohara_resnet as resnetlib
        base3D = resnetlib.generate_model(
            base_model_n,
            n_input_channels=feature_count_3D,
            n_classes=segment_count,
            conv1_t_stride=2)
                     
        c1 = base3D.conv1
        base3D.conv1 = nn.Conv3d(
            feature_count_3D, c1.out_channels,
            kernel_size=c1.kernel_size,
            stride=c1.stride,
            padding=c1.padding,
            bias=c1.bias)
                     
        layers3D = list(base3D.children())
        self.enc3D_layer0 = nn.Sequential(*layers3D[:3])
        self.enc3D_layer1 = nn.Sequential(*layers3D[3:5])
        self.enc3D_layer2 = layers3D[5]
        self.enc3D_layer3 = layers3D[6]
        self.enc3D_layer4 = layers3D[7]
 
        #2D Encoder
        import torchvision.models as mdls
        base2D = getattr(mdls, self.base_model)(
            weights=None,
            num_classes=segment_count)
                     
        layers2D = list(base2D.children())
        self.enc2D_layer0 = nn.Sequential(*layers2D[:3])
        self.enc2D_layer1 = nn.Sequential(*layers2D[3:5])
        self.enc2D_layer2 = layers2D[5]
        self.enc2D_layer3 = layers2D[6]
        self.enc2D_layer4 = layers2D[7]
 
        self._fusion_out_channels = 512
        self._fusion_out_h = 4
        self._fusion_out_w = 8
        self.fusion_linear = nn.LazyLinear(512 * 4 * 8)
                     
        # Shared 2D Decoder
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size0 = convrelu(feature_count_2D, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, segment_count, 1)
 
    def forward(self, inputs3D, inputs2D):
        # 3D Encode
        e3_0 = self.enc3D_layer0(inputs3D)
        e3_1 = self.enc3D_layer1(e3_0)
        e3_2 = self.enc3D_layer2(e3_1)
        e3_3 = self.enc3D_layer3(e3_2)
        e3_4 = self.enc3D_layer4(e3_3) 
 
        # 2D Encode
        x_original = self.conv_original_size0(inputs2D)
        x_original = self.conv_original_size1(x_original)
        e2_0 = self.enc2D_layer0(inputs2D)
        e2_1 = self.enc2D_layer1(e2_0)
        e2_2 = self.enc2D_layer2(e2_1)
        e2_3 = self.enc2D_layer3(e2_2)
        e2_4 = self.enc2D_layer4(e2_3) 
        
        N = e2_4.shape[0]
        e2_4_flat = e2_4.flatten(start_dim=1)
        e3_4_flat = e3_4.flatten(start_dim=1)
        combined = torch.cat([e2_4_flat, e3_4_flat], dim=1)
        fused_flat = self.fusion_linear(combined)
        fused = fused_flat.reshape(N, self._fusion_out_channels, self._fusion_out_h, self._fusion_out_w)
 
        # Decode with 2D Skip Connections
        x = self.upsample(fused)
        e2_3 = self.layer3_1x1(e2_3)
        x = torch.cat([x, e2_3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        e2_2 = self.layer2_1x1(e2_2)
        x = torch.cat([x, e2_2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        e2_1 = self.layer1_1x1(e2_1)
        x = torch.cat([x, e2_1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        e2_0 = self.layer0_1x1(e2_0)
        x = torch.cat([x, e2_0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        out = self.conv_last(x)
        if not self.logits:
            out = torch.sigmoid(out)
        return out

