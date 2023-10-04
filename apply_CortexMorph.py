#!/usr/bin/env python
# coding: utf-8

# In[1]:




import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.autograd import Variable, Function
from nnunet.network_architecture.generic_UNet import Generic_UNet
import nibabel as nib
import copy
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.ndimage import label
from batchgenerators.augmentations.utils import pad_nd_image










class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_crop(volume):
    nonempty = np.argwhere(volume)
    top_left = nonempty.min(axis=0)
    bottom_right = nonempty.max(axis=0)
    
    return (top_left, bottom_right)

def apply_crop(volume, crop ):
    top_left, bottom_right = crop
    cropped = volume[top_left[0]:bottom_right[0]+1,
                   top_left[1]:bottom_right[1]+1,
                   top_left[2]:bottom_right[2]+1]
    return cropped

def reverse_crop(volume, crop, new_shape):
    top_left, bottom_right = crop 
    image_size = volume.shape
    if len(volume.shape) == 3:
        pad_width = ((top_left[0], (new_shape[0]-top_left[0]-image_size[0])),
                      (top_left[1], (new_shape[1]-top_left[1]-image_size[1])),
                      (top_left[2], (new_shape[2]-top_left[2]-image_size[2])))
    else:
        pad_width = ((0,0),(top_left[0], (new_shape[0]-top_left[0]-image_size[1])),
                      (top_left[1], (new_shape[1]-top_left[1]-image_size[2])),
                      (top_left[2], (new_shape[2]-top_left[2]-image_size[3])))
    return np.pad(volume, pad_width,mode='constant')


def transform_zooms(zooms, ornt_transform):
    # takes the zooms (voxel sizes) of an image, and returns the correct voxel sizes after applying 
    # a reorientation ornt_transform
    
    #per https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py we need
    #to invert the permutation to get the correct transformation
    ornt_perm = np.argsort(ornt_transform[:,0])
    return([zooms[int(x)] for x in ornt_perm])

def get_parcellation(parcellation, wm_image, gm_image, largest_component=False):
        affine = wm_image.affine
        orientation = nib.orientations.axcodes2ornt(nib.aff2axcodes(affine))
        forward_transform = nib.orientations.ornt_transform(orientation, orientation_fs)
        seg_data = nib.orientations.apply_orientation(parcellation.get_fdata(), forward_transform)
        gm = nib.orientations.apply_orientation(gm_image.get_fdata(), forward_transform)
        wm = nib.orientations.apply_orientation(wm_image.get_fdata(), forward_transform)
        gm_parcellation = seg_data
        gm_parcellation[np.logical_not(gm)] = 0
    # zooms gives the voxel size
    
        zooms = wm_image.header.get_zooms()
    

        
        crop = get_crop(gm>0.1)
        gm =apply_crop(gm, crop)
        wm = apply_crop(wm, crop)
        parcellation = apply_crop(gm_parcellation,crop)

        return wm, gm, parcellation, transform_zooms(zooms, forward_transform)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

def get_regional_averages(case_id, parcellation, wm_image, gm_image): 
    
    wm, gm, seg , zooms = get_parcellation(parcellation, wm_image, gm_image,False)    
        
    input, slicer =  pad_nd_image(np.stack([gm,wm],0), PATCH_SIZE, return_slicer=True)
    input_mirror, slicer_mirror =  pad_nd_image(np.stack([gm[::-1],wm[::-1]],0), PATCH_SIZE, return_slicer=True)
                
                
                
    with HiddenPrints():
        output = unet.predict_3D(torch.from_numpy(input).float().cuda(),do_mirroring=False, patch_size=PATCH_SIZE,
                                     use_sliding_window=True, use_gaussian = True)
        output_mirror = unet.predict_3D(torch.from_numpy(input_mirror).float().cuda(),do_mirroring=False, patch_size=PATCH_SIZE,
                                     use_sliding_window=True, use_gaussian = True)
        
        
    
    
    
    len_output = np.shape(output[1])[0]

    slicer[0] = slice(0,len_output,None)        
    output = output[1][tuple(slicer)]
    
    len_output = np.shape(output_mirror[1])[0]
    slicer_mirror[0] = slice(0,len_output,None)        

    output_mirror = output_mirror[1][tuple(slicer_mirror)]

    output_mirror = np.flip(output_mirror,1)
    output_mirror[0] = -output_mirror[0]

    
    
    
    output = (output+ output_mirror.copy())/2
    
    
    
    integrate = VecInt(output.shape[1:], 7)


    inverse_field = np.moveaxis(integrate(torch.from_numpy(output).unsqueeze(0))[0].numpy(),0,3)
    


        
    
    zooms = [zooms[2],zooms[1],zooms[0]]
    
    output = inverse_field*list(zooms)


    whiteMatter = wm>0.5
    
    #affine = nib.load(f'{casedir}softmax_seg.nii.gz').affine

    field = np.linalg.norm(output, axis=3)
    
    boundary = np.zeros_like(seg)
    nonboundary = np.zeros_like(seg)
    
    lh_boundary = np.zeros_like(seg).astype(bool)
    rh_boundary = np.zeros_like(seg).astype(bool)
    
    headers = []
    summary=[]
    
    for index, row in lut.iterrows():
        roi = np.logical_and(binary_dilation(seg==row[0]),whiteMatter)
        boundary_voxels = np.logical_and(roi, field>0.0001)
        nonboundary_voxels = np.logical_and(roi, field<0.1)
        boundary[boundary_voxels] = row[0]
        nonboundary[nonboundary_voxels] = row[0]
        headers.append('-'.join(row[1].split('-')[1:]))
        if np.sum(boundary_voxels>0):
            summary.append([np.mean(field[boundary_voxels])])
        else:
            summary.append([np.NaN])
        if row[0] < 2000:
            lh_boundary[boundary_voxels] = 1
        else:
            rh_boundary[boundary_voxels] = 1
    
    summary.append([np.mean(field[lh_boundary])])
    headers.append('lh-MeanThickness')
    
    summary.append([np.mean(field[rh_boundary])])
    headers.append('rh-MeanThickness')
    
    summary.append([np.prod(zooms)])
    
    headers.append('voxel_volume')
    
    df = pd.DataFrame(np.array(summary).T, columns=headers, index = [case_id])


    
    return df

if __name__ == "__main__":

	test_dir = 'test_data'
	OASIS_seg_dir = 'test_data'

	case_id = 'OAS30564_ses-d2808_CortexMorph_test'
	parcellation_file = test_dir+'/softmax_seg.nii.gz'
	wm_image_file = test_dir+'/wmprobT.nii.gz'
	gm_image_file = test_dir +'/gmprobT.nii.gz'
	stats_path = 'thickness_stats.csv'
	
	PATCH_SIZE = (128, 128, 128)
	TARGET_LABEL_NAMES = ['x', 'y', 'z']
	BASE_FEATURE_DEPTH=24
	POOL_MULTIPLIER = 1
	
	orientation_fs = nib.orientations.axcodes2ornt(('L', 'I', 'A'))




	parcellation = nib.load(parcellation_file)
	wm_image = nib.load(wm_image_file)
	gm_image = nib.load(gm_image_file)




	lut = pd.read_csv('freesurfer_cortex_lut.csv', sep='    ',engine='python').iloc[:,:2]




	unet = Generic_UNet(input_channels=2, base_num_features=BASE_FEATURE_DEPTH, num_classes=3, num_pool=3, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=POOL_MULTIPLIER, conv_op=nn.Conv3d,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU,deep_supervision=False,final_nonlin = nn.Identity(), seg_output_use_bias=True)
                 
	unet.load_state_dict(torch.load(f'cortexmorph_weights.pth.tar'))
	unet = unet.cuda()

	stats = get_regional_averages(case_id, parcellation, wm_image, gm_image)

	stats.to_csv(stats_path)







