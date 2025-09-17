import glob
import os.path
import SimpleITK as sitk
import numpy as np
import shutil
from functions import largestConnectComponent
import pandas as pd


def get_meta(img):
    """
    Extract metadata from SimpleITK image

    Args:
        img: SimpleITK image object

    Returns:
        list: [origin, size, spacing, direction]
    """
    meta_data = []
    meta_data.append(img.GetOrigin())
    meta_data.append(img.GetSize())
    meta_data.append(img.GetSpacing())
    meta_data.append(img.GetDirection())
    return meta_data


def set_meta(img, meta_data):
    """
    Set metadata to image (numpy array or SimpleITK image)

    Args:
        img: numpy array or SimpleITK image
        meta_data: metadata list from get_meta()

    Returns:
        SimpleITK image with metadata
    """
    if type(img) == np.ndarray:
        img = sitk.GetImageFromArray(img)
    img.SetOrigin(meta_data[0])
    img.SetSpacing(meta_data[2])
    img.SetDirection(meta_data[3])
    return img


def crop(img, mask, gtv, crop_shape=[32, 256, 256]):
    """
    Crop image and mask around GTV center

    Args:
        img: input image array (CT)
        mask: mask array (PET mask)
        gtv: GTV segmentation array
        crop_shape: desired crop dimensions [z, y, x]

    Returns:
        tuple: (cropped_ct, cropped_mask)
    """
    # Find GTV region coordinates
    mask_situation = np.where(gtv > 0)

    # Calculate center of GTV region
    z_center = (np.max(mask_situation[0]) + np.min(mask_situation[0])) // 2
    y_center = (np.max(mask_situation[1]) + np.min(mask_situation[1])) // 2
    x_center = (np.max(mask_situation[2]) + np.min(mask_situation[2])) // 2

    # Calculate crop start coordinates
    z_start = z_center - crop_shape[0] // 2
    y_start = y_center - crop_shape[1] // 2
    x_start = x_center - crop_shape[2] // 2

    # Perform cropping
    ct_cropped = img[z_start:z_start + crop_shape[0],
                 y_start:y_start + crop_shape[1],
                 x_start:x_start + crop_shape[2]]

    pet_mask_cropped = mask[z_start:z_start + crop_shape[0],
                       y_start:y_start + crop_shape[1],
                       x_start:x_start + crop_shape[2]]

    return ct_cropped, pet_mask_cropped


# Main processing pipeline
path = r'D:\Original_data'  # Path to original data
path_all = glob.glob(os.path.join(path, '*'))

for pa in path_all:
    # Extract patient ID from path
    ID = os.path.basename(pa).split('_')[0]  # Fixed path separator issue

    # Define file paths
    ct_path = os.path.join(pa, 'PETCT.nii.gz')  # PET/CT image
    pet_path = os.path.join(pa, 'PET.nii.gz')  # PET image
    GTV_path = os.path.join(pa, 'GTV.nii.gz')  # GTV segmentation
    brain_path = os.path.join(pa, 'Brain_Oral.nii.gz')  # Brain/oral cavity segmentation

    # Read images
    ct = sitk.ReadImage(ct_path)  # PET/CT image
    pet = sitk.ReadImage(pet_path)  # PET image
    gtv = sitk.ReadImage(GTV_path)  # GTV segmentation
    brain = sitk.ReadImage(brain_path)  # Brain/oral cavity segmentation

    # Extract metadata and convert to numpy arrays
    img_meta = get_meta(pet)
    ct = sitk.GetArrayFromImage(ct)
    pet = sitk.GetArrayFromImage(pet)
    gtv = sitk.GetArrayFromImage(gtv)
    brain = sitk.GetArrayFromImage(brain)

    # Create PET image with only GTV region (set non-GTV to zero)
    pet_gtv = pet.copy()  # Copy PET image
    pet_gtv[gtv == 0] = 0  # Set non-GTV regions to zero
    pet_gtv[brain == 2] = 0  # Remove brain regions (value 2 in brain mask)

    # Copy CT image for processing
    ct_gtv = ct.copy()

    # Create SUV-based mask (threshold = 2.5)
    pet_mask = np.zeros(shape=pet_gtv.shape)  # Initialize empty mask
    threshold = 2.5  # SUV threshold for MTV segmentation
    pet_mask[pet_gtv >= threshold] = 1  # Create binary mask based on SUV threshold

    # Remove bone (CT > 100 HU) and air cavities (CT < -50 HU) from mask
    pet_mask[ct_gtv > 100] = 0  # Exclude bone
    pet_mask[ct_gtv < -50] = 0  # Exclude air cavities

    # Process oral cavity segmentation
    oral_cavity = np.zeros(shape=brain.shape)
    oral_cavity[brain == 1] = 1  # Extract oral cavity region (value 1 in brain mask)
    oral_cavity_processed, oral_cavity_num = largestConnectComponent(
        oral_cavity) * 1  # Keep largest connected component

    # Remove regions above oral cavity
    z_oral_max = np.max(np.where(oral_cavity_processed == 1)[0])  # Find upper boundary of oral cavity
    pet_mask[:z_oral_max, :, :] = 0  # Remove mask regions above oral cavity

    # Keep original mask for cropping
    pet_mask_original = pet_mask.copy()

    # Crop images around GTV center
    ct_crop, pet_mask_crop = crop(ct_gtv, pet_mask_original, gtv)
    pet_crop, _ = crop(pet, pet_mask_original, gtv)
    gtv_crop, _ = crop(gtv, pet_mask_original, gtv)
    brain_crop, _ = crop(brain, pet_mask_original, gtv)

    # Define output filenames
    suv_mask_name = r'suv_mask.nii.gz'  # SUV-based MTV mask
    ct_crop_name = r'ct_crop.nii.gz'  # Cropped CT image
    pet_crop_name = r'pet_crop.nii.gz'  # Cropped PET image
    gtv_crop_name = r'gtv_crop.nii.gz'  # Cropped GTV segmentation
    brain_crop_name = r'brain_crop.nii.gz'  # Cropped brain/oral cavity segmentation

    # Save processed images with original metadata
    sitk.WriteImage(set_meta(sitk.GetImageFromArray(pet_mask_crop), img_meta),
                    os.path.join(pa, suv_mask_name))  # Save MTV mask

    sitk.WriteImage(set_meta(sitk.GetImageFromArray(ct_crop), img_meta),
                    os.path.join(pa, ct_crop_name))  # Save cropped CT

    sitk.WriteImage(set_meta(sitk.GetImageFromArray(pet_crop), img_meta),
                    os.path.join(pa, pet_crop_name))  # Save cropped PET

    sitk.WriteImage(set_meta(sitk.GetImageFromArray(gtv_crop), img_meta),
                    os.path.join(pa, gtv_crop_name))  # Save cropped GTV

    sitk.WriteImage(set_meta(sitk.GetImageFromArray(brain_crop), img_meta),
                    os.path.join(pa, brain_crop_name))  # Save cropped brain

print("MTV automatic generation completed successfully!")