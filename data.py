import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import nibabel as nib

def check_id(id, g_path, w_path):
    g_dir = os.path.join(g_path, id, 'Baseline')
    w_dir = os.path.join(w_path, id, 'Baseline', 'dti', 'dti_FA')

    # Check if all directories exist for gray matter and white matter
    if not os.path.exists(g_dir) or not os.path.exists(w_dir):
        return False

    # Check if there's a directory starting with 'anat_201' inside g_dir
    anat_dir_path = None
    for subdir in os.listdir(g_dir):
        if subdir.startswith('anat_201') and os.path.isdir(os.path.join(g_dir, subdir)):
            anat_dir_path = os.path.join(g_dir, subdir)
            break

    # If 'anat_201' directory not found, return False
    if not anat_dir_path:
        return False

    # Check if 'smwc1pT1.nii' exists inside the 'anat_201' directory
    if not os.path.exists(os.path.join(anat_dir_path, 'smwc1pT1.nii')):
        return False

    # Check if 'tbdti32ch_FA.nii.gz' exists inside w_dir
    if not os.path.exists(os.path.join(w_dir, 'tbdti32ch_FA.nii.gz')):
        return False

    # All conditions met, return True
    return True

def process_data(g_path, w_path):
    g_list = os.listdir(g_path)[:20]
    w_list = os.listdir(w_path)[:20]
    gray_paths = []
    white_paths = []

    common_ids = list(set(g_list) & set(w_list))
    id=pd.DataFrame(common_ids)

    valid_ids = [id for id in common_ids if check_id(id,g_path,w_path)]
    df_valid_id=pd.DataFrame(valid_ids)
    #df_valid_id.to_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/valid_common_id.csv')

    for id in valid_ids:
        g_dir = os.path.join(g_path, id, 'Baseline')
        w_dir = os.path.join(w_path, id, 'Baseline', 'dti', 'dti_FA')

        # Check if 'Baseline' directory exists for gray matter and white matter
        if not os.path.exists(g_dir) or not os.path.exists(w_dir):
            continue  # Skip this ID if directories don't exist

        # Check if there's a directory starting with 'anat_201' inside g_dir
        anat_dir_path = None
        for subdir in os.listdir(g_dir):
            if subdir.startswith('anat_201') and os.path.isdir(os.path.join(g_dir, subdir)):
                anat_dir_path = os.path.join(g_dir, subdir)
                break

        if not anat_dir_path:
            continue  # Skip this ID if 'anat_201' directory not found

        # Construct the path to 'smwc1pT1.nii' inside the 'anat_201' directory
        g_file = os.path.join(anat_dir_path, 'smwc1pT1.nii')
        w_file = os.path.join(w_dir, 'tbdti32ch_FA.nii.gz')

        gray_paths.append(g_file)
        white_paths.append(w_file)

    return gray_paths, white_paths

# Define custom dataset
class MRI_Dataset(Dataset):
    def __init__(self, gray_paths, white_paths, gray_mask, white_mask):
        self.gray_paths = gray_paths[:2000]
        self.white_paths = white_paths[:2000]
        self.gray_mask=gray_mask
        self.white_mask=white_mask
        
        self.gray_total=[]
        self.white_total=[]

    def __len__(self):
        return len(self.gray_paths)

    def __getitem__(self, idx):
        gray_path = self.gray_paths[idx]
        white_path = self.white_paths[idx]
        
        gray_img = nib.load(gray_path).get_fdata()
        white_img = nib.load(white_path).get_fdata()

        # gb_data_masked=gray_img[self.gray_mask]
        # wb_data_masked=white_img[self.white_mask]

        gb_data_masked=gray_img * self.gray_mask
        wb_data_masked=white_img * self.white_mask

        return gb_data_masked, wb_data_masked

# class DataProcessor:
#     @staticmethod
#     def process_data(g_path, w_path):
#         g_list = os.listdir(g_path)
#         w_list = os.listdir(w_path)
#         gray_paths = []
#         white_paths = []

#         for g_ids, w_ids in zip(g_list, w_list):
#             newpath_g = os.path.join(g_path, g_ids, 'Baseline', 'smwc1pT1.nii')
#             newpath_w = os.path.join(w_path, w_ids, 'Baseline', 'dti', 'dti_FA', 'tbdti32ch_FA.nii.gz')

#             if os.path.exists(newpath_g) and os.path.exists(newpath_w):
#                 gray_paths.append(newpath_g)
#                 white_paths.append(newpath_w)

#         return gray_paths, white_paths