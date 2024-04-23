import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import model
from data import process_data
# from correlation import correlation_loss
from torchvision.transforms import transforms
from data import MRI_Dataset
import nibabel as nib
import pickle
from alexnet_model_gray import AlexNet3D_Dropout
from alexnet_model_white import AlexNet3D_Dropout_White
from correlation import DCCA_Loss
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import random
import pandas as pd



# Paths to gray and white matter data
g_path = '/data/neuromark2/Data/ABCD/Data_BIDS_5/Raw_Data/'
w_path = '/data/neuromark2/Data/ABCD/DTI_Data_BIDS/Raw_Data/'

# # Process data
# gray_paths, white_paths = process_data(g_path, w_path)

# gray_total_data=np.zeros((121, 145, 121))
# white_total_data=np.zeros((182, 218, 182))

# for gp, wp in zip(gray_paths,white_paths):
#     gray_img=nib.load(gp).get_fdata()
#     gray_total_data +=gray_img

#     white_img=nib.load(wp).get_fdata()
#     white_total_data +=white_img

# gray_total_data_avg = gray_total_data /len(gray_paths)
# Baseline_mask=gray_total_data_avg>0.2

# white_total_data_avg = white_total_data /len(gray_paths)
# Baseline_mask_white=white_total_data_avg>0.2

# Create datasets and dataloaders
# dataset = MRI_Dataset(gray_paths, white_paths, Baseline_mask,Baseline_mask_white)


# # Paths to save the dataset
# dataset_save_path = '/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/dataset_full.pkl'

# # Save dataset using pickle
# with open(dataset_save_path, 'wb') as f:
#     pickle.dump(dataset, f)

# # Path to the saved dataset
# dataset_save_path = '/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/dataset.pkl'

# # Load dataset using pickle
# with open(dataset_save_path, 'rb') as f:
#     loaded_dataset = pickle.load(f)

_dccaLoss = DCCA_Loss()
# Open the file in binary mode for reading
with open("/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/dataset_full.pkl", "rb") as f:
    # Load the object from the file
    dataset = pickle.load(f)

dataset.gray_paths = dataset.gray_paths[:2000]
dataset.white_paths = dataset.white_paths[:2000]

# #Save dataset as torch pkl
# #########################################################################
seed = 42
torch.manual_seed(seed)
random.seed(seed)
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)  # 70% for training
val_size = int(0.15 * dataset_size)   # 15% for validation
test_size = dataset_size - train_size - val_size  # Remaining 15% for holdout

# Split the dataset into training, validation, and holdout sets
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each set
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)  
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)  

# Define hyperparameters
num_epochs = 20
lr = 0.0001

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
greynet = AlexNet3D_Dropout().to(device)
whitenet = AlexNet3D_Dropout_White().to(device)
greyoptimizer = optim.Adam(greynet.parameters(), lr=lr)
whiteoptimizer = optim.Adam(whitenet.parameters(), lr=lr)

#dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
torch.cuda.empty_cache()
print("a")

train_epoch_losses = []
train_batch_losses = []
train_epoch_val=[]

val_epoch_losses = []
val_batch_losses = []
val_epoch_val=[]

# # Training loop
for epoch in range(num_epochs):
    greynet.train()
    whitenet.train()
    total_loss = 0.0
    for idx, (gray_img, white_img) in enumerate(train_loader):
        gray_img, white_img = gray_img.to(device), white_img.to(device)

        greynet.zero_grad()
        whitenet.zero_grad()

        # Forward pass
        gray_features = greynet(gray_img.unsqueeze(1).float())
        white_features = whitenet(white_img.unsqueeze(1).float())

        # Replace NaN values with 0
        gray_features = torch.nan_to_num(gray_features, nan=0.0)
        white_features = torch.nan_to_num(white_features, nan=0.0)

        # Calculate correlation loss
        # loss = _dccaLoss.mean_square_error(gray_features, white_features)

        loss = nn.functional.mse_loss(gray_features, white_features)

        # Backward pass
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(greynet.parameters(), 1.0)
        # Clip gradients
        nn.utils.clip_grad_norm_(whitenet.parameters(), 1.0)

        greyoptimizer.step()
        whiteoptimizer.step()

        total_loss += loss.item()
        train_batch_losses.append(loss.item())
        # print("Loss for minibatch:" + str(idx)+" is:"+ str(loss.item()))
        #batch_loss=pd.DataFrame(train_batch_losses)
        #batch_loss.to_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/train_batch_loss.csv')

    epoch_loss = total_loss / len(train_loader)
    train_epoch_losses.append(epoch_loss) 
    print(f"Training Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss}")
    
    
    # Validation loop
    greynet.eval()
    whitenet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for gray_img, white_img in val_loader:
            gray_img, white_img = gray_img.to(device), white_img.to(device)

            # Forward pass
            gray_features = greynet(gray_img.unsqueeze(1).float())
            white_features = whitenet(white_img.unsqueeze(1).float())

            # Calculate correlation loss
            loss = _dccaLoss.correlation_loss(gray_features, white_features)

            val_loss += loss.item()
            val_batch_losses.append(val_loss) 
            # print(f"Validation Epoch [{epoch+1}/{num_epochs}], Validation Loss: {loss.item()}")
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_losses.append(val_epoch_loss) 
        print(f"Validation Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss}")

batch_loss=pd.DataFrame(train_batch_losses)
batch_loss.to_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/train_batch_loss.csv')

val_batch_loss=pd.DataFrame(val_batch_losses)
batch_loss.to_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/val_batch_loss.csv')

epoch_loss=pd.DataFrame(train_epoch_losses)
epoch_loss.to_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/train_epoch_loss.csv')

val_epoch_loss=pd.DataFrame(val_epoch_losses)
epoch_loss.to_csv('/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/val_epoch_loss.csv')

#np.savetxt("/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/gray_features.csv", gray_features, delimiter=",")

#np.savetxt("/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/white_features.csv", white_features, delimiter=",")

# Save the models
torch.save(greynet.state_dict(), '/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/greynet_model.pth')
torch.save(whitenet.state_dict(), '/data/users4/rsapkota/DCCA_AE/REDO_CODE/RESNET/whitenet_model.pth')

