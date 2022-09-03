from feature_embedding import *
from glob import glob
import os
import numpy as np
from torchsummary import summary
import datetime
from torch.utils.tensorboard import SummaryWriter


#Get list of all images
# data_dir = "/mnt/shared/teamleader/hackathon_data_uniform"
data_dir = "/Users/mpanlili/Library/CloudStorage/OneDrive-St.JudeChildren'sResearchHospital/Data/hackathon_data_uniform"
expression = '*.tif'
data_files = glob(os.path.join(data_dir,expression))

#Split images into holdout vs training
holdout = 0.2
rand_idx = np.random.permutation(np.arange(len(data_files))).astype(int)
split_idx = int((1-holdout)*len(data_files))

data_train = [data_files[i] for i in rand_idx[:split_idx]]
data_holdout = [data_files[i] for i in rand_idx[split_idx:]]

train_dataset = microglia_dataset(filenames=data_train, split='train')
holdout_dataset = microglia_dataset(filenames=data_holdout, split='valid')

#Initialize model
model = AE(input_size=train_dataset[0].shape[-1:None:-1],depth=3,nfeats_final=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)

summary(model,input_size=train_dataset[0].shape)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(logdir)

#Hyperparameters
training_steps = 100
train_batch_size = 32
holdout_batch_size = 1
loss_fn = torch.nn.MSELoss()

train_loaded = DataLoader(train_dataset,batch_size=train_batch_size)
holdout_loaded = DataLoader(holdout_dataset,batch_size=holdout_batch_size)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x = torch.unsqueeze(train_dataset[0],0)


train(train_loaded, holdout_loaded, model, loss_fn, optimizer, training_steps, device=device,writer=writer)

model.eval()

writer.close()

if not(os.path.isdir("models")):
	os.mkdir("models")

model_name = f"{os.path.basename(logdir)}.pth"
model_name_fullpath = os.path.join("models",model_name)
torch.save(model,model_name_fullpath)

print(f"Logged in: {logdir}")
print(f"Model saved to: {model_name_fullpath}")
