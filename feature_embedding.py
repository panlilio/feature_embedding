import torch
from torch import nn
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
from tqdm.auto import tqdm

class AE(nn.Module):
    def __init__(self, input_size=(128,128,3), output_channels_init=64, depth=4, nfeats_final=8,
                 activation=nn.ReLU(), **kwargs):
        super().__init__()
        self.encode = []
        self.decode = []

        h,w,input_channels = input_size

        for i in range(depth):
            if i==0:
                in_channels = int(input_channels)
            else:
                in_channels = int(output_channels_init * 2 ** (i - 1))
            out_channels = int(output_channels_init * 2 ** i)

            self.encode.append(nn.Conv2d(in_channels,out_channels,(3,3),padding=1))
            self.encode.append(nn.InstanceNorm2d(out_channels))
            self.encode.append(activation)

            self.encode.append(nn.Conv2d(out_channels, out_channels,(3, 3),padding=1))
            self.encode.append(nn.InstanceNorm2d(out_channels))
            self.encode.append(activation)

            self.encode.append(nn.MaxPool2d((2,2)))

            h,w = h//2,w//2
            nfeats = out_channels

        self.encode.append(nn.Flatten())

        final_encoder_conv_size = int(h * w * nfeats)
        self.encode.append(nn.Linear(final_encoder_conv_size,nfeats_final))

        #####

        self.decode.append(nn.Linear(nfeats_final, final_encoder_conv_size))
        self.decode.append(nn.Unflatten(1,(nfeats,w,h)))
        for i in range(depth-1,-1,-1):
            #Note the decoder here is constructed in reverse order
            if i==0:
                in_channels = input_channels
            else:
                in_channels = int(output_channels_init * 2 ** (i - 1))
            out_channels = int(output_channels_init * 2 ** (i))

            self.decode.append(nn.Upsample(scale_factor=2))
            self.decode.append(nn.Conv2d(out_channels,out_channels,(3,3),padding=1))
            self.decode.append(nn.InstanceNorm2d(out_channels))
            self.decode.append(activation)
            self.decode.append(nn.Conv2d(out_channels,in_channels,(3,3),padding=1))
            self.decode.append(nn.InstanceNorm2d(in_channels))
            self.decode.append(activation)

            h, w = h * 2, w * 2

        self.encode = nn.Sequential(*self.encode)
        self.decode = nn.Sequential(*self.decode)

    def forward(self,x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded

#######################

class microglia_dataset(Dataset):
    def __init__(self,
                    filenames,
                    crop_size=128,
                    split='train',
                    rotate_90=True,
                    dtype=np.uint8,
                 ):
        self.filenames = filenames
        self.crop_size=crop_size
        self.split=split
        self.rotate_90=rotate_90
        self.dtype=dtype

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = io.imread(self.filenames[idx]).astype(np.float32)
        image = image/np.iinfo(self.dtype).max
        image = torch.from_numpy(image)
        if self.split.lower()=='train':
            image = self.augment_image(image)
        image = self.crop_im(image)
        image = torch.permute(image,(2,1,0))
        return image

    def augment_image(self,im):
        im = transforms.RandomVerticalFlip(p=0.5)(im)
        im = transforms.RandomHorizontalFlip(p=0.5)(im)
        #Random rotation of Z*90 deg
        if self.rotate_90:
            theta = np.random.randint(0,4)*90
        im = transforms.functional.rotate(im,theta)
        im = self.crop_im(im)
        return im

    def crop_im(self,im):
        # Random crop within image bounds
        if im.shape[1] != self.crop_size:
            x0 = torch.randint(0, im.shape[1] - self.crop_size,(1,1))
        else:
            x0 = 0

        if im.shape[0] != self.crop_size:
            y0 = torch.randint(0, im.shape[0] - self.crop_size,(1,1))
        else:
            y0 = 0
        im = im[y0:y0 + self.crop_size, x0:x0 + self.crop_size,:]
        return im

def model_step(model, loss_fn, optimizer, x, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    predicted, encoded = model(x)

    # pass through loss
    loss_value = loss_fn(input=predicted, target=x)

    # backward if training mode
    if train_step:
        loss_value.backward()
        optimizer.step()

    outputs = {
        'predicted': predicted,
        'encoded': encoded,
    }

    return loss_value, outputs

def train(train_loaded, holdout_loaded, model, loss_fn, optimizer, training_steps, device='cpu',writer=None):
    # set train flags, initialize step
    model.train()
    loss_fn.train()
    step = 0

    with tqdm(total=training_steps) as pbar:
        while step < training_steps:
            # reset data loader to get random augmentations
            np.random.seed()
            tmp_loaded = iter(train_loaded)
            for image in tmp_loaded:
                image = image.to(device)
                loss_value, _ = model_step(model, loss_fn, optimizer, image, train_step=True)
                if writer is not None:
                    writer.add_scalar('loss',loss_value.cpu().detach().numpy(),step)
                step += 1
                pbar.update(1)
                if step % 100 == 0:
                    model.eval()
                    tmp_val_loader = iter(holdout_loaded)
                    acc_loss = []
                    for im in tmp_val_loader:
                        im = im.to(device)
                        loss_value, pred = model_step(model, loss_fn, optimizer, im, train_step=True)
                        acc_loss.append(loss_value.cpu().detach().numpy())
                    writer.add_scalar('holdout_loss',np.mean(acc_loss),step)
                    writer.add_image('holdout_image_predicted', pred["predicted"])
                    writer.add_image('holdout_image_actual',im)
                    model.train()
                    print(np.mean(acc_loss))
