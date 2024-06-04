import numpy as np

import sys
import os
import pickle as pkl
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from tqdm import tqdm
from PIL import Image
from typing import Type, Union, List, Tuple, Dict, Optional, Callable
from data_loader import load_mnist
import neunet as nnet
from neunet.optim import Adam
import neunet.nn as nn
from neunet import Tensor

# https://arxiv.org/abs/2006.11239
# https://arxiv.org/abs/2102.09672

# https://huggingface.co/blog/annotated-diffusion
# https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
# https://nn.labml.ai/diffusion/ddpm/index.html


def linear_schedule(start, end, timesteps):
    return nnet.tensor(np.linspace(start, end, timesteps, dtype = np.float32), requires_grad = False, device = device)


class Diffusion():
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, criterion, model = None):
        self.model = model

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        self.betas = linear_schedule(beta_start, beta_end, timesteps)
        self.sqrt_betas =  nnet.tensor(np.sqrt(self.betas.data), requires_grad = False, device = device)

        self.alphas = 1 - self.betas
        self.inv_sqrt_alphas =  nnet.tensor(1 / np.sqrt(self.alphas.data), requires_grad = False, device = device)
        
        self.alphas_cumprod =  nnet.tensor(np.cumprod(self.alphas.data, axis = 0), requires_grad = False, device = device)
        self.sqrt_alphas_cumprod = nnet.tensor(np.sqrt(self.alphas_cumprod.data), requires_grad = False, device = device)
        self.sqrt_one_minus_alphas_cumprod =  nnet.tensor(np.sqrt(1 - self.alphas_cumprod.data), requires_grad = False, device = device)

        self.scaled_alphas =  (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod

        # self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        # self.alphas_cumprod_prev = np.concatenate([np.array([1]), self.alphas_cumprod[:-1]]) #np.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


        self.criterion = criterion
        self.optimizer = Adam(self.model.parameters(), lr = 2e-4)


    def forward(self, x: Tensor, t = None):
        """
        https://arxiv.org/abs/2006.11239

        Algorithm 1: Training; according to the paper
        """

        timesteps_selection = np.random.randint(1, self.timesteps, (x.shape[0],))
        noise = nnet.tensor(np.random.normal(size = x.shape), requires_grad = False, device=x.device)
       
        x_t = self.sqrt_alphas_cumprod[timesteps_selection, None, None, None] * x + self.sqrt_one_minus_alphas_cumprod[timesteps_selection, None, None, None] * noise
        # print(f"init forward: {x_t.shape}, {noise.shape}")
        x = self.model.forward(nnet.tensor(x_t, requires_grad = False, device=x.device), timesteps_selection / self.timesteps)
        
        return x, noise



    def ddpm_denoise_sample(self, sample_num = None, image_size = None, states_step_size = 1, x_t = None, orig_x = None, mask = None):
        """
        https://arxiv.org/abs/2006.11239

        Algorithm 2: Sampling; according to the paper
        """

        if mask is not None:
            assert orig_x is not None
            assert orig_x.shape == mask.shape

        if x_t is None:
            if orig_x is None:
                x_t = np.random.normal(size = (sample_num, *image_size))
            else:
                x_t = np.random.normal(size = orig_x.shape)

        x_t = nnet.tensor(x_t, requires_grad = False, device=device)
        x_ts = []
        for t in tqdm(reversed(range(0, self.timesteps)), desc = 'ddpm denoisinig samples', total = self.timesteps):
            noise = nnet.tensor(np.random.normal(size = x_t.shape), requires_grad = False, device=device) if t > 1 else 0
            eps = self.model.forward(x_t, np.array([t]) / self.timesteps).reshape(x_t.shape).detach()

            x_t = self.inv_sqrt_alphas[t] * (x_t - eps * self.scaled_alphas[t]) + self.sqrt_betas[t] * noise
            # x_t = self.sqrt_recip_alphas[t] * (x_t - self.betas[t] * eps / self.sqrt_one_minus_alphas_cumprod[t]) + np.sqrt(self.posterior_variance[t]) * noise

            if mask is not None:
                orig_x_noise = nnet.tensor(np.random.normal(size = orig_x.shape), requires_grad = False, device=device)
                
                orig_x_t = self.sqrt_alphas_cumprod[t] * orig_x + self.sqrt_one_minus_alphas_cumprod[t] * orig_x_noise
                x_t = orig_x_t * mask + x_t * (1 - mask)

            if t % states_step_size == 0:
                x_ts.append(x_t.to('cpu').detach().data)

        return x_t.to('cpu').detach().data, x_ts

    def ddim_denoise_sample(self, sample_num = None, image_size = None, states_step_size = 1, eta = 1., perform_steps = 100, x_t = None, orig_x = None, mask = None):
        """
        https://arxiv.org/abs/2010.02502

        Denoising Diffusion Implicit Models (DDIM) sampling; according to the paper
        """

        if mask is not None:
            assert orig_x is not None
            assert orig_x.shape == mask.shape

        if x_t is None:
            if orig_x is None:
                x_t = np.random.normal(size = (sample_num, *image_size))
            else:
                x_t = np.random.normal(size = orig_x.shape)

        x_ts = []
        for t in tqdm(reversed(range(1, self.timesteps)[:perform_steps]), desc = 'ddim denoisinig samples', total = perform_steps):
            noise = np.random.normal(size = x_t.shape) if t > 1 else 0
            eps = self.model.forward(x_t, np.array([t]) / self.timesteps, training = False).reshape(x_t.shape)

            x0_t = (x_t - eps * np.sqrt(1 - self.alphas_cumprod[t])) / np.sqrt(self.alphas_cumprod[t])

            sigma = eta * np.sqrt((1 - self.alphas_cumprod[t - 1]) / (1 - self.alphas_cumprod[t]) * (1 - self.alphas_cumprod[t] / self.alphas_cumprod[t - 1]))
            c = np.sqrt((1 - self.alphas_cumprod[t - 1]) - sigma ** 2)

            x_t = np.sqrt(self.alphas_cumprod[t - 1]) * x0_t - c * eps + sigma * noise

            if mask is not None:
                orig_x_noise = np.random.normal(size = orig_x.shape)
                
                orig_x_t = self.sqrt_alphas_cumprod[t] * orig_x + self.sqrt_one_minus_alphas_cumprod[t] * orig_x_noise
                x_t = orig_x_t * mask + x_t * (1 - mask)
           
            if t % states_step_size == 0:
                x_ts.append(x_t)

        return x_t, x_ts



    def get_images_set(self, x_num: int, y_num: int, margin: int, images: np.float32, image_size: Tuple[int, int, int]):

        def denormalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
       

        channels, H_size, W_size = image_size

        images_array = np.full((y_num * (margin + H_size), x_num * (margin + W_size), channels), 255, dtype = np.uint8)
        num = 0
        for i in range(y_num):
            for j in range(x_num):
                y = i * (margin + H_size)
                x = j * (margin + W_size)

                images_array[y :y + H_size, x: x + W_size] = denormalize(images[num].transpose(1, 2, 0))

                num += 1

        images_array = images_array[: (y_num - 1) * (H_size + margin) + H_size, : (x_num - 1) * (W_size + margin) + W_size]

        if channels == 1:
            return Image.fromarray(images_array.squeeze(axis = 2)).convert("L")
        else:
            return Image.fromarray(images_array)


    def train(self, dataset, epochs, batch_size, image_path, image_size, save_every_epochs = 1):
        channels, H_size, W_size = image_size

        data_batches = np.array_split(dataset, np.arange(batch_size, len(dataset), batch_size))

        loss_history = []
        for epoch in range(epochs):
            tqdm_range = tqdm(enumerate(data_batches), total = len(data_batches))

            losses = []
            for batch_num, (batch) in tqdm_range:
                batch = batch.reshape(-1, channels, H_size, W_size)
                # print(batch.shape)
                output, noise = self.forward(nnet.tensor(batch, requires_grad = True, device = device))
                loss = self.criterion(output, noise)
                losses.append(loss.data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                tqdm_range.set_description(
                    f"loss: {losses[-1]:.7f} | epoch {epoch + 1}/{epochs}"
                )


                if batch_num == (len(data_batches) - 1):
                    epoch_loss = nnet.tensor(losses, device = device).mean().data

                    tqdm_range.set_description(
                            f"loss: {epoch_loss:.7f} | epoch {epoch + 1}/{epochs}"
                    )

            if ((epoch + 1) % save_every_epochs == 0):
    
                margin = 10
                x_num, y_num = 5, 5

                samples, samples_in_time = self.ddpm_denoise_sample(x_num * y_num, (channels, H_size, W_size))
                images_grid = self.get_images_set(x_num, y_num, margin, samples, (channels, H_size, W_size))
                images_grid.save(f"{image_path}/np_ddpm_{epoch + 1}.png")

                images_grid_in_time = []
                for sample in samples_in_time:
                    images_grid_in_time.append(self.get_images_set(x_num, y_num, margin, sample, (channels, H_size, W_size)))

                images_grid_in_time[0].save(f"{image_path}/np_ddpm_in_time.gif", save_all = True, append_images = images_grid_in_time[1:], duration = 50, loop = 0)
                
                
               
                
            loss_history.append(epoch_loss)

        return loss_history



class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_emb_dim, up = False):
        super().__init__()
        self.time_embedding =  nn.Linear(time_emb_dim, output_channels)
        self.input_channels = input_channels
        self.output_channels = output_channels
        if up:
            self.conv1 = nn.Conv2d(2 * input_channels, output_channels, kernel_size = (3, 3), padding = (1, 1))
            self.transform = nn.ConvTranspose2d(output_channels, output_channels, kernel_size = (4, 4), stride = (2, 2), padding = (1, 1))
            
        else:
            self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size = (3, 3), padding=(1, 1))
            self.transform = nn.Conv2d(output_channels, output_channels, kernel_size = (4, 4), stride = (2, 2),  padding = (1, 1))

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = (3, 3), padding=(1, 1))
        self.relu1  = nn.LeakyReLU(alpha = 0.01)
        self.relu2  = nn.LeakyReLU(alpha = 0.01)
        self.relu3  = nn.LeakyReLU(alpha = 0.01)

        self.bnorm1 = nn.BatchNorm2d(output_channels, momentum = 0.1, eps = 1e-5)
        self.bnorm2 = nn.BatchNorm2d(output_channels, momentum = 0.1, eps = 1e-5)
        
        
    def forward(self, x, t):
        x = self.conv1.forward(x)
        h = self.relu1.forward(x)
        h = self.bnorm1.forward(h)

        t = self.time_embedding.forward(t)
        
        time_emb = self.relu2.forward(t)

        time_emb = time_emb[(..., ) + (None, ) * 2]
      
        h = h + time_emb
      
        
        h = self.conv2.forward(h)
        h = self.relu3.forward(h)
        h = self.bnorm2.forward(h)
 

        return self.transform.forward(h)

class PositionalEncoding(nn.Module):
    """ Implements the sinusoidal positional encoding.
    """

    def __init__(self,max_len, d_model, dropout_rate=0.1, data_type = np.float32):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.data_type = data_type
 
        pe = np.zeros((max_len, d_model))  # (max_len, d_model)
        position = np.arange(0, max_len)[:, np.newaxis]# (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model,)

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = nnet.tensor(pe[:, np.newaxis, :].astype(self.data_type), requires_grad=False, device = device)   # (max_len, 1, d_model)


    def forward(self, x):
        """ x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.shape[0], :]  # (batch_size, seq_len, d_model)

        # x: (batch_size, seq_len, d_model)
        return x


class SimpleUNet(nn.Module):

    def __init__(self, image_channels,  image_size, down_channels = (32, 64, 128, 256, 512), up_channels = (512, 256, 128, 64, 32)):
      
        noise_channels = image_channels
        time_emb_dim = 32

       
        self.time_embedding = nn.Sequential(
            PositionalEncoding(max_len = 1000, d_model = time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.LeakyReLU()
        )
          
            
        
       
        if image_size & (image_size - 1) != 0:
            self.input_conv = nn.ConvTranspose2d(image_channels, down_channels[0], kernel_size=(5, 5))
            self.output_conv = nn.Conv2d(up_channels[-1], noise_channels,  kernel_size=(5, 5))
        else:
            self.input_conv = nn.Conv2d(image_channels, down_channels[0], kernel_size=(3, 3), padding = (1, 1))
            self.output_conv = nn.ConvTranspose2d(up_channels[-1], noise_channels,  kernel_size=(3, 3), padding = (1, 1))

       
        self.down_layers  = nn.ModuleList([ResBlock(down_channels[i], down_channels[i+1], time_emb_dim).to(device) for i in range(len(down_channels)-1)])
       
        self.up_layers = nn.ModuleList([ResBlock(up_channels[i], up_channels[i+1], time_emb_dim, up=True).to(device) for i in range(len(up_channels)-1)])

        

    def forward(self, x, t):
        # x = nnet.tensor(np.asarray(x))
        
        t = nnet.tensor(np.asarray(t[:, None, None], dtype = np.float32), requires_grad = False, device = x.device)

        # print(f"init simple unet input shape: {x.shape}, t.shape: {t.shape}")
        # for layer in self.time_embedding:
            
        #     # t = t.reshape(t.shape[0], -1)
        #     # print(f"t.shape: {t.shape}")
        #     t = layer.forward(t)
        t = self.time_embedding.forward(t)
        t = t.reshape(t.shape[0], -1)
        
        x = self.input_conv.forward(x)
        
        residual_inputs = []
        for down_layer in self.down_layers:
            x = down_layer.forward(x, t)
            residual_inputs.append(x)
          
        for up_layer in self.up_layers:
            residual_x = residual_inputs.pop()

            x = nnet.concatenate(*(x, residual_x), axis = 1)      
            x = up_layer.forward(x, t)
        return self.output_conv.forward(x)

        







device = "cuda"


diffusion = Diffusion(
    model = SimpleUNet(
        image_channels = 1, 
        image_size = 28, 
        down_channels = (32, 64, 128), 
        up_channels = (128, 64, 32)
        ).to(device), 
    timesteps = 300, 
    beta_start = 0.0001, 
    beta_end = 0.02, 
    criterion = nn.MSELoss(), 
    )

training_data, test_data, training_labels, test_labels = load_mnist()
training_data = training_data / 127.5 - 1 # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]

# diffusion.ddpm_denoise_sample(25, (1, 28, 28))
diffusion.train(training_data, epochs = 3, batch_size = 16, image_path = f"generated images", image_size = (1, 28, 28))


