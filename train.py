## Import libraries
"""

import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings("ignore")

!pip install wandb

import wandb
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## Load datasets"""

train_data_path = "/kaggle/input/isic-skin-lesion-dataset-2016/Train_data-001/Train_data"
test_data_path = "/kaggle/input/isic-skin-lesion-dataset-2016/Test-20240422T183231Z-001/Test/Test_data/Test_data"

demo_img1 = plt.imread("/kaggle/input/isic-skin-lesion-dataset-2016/Train_data-001/Train_data/ISIC_0024327.jpg")
plt.imshow(demo_img1)

# SHAPE OF ONE OF THE TRAIN IMAGES
demo_img1.shape

train_labels_df = pd.read_csv("/kaggle/input/isic-skin-lesion-dataset-2016/Train-20240422T183231Z-002/Train/Train_labels.csv")
train_labels_df.tail()

test_labels_df = pd.read_csv("/kaggle/input/isic-skin-lesion-dataset-2016/Test-20240422T183231Z-001/Test/Test_labels.csv")
test_labels_df.tail()

"""## Preprocessing"""

class CustomDataset():
    def __init__(self, root_dir, labels_df, transform=None):
        self.root_dir = root_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(self.labels_df.iloc[idx, 1:], dtype=torch.float32)

        return image, labels

batch_size = 32
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data = CustomDataset(root_dir=train_data_path, labels_df=train_labels_df, transform=transform)
test_data = CustomDataset(root_dir=test_data_path, labels_df=test_labels_df, transform=transform)

train_size = int(0.8*len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

def get_variance(dataset):
    images = []
    for i in range(len(dataset)):
        image, _ = dataset[i]
        images.append(image)

    images_tensor = torch.stack(images)
    variance = torch.var(images_tensor)
    return variance.item()

x_var = get_variance(train_data)

"""## Residuals"""

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace = True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class StackedResiduals(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(StackedResiduals, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)

        x = F.relu(x)
        return x

"""## Encoder"""

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            StackedResiduals(h_dim, h_dim, res_h_dim, n_res_layers)
         )

    def forward(self, x):
        return self.cnn(x)

"""## Quantizer"""

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
      #  z (continuous) -> z_q (discrete)
      #  z.shape = (batch, c, H, H)

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

"""## Decoder"""

class Decoder(nn.Module):
    """
    Given a latent sample z p_phi
    maps back to the original space z -> x.
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            StackedResiduals(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=4,stride=2, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)

"""## VQ-VAE"""

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)

        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)

        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)

        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def forward(self, x, flag=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity

"""
Hyperparameters
"""
batch_size = 32
epochs = 10000
n_hiddens = 128
n_residual_hiddens = 32
n_residual_layers = 2
n_embeddings = 512
embedding_dim = 64
lr = 3e-4
beta = 0.25

model = VQVAE(n_hiddens, n_residual_hiddens, n_residual_layers, n_embeddings, embedding_dim, beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

results = {
    'epochs': 0,
    'recon_losses': [],
    'losses': [],
    'perplexities': [],
}

model.train()

def train(model, optimizer, dataloader, epochs):
    wandb.init(project='vqvae_V1', entity='m23mac009')
    wandb.watch(model)

    for i in range(epochs):
        (x, label) = next(iter(dataloader))
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["epochs"] = i+1
        results["recon_losses"].append(recon_loss.cpu().detach().numpy())
        results["losses"].append(loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())

        if i % 50 == 0:
            wandb.log({'Epoch': i+1,
                      'Reconstruction loss': np.mean(results["recon_losses"][-50:]),
                      'Overall Loss': np.mean(results["losses"][-50:])})

        if i % 100 == 0:
            res_to_save = {'model': model.state_dict(),
                           'results': results,
                           }

            torch.save(res_to_save, "/kaggle/working/vqvae_model.pth")

            print(f'\nEpoch {i+1}', '\nReconstruction loss:', np.mean(results["recon_losses"][-100:]),
                  '\nLoss:', np.mean(results["losses"][-100:]),
                  '\nPerplexity:', np.mean(results["perplexities"][-100:]))

    wandb.save('model_vqvae.pth')
    wandb.finish()

train(model, optimizer, train_loader, epochs=7500)

"""## Reconstruct images (show in a grid)"""

def show_image_grid(x_orig, x_recon, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Creating a subplot with 1 row and 2 columns

    # Plot original images
    x_orig_grid = make_grid(x_orig.cpu().detach() + 0.5).numpy()
    axs[0].imshow(np.transpose(x_orig_grid, (1, 2, 0)), interpolation='nearest')
    axs[0].set_title('Original Images')
    axs[0].axis('off')

    # Plot reconstructed images
    x_recon_grid = make_grid(x_recon.cpu().detach() + 0.5).numpy()
    axs[1].imshow(np.transpose(x_recon_grid, (1, 2, 0)), interpolation='nearest')
    axs[1].set_title('Reconstructed Images')
    axs[1].axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def reconstruct(data_loader,model):
    model.eval()
    (x, _) = next(iter(data_loader))
    x = x.to(device)
    vq_enc_out = model.pre_quantization_conv(model.encoder(x))
    _, z_q, _, _,e_indices = model.vector_quantization(vq_enc_out)

    x_recon = model.decoder(z_q)
    return x, x_recon, z_q, e_indices

x_val, x_val_recon, z_q, e_indices = reconstruct(val_loader, model)
print(x_val.shape)
show_image_grid(x_val, x_val_recon, '/kaggle/working/gen_img_grid_run2.png')

"""## For inference during demonstration"""

def show_img(x_orig, x_recon, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Creating a subplot with 1 row and 2 columns

    # Plot original image
    axs[0].imshow(np.transpose(x_orig.cpu().detach().numpy(), (1, 2, 0)), interpolation='nearest')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot reconstructed image
    axs[1].imshow(np.transpose(x_recon.cpu().detach().numpy(), (1, 2, 0)), interpolation='nearest')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def inference(test_image, model, save_path):
    model.eval()
    with torch.no_grad():
        x = test_image.unsqueeze(0).to(device) # test img should be tensor
        vq_enc_out = model.pre_quantization_conv(model.encoder(x))
        _, z_q, _, _, e_indices = model.vector_quantization(vq_enc_out)
        x_recon = model.decoder(z_q)
        show_img(x[0], x_recon[0], save_path=save_path)

# test image tensor
inference(test_image, model, save_path="/kaggle/working/test_img_result.png")

"""## Auto-regressive model using PixelCNN"""

hp = {
    'epochs': 50,
    'n_embeddings': 512,
    'hidden_dim': 64,
    "img_dim": 8,
    'num_layers': 15,
    'lr': 0.001,
    'batch_size': 32
}

# # LOAD VQ-VAE MODEL
# model = VQVAE(n_hiddens, n_residual_hiddens, n_residual_layers, n_embeddings, embedding_dim, beta).to(device)
# model.load_state_dict(torch.load('/kaggle/working/vqvae_model.pth'))

def initialize_weights(m):
    cname = m.__class__.__name__
    if cname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print(f"init of {cname} skipped")

class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=7):
        super(GatedMaskedConv2d, self).__init__()
        self.mask_type = mask_type
        self.residual = residual
        self.class_cond_embedding = nn.Embedding(n_classes, 2 * dim)
        self.vert_stack = nn.Conv2d(dim, dim * 2,(kernel // 2 + 1, kernel) , 1, (kernel // 2, kernel // 2))
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.horiz_stack = nn.Conv2d(dim, dim * 2,(1, kernel // 2 + 1), 1, (0, kernel // 2))
        self.horiz_resid = nn.Conv2d(dim, dim, 1)

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()
        self.horiz_stack.weight.data[:, :, :, -1].zero_()

    def gated_activation(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.tanh(x1) * torch.sigmoid(x2)

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()
        out_v = self.gated_activation(self.vert_stack(x_v)[:, :, :x_v.size(-1), :] + self.class_cond_embedding(h)[:, :, None, None])
        out = self.gated_activation(self.vert_to_horiz(self.vert_stack(x_v)[:, :, :x_v.size(-1), :]) + self.horiz_stack(x_h)[:, :, :, :x_h.size(-2)] + self.class_cond_embedding(h)[:, :, None, None])
        out_h = self.horiz_resid(out) + x_h if self.residual else self.horiz_resid(out)


        return out_v, out_h

class PixelCNN(nn.Module):
    def __init__(self, input_dim, dim, n_layers, n_classes=7):
        super(PixelCNN, self).__init__()
        self.dim = dim
        self.embedding = nn.Embedding(input_dim, dim)
        self.layers = nn.ModuleList([GatedMaskedConv2d('A', dim, 7, False, n_classes)] + [GatedMaskedConv2d('B', dim, 3, True, n_classes)for _ in range(n_layers - 1)])
        self.output_conv = nn.Sequential(nn.Conv2d(dim, 512, 1),nn.ReLU(True),nn.Conv2d(512, input_dim, 1))
        self.apply(initialize_weights)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp).permute(0, 3, 1, 2)
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        device = next(self.parameters()).device
        x = torch.zeros((batch_size, *shape), dtype=torch.int64, device=device)
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x[:, i, j] = probs.multinomial(1).squeeze()
        return x

pcnn_model = PixelCNN(hp['n_embeddings'], hp['img_dim']**2, hp['num_layers']).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(pcnn_model.parameters(), lr=hp['lr'])

def train_pcnn():

    pcnn_model.train()
    wandb.init(project='pcnn_model_V2', entity='m23mac009')
    wandb.watch(pcnn_model)

    train_losses = []
    for batch_idx, (x, label) in enumerate(train_loader):
        vq_out = model.pre_quantization_conv(model.encoder(x.to(device)))
        _, _, _,_, code_indices = model.vector_quantization(vq_out)      # Get min codebook indices
        code_indices = (code_indices[:, 0])
        print("Type of code indices ", type(code_indices))

        code_indices = code_indices.view(hp['batch_size'], 8, 8)
        code_indices = code_indices.long()
        code_indices = code_indices.to(device)
        label = label.to(device)

        # Train PixelCNN with images
        logits = pcnn_model(code_indices, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, hp['n_embeddings']),
            code_indices.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (batch_idx + 1) % 100 == 0:
            print(f'\nLoss: {np.asarray(train_losses)[-100:].mean(0)}')

    return np.asarray(train_losses).mean(0)


def test_pcnn():
    pcnn_model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(val_loader):
            vq_out = model.pre_quantization_conv(model.encoder(x.to(device)))
            _, _, _,_, code_indices = model.vector_quantization(vq_out)
            code_indices = (code_indices[:, 0]).long()
            code_indices = code_indices.to(device)
            label = label.to(device)

            logits = pcnn_model(x, label)

            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(logits.view(-1, hp['n_embeddings']), code_indices.view(-1))

            val_losses.append(loss.item())

    print(f'Validation Loss: {np.asarray(val_losses).mean(0)}')
    return np.asarray(val_losses).mean(0)

def get_samples(epoch):
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().to(device)

    x_hat = pcnn_model.generate(label, shape=(hp['img_dim'],hp['img_dim']), batch_size=100)

    print(x_hat[0])

min_loss = 1000
saved_epoch = 0
for epoch in range(1, hp['epochs']):
    print(f"\nEpoch {epoch}:")
    pcnn_train_losses = train_pcnn()

    wandb.log({'Epoch': epoch,
               'train loss': pcnn_train_losses})

    cur_loss = test_pcnn()

    if cur_loss <= min_loss:
        min_loss = cur_loss
        saved_epoch = epoch

        torch.save(pcnn_model.state_dict(), '/kaggle/working/pcnn_model.pth')

    else:
        print(f"\nLast saved in epoch: {saved_epoch}")

    get_samples(epoch)
