import torch
import torch.nn as nn
# import from dyneODE
from stylegan2.model import Generator

class ODEfunc(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.depth = depth
        layers = []
        for i in range(depth-1):
            layers += [nn.Linear(dim, dim), nn.LeakyReLU(0.2)]
        layers.append(nn.Linear(dim, dim))
        self.model = nn.Sequential(*layers)

    def forward(self, t, x, eps=1e-6):
        out = self.model(x)
        out = out / (torch.norm(out, dim=1, keepdim=True) + eps)
        return out

def load_generator(args):
    print("Loading StyleGAN2 generator at 256x256 (e4e compatible)")
    g_ema = Generator(args.image_size, 512, 8, channel_multiplier=1)


    # Load e4e-style checkpoint
    ckpt = torch.load(args.stylegan_ckpt, map_location='cpu')
    state_dict = ckpt['state_dict']

    # Extract and remap decoder weights
    decoder_state_dict = {
        k.replace('decoder.', ''): v
        for k, v in state_dict.items()
        if k.startswith('decoder.')
    }

    # Load generator weights (ignore missing/unexpected keys safely)
    missing, unexpected = g_ema.load_state_dict(decoder_state_dict, strict=False)
    print(f" Loaded generator with {len(missing)} missing and {len(unexpected)} unexpected keys")

    g_ema.eval()
    for param in g_ema.parameters():
        param.requires_grad = False

    return g_ema