import torch
from generator import Generator
from mappingmlp import MappingMLP
import math
from train import gen_images, save_generated_images
from calc_metrics import save_real_images
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from scipy.stats import truncnorm
from dataset import get_loader
from torch_fidelity import calculate_metrics

def load_model(path, dataset, res, device):
    checkpoint = torch.load(f"{path}/stylegan2_{dataset}_{res}.pt", map_location=device)
    generator = Generator(int(math.log2(res)),512).to(device) # keep latent dimensions to 512 in all experiments
    ema = Generator(int(math.log2(res)),512).to(device)
    best = Generator(int(math.log2(res)),512).to(device)
    mapping_net = MappingMLP(512, 8).to(device)
    generator.load_state_dict(checkpoint['generator'])
    mapping_net.load_state_dict(checkpoint['mapping_net'])
    ema.load_state_dict(checkpoint['ema'])
    best.load_state_dict(checkpoint['best_model'])
    return ema,best, generator, mapping_net

def main():
    dataset_name = "CelebA"
    generated_images = []
    to_test = 5000
    batch_size=32
    res = 64
    num_blocks = int(math.log2(res))-1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ema,best, generator, mapping_net = load_model("data",dataset_name, res, device)

    real_loader = get_loader(batch_size,res,dataset_name)
    save_real_images(real_loader)

    for i in range(0, to_test, batch_size):
        imgs, w = gen_images(batch_size, generator, num_blocks, 0.9, 512, mapping_net, device)
        generated_images.append(imgs)
    generated_images = torch.cat(generated_images, dim=0)
    save_generated_images(generated_images,"evaluation","CelebA",res)

    prc_dict = calculate_metrics(
        input1=f'./real', 
        input2=f'./data/diff_stylegan_{dataset_name}_{str(res)}/epoch_evaluation', 
        cuda=True, 
        isc=False, 
        fid=True, 
        kid=True, 
        prc=True, 
        verbose=False
    )
    inception_dict = calculate_metrics(
        input1=f'./data/diff_stylegan_{dataset_name}_{str(res)}/epoch_evaluation', 
        cuda=True, 
        isc=True, 
        fid=False, 
        kid=False, 
        prc=False, 
        verbose=False
    )
    prc_dict['inception_score_mean'] = inception_dict['inception_score_mean']
    print(prc_dict)
    return
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    imgs, w = gen_images(32, ema, num_blocks, 0.9, 512, mapping_net, device)
    imgs = (imgs + 1) / 2
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    imgs, w = gen_images(32, best , num_blocks, 0.9, 512, mapping_net, device)
    imgs = (imgs + 1) / 2
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    

def truncated_z_sample(batch_size, z_dim, truncation=0.5):
  values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim))
  return truncation * values

if __name__=="__main__":
    main()
