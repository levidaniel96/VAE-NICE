"""Training procedure for VAE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from VAE import Model

def train(vae, trainloader, optimizer, epoch, device):
    vae.train()  # set to training mode
    
    epoch_loss = 0
    for inputs,_ in trainloader:
        inputs = inputs.to(device)      
        # zero the parameter gradients
        optimizer.zero_grad()
        # loss
        loss = vae(inputs).mean() 
        loss.backward()
        optimizer.step()  
        epoch_loss += loss.item()
    
    print(f'epoch {epoch}: epoch_loss = {epoch_loss}')
                
    return epoch_loss


def test(vae, testloader, filename, epoch, device):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        
        epoch_loss = 0
        for inputs,_ in testloader:              
            inputs =  inputs.to(device)  
            # loss
            loss = vae(inputs).mean() 
            epoch_loss += loss.item()
        
        if (epoch % 10 == 0) or (epoch == (args.epochs-1)):
            samples = vae.sample(args.sample_size)
            torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
            
    return epoch_loss


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(in_out_dim=args.in_out_dim,mid_dim=args.mid_dim,device=device, coupling_type=args.coupling_type, coupling=args.coupling,hidden=args.hidden, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)
    
    train_loss = []
    test_loss = []

    for epoch in range(args.epochs):
        train_loss.append(train(vae, trainloader, optimizer, epoch, device) / len(trainloader))
        print(f'epoch(mean) {epoch}: epoch_loss = {train_loss[epoch]}')

        test_loss.append(test(vae, testloader, filename, epoch, device) / len(testloader)) 
        print(f'epoch(mean) {epoch}: epoch_loss = {test_loss[epoch]}')
    
    # Plots
    plt.figure(1) 
    plt.plot(train_loss) 
    plt.plot(test_loss) 
    plt.xlabel('Epochs') 
    plt.ylabel('-ELBO') 
    plt.legend(['train','test'])
    plt.grid(True)
    plt.title(f'Dataset: {args.dataset}')
    plt.savefig(f'{args.dataset}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--latent_dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)    
    parser.add_argument('--coupling_type',
                        help='.',
                        type=str,
                        default='affine')
    parser.add_argument('--coupling',
                        help='.',
                        type=int,
                        default=0)
    parser.add_argument('--mid_dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=0)
    parser.add_argument('--in_out_dim',
                        help='.',
                        type=int,
                        default=1*28*28)

    args = parser.parse_args()
    main(args)
