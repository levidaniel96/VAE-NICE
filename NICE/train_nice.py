"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import trange
import matplotlib.pyplot as plt
import nice


def train(flow, trainloader, optimizer, epoch, device):
    flow.train()  # set to training mode
    epoch_loss = 0
    for inputs,_ in trainloader:
        inputs =  (inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2]*inputs.shape[3])).to(device) #change  shape from BxCxHxW to Bx(C*H*W)
        #TODO Fill in
        # zero the parameter gradients
        optimizer.zero_grad()
        # loss
        loss = -flow(inputs).mean() 
        loss.backward()
        optimizer.step()     
        epoch_loss += loss.item()
                
    return epoch_loss
        

def test(flow, testloader, filename, epoch, sample_shape, device):
    flow.eval()  # set to inference mode
    with torch.no_grad():
        epoch_loss = 0
        for inputs,_ in testloader:
            inputs = (inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2]*inputs.shape[3])).to(device) #change  shape from BxCxHxW to Bx(C*H*W)
            # loss
            loss = -flow(inputs).mean() 
            epoch_loss += loss.item()
         
        if (epoch % 10 == 0) or (epoch == (args.epochs-1)):
            samples = flow.sample(args.sample_size) + 0.5
            samples = samples.view(-1,sample_shape[0],sample_shape[1],sample_shape[2])
            torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                         './samples/' + filename + 'epoch%d.png' % epoch)

    return epoch_loss

def main(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    sample_shape = [1,28,28]
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)) #dequantization
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
             + 'batch%d_' % args.batch_size 

    flow = nice.NICE(
                prior=args.prior,
                coupling=args.coupling,
                coupling_type=args.coupling_type,
                in_out_dim=28*28,#args.full_dim, 
                mid_dim=args.mid_dim,
                hidden=args.hidden,
                device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    #TODO fill in
    
    train_loss = []
    test_loss = []

    for epoch in range(args.epochs):
        train_loss.append(train(flow, trainloader, optimizer, epoch, device) / len(trainloader))
        print(f'epoch(mean) {epoch}: epoch_loss = {train_loss[epoch]}')
        
        test_loss.append(test(flow, testloader, filename, epoch, sample_shape, device) / len(testloader)) 
        print(f'epoch(mean) {epoch}: epoch_loss = {test_loss[epoch]}')
    
    # Plots
    plt.figure(1) 
    plt.plot(train_loss) 
    plt.plot(test_loss) 
    plt.xlabel('Epochs') 
    plt.ylabel('Log Lielihood') 
    plt.legend(['train','test'])
    plt.grid(True)
    plt.title(f'Dataset: {args.dataset}, Coupling Type: {args.coupling_type}')
    plt.savefig(f'{args.dataset}_{args.coupling_type}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='gaussian')
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
    parser.add_argument('--coupling_type',
                        help='.',
                        type=str,
                        default='affine')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid_dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
