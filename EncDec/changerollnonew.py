from torchvision.transforms import Grayscale
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

all_cleans=[]
def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))
class AlteredMNIST(Dataset):
    def __init__(self,):
        # self.transform = ToTensor()
        self.grayscale = Grayscale()

        self.aug_list=os.listdir("Data/aug")
        self.clean_list=os.listdir("Data/clean")

        self.aug_labelwise = {}
        self.clean_labelwise = {}
        for i in range(10):
            self.aug_labelwise[i] = []
            self.clean_labelwise[i] = []

        for a_img in self.aug_list:
            label = int(a_img.split('_')[-1].split('.')[0])

            self.aug_labelwise[label].append("Data/aug/"+a_img)

        for c_img in self.clean_list:
            label = int(c_img.split('_')[-1].split('.')[0])

            self.clean_labelwise[label].append("Data/clean/"+c_img)
        
        self.train_pairs=[]
        for label, aug_imgs in self.aug_labelwise.items():
            image_paths=aug_imgs
            images=[]
            for image_path in image_paths:
                image = torchvision.io.read_image(image_path)
                image=image/255
                image = self.grayscale(image)
                images.append(image)
            images_tensor = torch.stack(images).view(len(images), -1)
            num_images = len(images)
            distances_matrix = torch.zeros(num_images, num_images)
            for i in range(num_images):
                for j in range(i + 1, num_images):
                    distance = euclidean_distance(images_tensor[i], images_tensor[j])
                    distances_matrix[i, j] = distance
                    distances_matrix[j, i] = distance
            distances_matrix=distances_matrix.tril()
            distance_dict = {}
            for i in range(num_images):
                for j in range(i):
                    distance = distances_matrix[i, j]
                    distance_dict[distance.item()] = (i, j)
            sorted_distances = sorted(distance_dict.keys())
            new_paths = set()
            for distance in sorted_distances[:1400]:
                row, col = distance_dict[distance]
                new_paths.add(image_paths[row])
                new_paths.add(image_paths[col])
            new_paths=list(new_paths)
            for i in (new_paths):
                self.train_pairs.append((i,label))
            

        self.average_intensities=[]
        for i in range(10):
            img_paths = self.clean_labelwise[i]
            imagess=[]
            for path in img_paths:
                img = torchvision.io.read_image(path)
                # img = self.transform(img)
                img=img/255
                img = self.grayscale(img)
                imagess.append(img)
            tensor_images = torch.stack(imagess).float()
            self.average_intensities.append(torch.mean(tensor_images, dim=0))
        
        self.labels=[]
        for idx in range(len(self.train_pairs)):
            aimg_ = self.train_pairs[idx][0]
            image = torchvision.io.read_image(aimg_)
            image=image/255
            image = self.grayscale(image)
            # image = self.transform(image)
            distances = [euclidean_distance(avg_img, image) for avg_img in self.average_intensities]
            highest_dissimilarity_idx = torch.argmax(torch.tensor(distances))
            highest_dissimilarity_idx =int(highest_dissimilarity_idx.numpy())
            self.labels.append(highest_dissimilarity_idx)
        
        self.all_cleans=[]
        for i in range(10):
            label=i
            all_ims=[]
            for i in range(100):
                pth = self.clean_labelwise[label][i]
                cimage = torchvision.io.read_image(pth)
                cimage=cimage/255
                cimage = self.grayscale(cimage)
                # cimage = self.transform(cimage)
                all_ims.append(cimage)
            all_ims=torch.stack(all_ims, axis=0)
            self.all_cleans.append(all_ims)
        global all_cleans
        all_cleans=self.all_cleans

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        aimg_ = self.train_pairs[idx][0]
        image = torchvision.io.read_image(aimg_)
        image=image/255
        image = self.grayscale(image)
        # image = self.transform(image)
        label=self.labels[idx]
        return {'aug_image': image,'label':label}
    

    




class Basic_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, enc=True):
        super(Basic_Resblock, self).__init__()
        self.enc=enc
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if self.enc==False:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        if (in_channels!=out_channels) or (stride>1) :
            self.change_size=True
        elif (padding == 0 and stride == 1):
            self.change_size=True
        else:
            self.change_size=False

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm(out)
        if self.change_size:
            skip = self.batch_norm(self.conv1(x))
        else:
            skip = x
        out += skip
        out = self.relu(out)
        return out
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.start = nn.Conv2d(1, 6, 3, 1, 1)
        self.start2 = nn.Conv2d(6,16, 3, 1, 1)
        self.aeseqs=[]
        temp1,temp2=16,16
        for i in range(3):
            tres=nn.Sequential(
            Basic_Resblock(in_channels=temp1, out_channels=temp1, kernel_size=3, stride=2, padding=1),
            Basic_Resblock(in_channels=temp1, out_channels=temp1+temp2, kernel_size=3, stride=1, padding=1))
            if i==2:
                tres=nn.Sequential(
                    Basic_Resblock(in_channels=temp1, out_channels=temp1, kernel_size=3, stride=2, padding=1),
                    Basic_Resblock(in_channels=temp1, out_channels=temp1+temp2, kernel_size=3, stride=2, padding=1))
            temp1+=temp2
            self.aeseqs.append(tres)

        self.last1=nn.Sequential(
            nn.ReLU(),nn.Linear(64*2*2, 128),nn.ReLU())
        self.last2=nn.Sequential(nn.Linear(128, 48),
            nn.ReLU()
        )
        self.mean_layers=nn.Sequential(
            nn.Linear(48, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.varience_layers=nn.Sequential(
            nn.Linear(48, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for seq in self.aeseqs:
            seq.to(device)

    def forward(self, input):
        out = self.start(input)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.start2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        for i in range(len(self.aeseqs)):
            out=self.aeseqs[i](out)
        out = out.view(out.size(0), -1)
        out = self.last2(self.last1(out))
        return out
    
    def forward1(self, input):
        out = self.start(input)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.start2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        for i in range(len(self.aeseqs)):
            out=self.aeseqs[i](out)
        out = out.view(out.size(0), -1)
        out = self.last2(self.last1(out))
        meann=self.mean_layers(out)
        var=self.varience_layers(out)
        std_dev = torch.exp(0.5*var)
        gen_= torch.randn_like(std_dev)
        gen_ = gen_*std_dev + meann
        return gen_,meann,var
    
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.aelast = nn.ConvTranspose2d(16, 6, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.aelast2 = nn.ConvTranspose2d(6, 1, 3, 1, 1)
        self.tanh2 = nn.Tanh()
        self.aeseqs=[]
        tres=nn.Sequential(
        Basic_Resblock(in_channels=64, out_channels=48, kernel_size=2, stride=2, padding=0,enc=False),
        Basic_Resblock(in_channels=48, out_channels=48, kernel_size=2, stride=2, padding=0,enc=False))
        self.aeseqs.append(tres)
        tres1=nn.Sequential(
        Basic_Resblock(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1,enc=False),
        Basic_Resblock(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0,enc=False))
        self.aeseqs.append(tres1)
        tres2=nn.Sequential(
        Basic_Resblock(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1,enc=False),
        Basic_Resblock(in_channels=16, out_channels=16, kernel_size=13, stride=1, padding=0,enc=False))
        self.aeseqs.append(tres2)
        self.first1 = nn.Sequential(nn.Linear(48, 128),
            nn.ReLU())
        self.first2 = nn.Sequential(
            nn.Linear(128, 64*2*2),
            nn.ReLU()
        )
        self.first0 = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20,48),
            nn.ReLU()
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for seq in self.aeseqs:
            seq.to(device)

    def forward(self, x):
        out = self.first1(x)
        out = self.first2(out)
        out = out.view(-1, 64, 2, 2)
        out=self.aeseqs[0](out)
        for i in range(1,len(self.aeseqs)):
            out=self.aeseqs[i](out)
        out = self.aelast(out)
        out = self.tanh(out)
        out = self.aelast2(out)
        out = self.tanh2(out)
        return out
    def forward1(self, x):
        out = self.first0(x)
        out = self.first1(out)
        out = self.first2(out)
        out = out.view(-1, 64, 2, 2)
        out=self.aeseqs[0](out)
        for i in range(1,len(self.aeseqs)):
            out=self.aeseqs[i](out)
        out = self.aelast(out)
        out = self.tanh(out)
        out = self.aelast2(out)
        out = self.tanh2(out)
        return out
    

class AELossFn:
    def __init__(self):
        self.criterion = nn.MSELoss()
        self.name="AE"
    def __call__(self, pred, crct):
        return self.criterion(pred, crct)

class VAELossFn:
    def __init__(self):
        self.criterion = nn.MSELoss()
        self.name="VAE"

    def __call__(self, pred, crct, mu, logvar):
        lossmse = self.criterion(pred, crct)
        kl = torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = -0.5*kl
        return lossmse + kl


def ParameterSelector(E, D):
    encoder_params = list(E.parameters())+list(E.aeseqs[0].parameters())+list(E.aeseqs[1].parameters())+list(E.aeseqs[2].parameters())
    decoder_params = list(D.parameters())+list(D.aeseqs[0].parameters())+list(D.aeseqs[1].parameters())+list(D.aeseqs[2].parameters())
    return (encoder_params) + (decoder_params)

class AETrainer:
    def __init__(self, dataloader,enc, dec, lossfun,optimizer,gpu):
        self.enc = enc
        self.dec = dec
        self.criterion = lossfun
        self.optimizer=optimizer
        self.gpu = gpu 
        self.device = torch.device("cpu")
        if self.gpu == "T":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train(dataloader)
        

    def train(self,data_loader,):
        self.enc=self.enc.to(self.device)
        self.dec=self.dec.to(self.device)
        for epoch in range(50):
            loss_total = 0.0
            total_ssim = 0.0
            self.enc.train()
            self.dec.train()
            all_logits = []
            for idxx,data in enumerate(data_loader):
                # print(idxx)
                cleans=torch.stack([all_cleans[idx][:10] for idx in data['label']],axis=0)
                images = data['aug_image'].float().to(self.device)
                target = cleans.float().to(self.device)
                self.optimizer.zero_grad()
                enc_out = self.enc.forward(images)
                pred = self.dec.forward(enc_out)
                preddd=pred.repeat(1, 10, 1, 1)
                preddd=preddd.unsqueeze(2)
                mseloss = self.criterion(preddd, target)
                ssim_val=0
                for i in range(len(target)):
                    # ssim_val += structure_similarity_index(pred[i].cpu(),target[i][0].cpu())
                    pred1=pred[i].detach().cpu().numpy()
                    tar1=target[i][0].detach().cpu().numpy()
                    ssim_val += ssim(pred1.squeeze(),tar1.squeeze(),data_range=1)
                mseloss.backward()
                self.optimizer.step()
                loss_total += (mseloss.item()/len(target))
                total_ssim += (ssim_val/len(target))
                all_logits.append(enc_out.detach().cpu())
            all_logits = torch.cat(all_logits, dim=0)
            all_logits_np = all_logits.numpy()
            if epoch%5==0:  
                tsne = TSNE(n_components=3, random_state=42)
                tsne_results = tsne.fit_transform(all_logits_np)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2])
                filename = f'AE_epoch_{epoch}.png'
                plt.savefig(filename)
                plt.close()
                torch.save(self.enc.state_dict(), "AE_enocder.pth")
                torch.save(self.dec.state_dict(), "AE_Decoder.pth")



            length=len(data_loader)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, loss_total /length,total_ssim/length))
            del cleans, images, target, enc_out, pred, preddd,all_logits,all_logits_np
            torch.cuda.empty_cache()
        torch.save(self.enc.state_dict(), "AE_enocder.pth")
        torch.save(self.dec.state_dict(), "AE_Decoder.pth")

    """
    Write code for training Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
class VAETrainer:
    def __init__(self, dataloader,enc, dec, lossfun,optimizer,gpu):
        self.enc = enc
        self.dec = dec
        self.criterion = lossfun
        self.optimizer=optimizer
        self.gpu = gpu 
        self.device = torch.device("cuda" if self.gpu == "T" else "cpu")
        self.train(dataloader)
        

    def train(self,data_loader,):
        self.enc=self.enc.to(self.device)
        self.dec=self.dec.to(self.device)
        for epoch in range(50):
            loss_total = 0.0
            total_ssim = 0.0
            self.enc.train()
            self.dec.train()
            all_logits = []
            for idxx,data in enumerate(data_loader):
                # print(idxx)
                cleans=torch.stack([all_cleans[idx][:10] for idx in data['label']],axis=0)
                images = data['aug_image'].float().to(self.device)
                target = cleans.float().to(self.device)
                self.optimizer.zero_grad()
                enc_out,mean,var = self.enc.forward1(images)
                pred = self.dec.forward1(enc_out)
                preddd=pred.repeat(1, 10, 1, 1)
                preddd=preddd.unsqueeze(2)
                mseloss = self.criterion(preddd, target,mean,var)
                ssim_val=0
                for i in range(len(target)):
                    # ssim_val += structure_similarity_index(pred[i].cpu(),target[i][0].cpu())
                    pred1=pred[i].detach().cpu().numpy()
                    tar1=target[i][0].detach().cpu().numpy()
                    ssim_val += ssim(pred1.squeeze(),tar1.squeeze(),data_range=1)
                mseloss.backward()
                self.optimizer.step()
                loss_total += (mseloss.item()/len(target))
                total_ssim += (ssim_val/len(target))
                all_logits.append(enc_out.detach().cpu())
            all_logits = torch.cat(all_logits, dim=0)
            all_logits_np = all_logits.numpy()
            if epoch%5==0:  
                tsne = TSNE(n_components=3, random_state=42)
                tsne_results = tsne.fit_transform(all_logits_np)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2])
                filename = f'VAE_epoch_{epoch}.png'
                plt.savefig(filename)
                plt.close()
                torch.save(self.enc.state_dict(), "VAE_enocder.pth")
                torch.save(self.dec.state_dict(), "VAE_Decoder.pth")



            length=len(data_loader)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, loss_total /length,total_ssim/length))
            del cleans, images, target, enc_out, pred, preddd,all_logits,all_logits_np
            torch.cuda.empty_cache()
        torch.save(self.enc.state_dict(), "VAE_enocder.pth")
        torch.save(self.dec.state_dict(), "VAE_Decoder.pth")


class AE_TRAINED:
    def __init__(self,gpu=False):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.load_state_dict(torch.load("AE_enocder.pth"))
        self.decoder.load_state_dict(torch.load("AE_Decoder.pth"))
        self.grayscale = Grayscale()
        self.device="cpu"
        if gpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def from_path(self,sample, original, type):
        image1 = torchvision.io.read_image(sample)
        image1=image1/255
        image1 = self.grayscale(image1)
        image2 = torchvision.io.read_image(original)
        image2=image2/255
        image2 = self.grayscale(image2)
        image1=image1.to(self.device)
        image2=image2.to(self.device)
        image1=image1.unsqueeze(0)
        self.encoder=self.encoder.to(self.device)
        self.decoder=self.decoder.to(self.device)
        enc_out = self.encoder.forward(image1)
        pred = self.decoder.forward(enc_out)
        if type=="SSIM":
            # ssim_val = structure_similarity_index(pred.cpu(),image2.cpu())
            pred1=pred.detach().cpu().numpy()
            tar1=image2.detach().cpu().numpy()
            ssim_val= ssim(pred1.squeeze(),tar1.squeeze(),data_range=1)
            return ssim_val
        elif type=="PSNR":
            # pred=pred.squeeze().cpu().numpy()
            # image2=image2.squeeze().cpu().numpy()
            pred=pred.squeeze(0)
            psnr = peak_signal_to_noise_ratio(pred.cpu(),image2.cpu())
            return psnr





class VAE_TRAINED:
    def __init__(self,gpu=False):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.load_state_dict(torch.load("VAE_enocder.pth"))
        self.decoder.load_state_dict(torch.load("VAE_Decoder.pth"))
        self.grayscale = Grayscale()
        self.device="cpu"
        if gpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def from_path(self,sample, original, type):
        image1 = torchvision.io.read_image(sample)
        image1=image1/255
        image1 = self.grayscale(image1)
        image2 = torchvision.io.read_image(original)
        image2=image2/255
        image2 = self.grayscale(image2)
        image1=image1.to(self.device)
        image2=image2.to(self.device)
        image1=image1.unsqueeze(0)
        self.encoder=self.encoder.to(self.device)
        self.decoder=self.decoder.to(self.device)
        enc_out = self.enc.forward1(image1)
        pred = self.dec.forward1(enc_out)
        if type=="SSIM":
            # ssim_val = structure_similarity_index(pred.cpu(),image2.cpu())
            pred1=pred.detach().cpu().numpy()
            tar1=image2.detach().cpu().numpy()
            ssim_val= ssim(pred1.squeeze(),tar1.squeeze(),data_range=1)
            return ssim_val
        elif type=="PSNR":
            pred=pred.squeeze(0)
            psnr = peak_signal_to_noise_ratio(pred.cpu(),image2.cpu())
            return psnr

class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass

class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()