import torch
import os
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from torchvision.transforms import Grayscale
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim
all_cleans=[]
def calc_euclidean_distance(tensor_a, tensor_b):
    difference = tensor_a - tensor_b
    squared_difference = difference ** 2
    squared_distance_sum = torch.sum(squared_difference)
    distance = torch.sqrt(squared_distance_sum)
    
    return distance

class AlteredMNIST(Dataset):
    def __init__(self,):
        self.transform = ToTensor()
        self.grayscale = Grayscale()


        dict1 = {}
        dict2 = {}
        self.aug_labelwise = dict1
        self.clean_labelwise = dict2
        for i in range(0,10):
            list1 = []
            list2 = []
            self.aug_labelwise[i] = list1
            self.clean_labelwise[i] = list2
        self.aug_list=os.listdir("Data/aug")
        for a_img in self.aug_list:
            file_parts = a_img.split('_')  
            last_part = file_parts[-1]  
            number_str = last_part.split('.')[0]  
            label = int(number_str)  
            clean_image_path = "Data/aug/" + a_img

            self.aug_labelwise[label].append(clean_image_path)
        self.clean_list=os.listdir("Data/clean")
        for c_img in self.clean_list:
            file_parts = c_img.split('_')  
            last_part = file_parts[-1]  
            number_str = last_part.split('.')[0]  
            label = int(number_str)  
            clean_image_path = "Data/clean/" + c_img
            self.clean_labelwise[label].append(clean_image_path)
        
        self.train_pairs=[]
        new_train_p = []
        for label, a_i in self.aug_labelwise.items():
            images=[]
            image_paths=a_i
            for image_path in image_paths:
                f = torchvision.io.read_image(image_path)
                d=len(new_train_p)
                image = f
                image = image.float()
                image = image/255.0
                new_train_p.append(image)
                image = self.grayscale(image)
                images.append(image)
            combined_images_tensor = torch.stack(images)
            images_tensor = combined_images_tensor.view(len(images), -1)
            distances_matrix = torch.zeros(len(images), len(images))
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    distances_matrix[i, j] = calc_euclidean_distance(images_tensor[i], images_tensor[j])
                    distances_matrix[j, i] = calc_euclidean_distance(images_tensor[i], images_tensor[j])
            distances_matrix_new=distances_matrix.tril()
            distance_dict = {}
            v = 0
            for i in range(0,len(images)):
                for j in range(0,i):
                    v+=1
                    # print(v)
                    distance = distances_matrix_new[i, j]
                    distance_dict[distance.item()] = (i, j)
            sorted_distances = sorted(distance_dict.keys())
            new_paths = set()
            new_x = []
            for distance in sorted_distances[:700]:
                row, col = distance_dict[distance]
                i_p = image_paths[row]
                new_paths.add(i_p)
                i_p_c = image_paths[col]
                new_x.append(i_p)
                new_paths.add(image_paths[col])
            new_paths=list(new_paths)
            for i in (new_paths):
                self.train_pairs.append((i,label))
            
        self.average_s = set()
        self.average_intensities=[]
        c = 0
        for i in range(10):
            img_paths = self.clean_labelwise[i]
            imagess=[]
            c+=1
            for path in img_paths:
                img = torchvision.io.read_image(path)
                img = img.float()
                img = img/255.0
                imagess.append(img)
                if(c%2==0):
                  c+=1
            tensor_images = torch.stack(imagess)
            mean_intensities = torch.mean(tensor_images, dim=0)
            self.average_intensities.append(mean_intensities)
        global all_cleans
        self.labels=[]
        len_train_pairs = len(self.train_pairs)
        for i in range(0,len_train_pairs):
            aimg_ = self.train_pairs[i][0]
            image = torchvision.io.read_image(aimg_)
            image = image.float()
            image = image/255.0
            image = self.grayscale(image)
            distances = []
            for avg_img in self.average_intensities:
              distance = calc_euclidean_distance(avg_img, image)
              distances.append(calc_euclidean_distance(avg_img, image))
            highest_dissimilarity_idx = torch.argmax(torch.tensor(distances))
            h_d_i = highest_dissimilarity_idx.numpy()
            highest_dissimilarity_idx =int(h_d_i)
            self.labels.append(highest_dissimilarity_idx)
        
        self.set_all_c = set()
        self.all_cleans=[]
        
        for i in range(0,10):
            label=i
            tot_ims=[]
            for i in range(0,100):
                pth = self.clean_labelwise[label][i]
                cimage = torchvision.io.read_image(pth)
                cimage = cimage.float()
                cimage = cimage/255.0
                cimage = self.grayscale(cimage)
                tot_ims.append(cimage)
            tot_ims=torch.stack(tot_ims, axis=0)
            self.all_cleans.append(tot_ims)
        all_cleans=self.all_cleans

        


    def __len__(self):
        x = len(self.train_pairs)
        return x

    def __getitem__(self, id):
        image = torchvision.io.read_image(self.train_pairs[id][0])
        new_image = image.float()
        new_image = new_image/255.0
        image = new_image
        image = self.grayscale(image)
        label=self.labels[id]
        return {'aug_image': image,'label':label}


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(1, 8, 3, 1, 1)
        self.BN = nn.BatchNorm2d(8)
        self.init_conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(16)
        self.init_conv3 = nn.Conv2d(1, 16, 3, 1, 1)
        self.BN3 = nn.BatchNorm2d(16)
        self.rb1 = ResBlock(16, 16, 3, 2, 1, 'encode')
        self.rb2 = ResBlock(16, 32, 3, 1, 1, 'encode')
        self.rb3 = ResBlock(32, 32, 3, 2, 1, 'encode')
        self.rb4 = ResBlock(32, 48, 3, 1, 1, 'encode')
        self.rb5 = ResBlock(48, 48, 3, 2, 1, 'encode')
        self.rb6 = ResBlock(48, 64, 3, 2, 1, 'encode')
        self.relu = nn.ReLU()

        self.encode = nn.Sequential(
            
            
            nn.ReLU(),
            nn.Linear(64*2*2, 196),
            nn.ReLU(),
            nn.Linear(196, 48),
            nn.ReLU()
        )

        # self.encode2 = nn.Sequential(
            
            
        #     nn.ReLU(),
        #     nn.Linear(64*2*2, 196),
        #     nn.ReLU(),
        #     nn.Linear(196, 48),
        #     nn.ReLU()
        # )
        self.mean_fc = nn.Sequential(
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.log_var_fc=nn.Sequential(
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, inputs):
        # print("shape in forward: ", inputs.shape)
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        init_conv2 = self.relu(self.BN2(self.init_conv2(init_conv)))
        rb1 = self.rb1(init_conv2)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        out = rb6
        # out = out.squeeze(0)
        
        out = out.view(out.size(0), -1)
        # print(out.shape) # 32*256
        out = self.encode(out)
        return out
    def forward2(self, inputs):
        # print("shape in forward: ", inputs.shape)
        init_conv = F.relu(self.BN3(self.init_conv3(inputs)))
        out = self.rb1(init_conv)
        out = self.rb2(out)
        out = self.rb3(out)
        out = self.rb4(out)
        out = self.rb5(out)
        out = self.rb6(out)
        # print(out.shape)
        out = out.squeeze(0)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.encode(out)
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)
        std = torch.exp(0.5*log_var)
        z = torch.randn_like(std)
        z = z*std + mean
        # print(z.shape)
        return [z, mean, log_var]

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            
            nn.Linear(48, 196),
            nn.ReLU(),
            nn.Linear(196, 64*2*2),
            nn.ReLU()
        )

        self.rb1 = ResBlock(64, 48, 2, 2, 0, 'decode')
        self.rb2 = ResBlock(48, 48, 2, 2, 0, 'decode')
        self.rb3 = ResBlock(48, 32, 3, 1, 1, 'decode')
        self.rb4 = ResBlock(32, 32, 2, 2, 0, 'decode')
        self.rb5 = ResBlock(32, 16, 3, 1, 1, 'decode')
        self.rb6 = ResBlock(16, 16, 13, 1, 0, 'decode')
        self.out_conv1 = nn.ConvTranspose2d(16, 8, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.out_conv2 = nn.ConvTranspose2d(8, 1, 3, 1, 1)
        self.tanh = nn.Tanh()

        self.decode2 = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 48),
            nn.ReLU(),
            nn.Linear(48, 196),
            nn.ReLU(),
            nn.Linear(196, 64*2*2),
            nn.ReLU()
        )
        self.decode_old = nn.Sequential(
            ResBlock(64, 48, 2, 2, 0, 'decode'),
            ResBlock(48, 48, 2, 2, 0, 'decode'),
            ResBlock(48, 32, 3, 1, 1, 'decode'),
            ResBlock(32, 32, 2, 2, 0, 'decode'),
            ResBlock(32, 16, 3, 1, 1, 'decode'),
            ResBlock(16, 16, 13, 1, 0, 'decode')
        )
        self.out_conv = nn.ConvTranspose2d(16, 1, 3, 1, 1)
        self.tanh = nn.Tanh()
    def forward(self, inputs):
        # print("decoder input shape: ",inputs.shape)

        # out = out.view(-1, 64, 2, 2)
        out = self.decode(inputs)
        out = out.view(-1, 64, 2, 2)
        rb1 = self.rb1(out)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        out_conv1 = self.out_conv1(rb6)
        output = self.tanh(out_conv1)
        output = self.out_conv2(output)
        output = self.tanh(output)
        # print("Output in decoder", output.shape)
        return output
    
    def forward2(self, inputs):
        # print("decoder entry: ",inputs.shape)
        out = self.decode2(inputs)
        out = out.view(-1, 64, 2, 2)
        # print("after decode: ", out.shape)
        out = self.decode_old(out)
        out_conv = self.out_conv(out)
        output = self.tanh(out_conv)
        return output 
    

class AELossFn:
    def __init__(self):
        self.criterion = nn.MSELoss()

    def __call__(self, output, target):
        return self.criterion(output, target)

class VAELossFn:
    def __init__(self):
        self.criterion = nn.MSELoss()

    def __call__(self, output, target, mu, logvar):
        mse_loss = self.criterion(output, target)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse_loss + kl_divergence

def ParameterSelector(E, D):
    encoder_params = list(E.init_conv3.parameters()) + \
                    list(E.rb1.parameters()) + \
                    list(E.rb2.parameters()) + \
                    list(E.rb3.parameters()) + \
                    list(E.rb4.parameters()) + \
                    list(E.rb5.parameters()) + \
                    list(E.rb6.parameters()) + \
                    list(E.encode.parameters()) + \
                    list(E.mean_fc.parameters()) + \
                    list(E.log_var_fc.parameters())

    decoder_params = list(D.decode.parameters()) + \
                    list(D.decode_old.parameters()) + \
                    list(D.out_conv.parameters())
    return list(E.init_conv.parameters()) + list(E.init_conv2.parameters()) + list(E.rb1.parameters()) + list(E.rb2.parameters()) + \
           list(E.rb3.parameters()) + list(E.rb4.parameters()) + list(E.rb5.parameters()) + \
           list(E.rb6.parameters()) + list(D.rb1.parameters()) + list(D.rb2.parameters()) + \
           list(D.rb3.parameters()) + list(D.rb4.parameters()) + list(D.rb5.parameters()) + \
           list(D.rb6.parameters()) + list(D.out_conv.parameters())+list(D.out_conv2.parameters())+list(E.encode.parameters())+list(D.decode.parameters()) + encoder_params + decoder_params



class AETrainer:
    
    def __init__(self, train_loader, encoder, decoder, criterion, optimizer, gpu):
        self.optimizer = optimizer
        self.loader = train_loader
        self.criterion = criterion
        self.gpu = gpu 
        self.device = torch.device("cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.train(self.loader, self.encoder, self.decoder, self.criterion, self.optimizer, self.gpu)

    def save_models(self, epoch):
        encoder_path = f'AE_encoder_epoch_{epoch}.pth'
        decoder_path = f'AE_decoder_epoch_{epoch}.pth'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        

    def train(self, train_loader, encoder, decoder, criterion, optimizer, gpu):
        # self.criterion = AELossFn()
        # all_logits = []
        for epoch in range(10):
            total_loss = 0.0
            total_ssim = 0.0
            all_logits = []
            for minibatch, data in enumerate(train_loader, 1):
                selected_tensors = []
                for idx in data['label']:
                    selected_tensor = all_cleans[idx]
                    sliced_tensor = selected_tensor[:10]
                    selected_tensors.append(sliced_tensor)
                stacked_tensors = torch.stack(selected_tensors, axis=0)
                cleans=stacked_tensors
                img1 = data['aug_image']
                img1 = img1/255.0
                img2 = img1.to(self.device)
                images = img2
                # label = data['clean_image'].to(self.device)
            
                target = cleans.to(self.device)
                # target = cleans
                optimizer.zero_grad()
                
                encoded = self.encoder.forward(images)
                # print("target shape: ", encoded.shape)
                all_logits.append(encoded.detach().cpu())
                outputs = self.decoder.forward(encoded)
                output_new=outputs.repeat(1, 10, 1, 1)
                output_new = output_new.unsqueeze(2)
                # print("target shape: ", target.shape)
                # print("output shape: ", output_new.shape)
                # Compute loss
                loss = self.criterion(output_new, target)
                # print("calculated Loss")
                ssim_val=0
                print("----------------->",  len(target))
                for i in range(len(target)):
                    decoded_np = outputs[i].detach()
                    decoded_np = decoded_np.cpu()
                    decoded_np = decoded_np.numpy()
                    images_np = target[i][0].detach()
                    images_np = images_np.cpu()
                    images_np = images_np.numpy()
                    images_np = images_np/255.0
                    ssim_value = ssim(images_np.squeeze(), decoded_np.squeeze(), data_range = 1.0)
                    if ssim_value is not None and not math.isnan(ssim_value):
                        ssim_val += ssim(images_np.squeeze(), decoded_np.squeeze(), data_range = 1.0) 
                    else:
                        print(images_np)
                        print("-----")
                        print(decoded_np)
                        ssim_val+=0
                    # if(i%10==0):
                    #     print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch%10, loss.item(), ssim_val))
                total_loss += loss.item()
                total_ssim += (ssim_val.item()/len(target))
                loss.backward()
                optimizer.step()
            
            if epoch == 49:  
                self.save_models(epoch)
            # print(len(train_loader))
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, total_loss / len(train_loader),total_ssim/len(train_loader)))

            
            all_logits = torch.cat(all_logits, dim=0)
            
            all_logits_np = all_logits.numpy()
            
            tsne = TSNE(n_components=3, random_state=42)
            tsne_results = tsne.fit_transform(all_logits_np)
            
            tsne_results_tensor = torch.tensor(tsne_results)
        
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_results_tensor[:, 0], tsne_results_tensor[:, 1], tsne_results_tensor[:, 2])
            if(epoch%5==0):
                filename = f'AE_epoch_{epoch}.png'
                plt.savefig(filename)
                plt.close()


class VAETrainer:
    def __init__(self, train_loader, encoder, decoder, criterion, optimizer, gpu):
        self.optimizer = optimizer
        self.loader = train_loader
        self.criterion = criterion
        self.gpu = gpu 
        self.device = torch.device("cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.train(self.loader, self.encoder, self.decoder, self.criterion, self.optimizer, self.gpu)
        
    def save_models(self, epoch):
        encoder_path = f'VAE_encoder_epoch_{epoch}.pth'
        decoder_path = f'VAE_decoder_epoch_{epoch}.pth'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def train(self, train_loader, encoder, decoder, criterion, optimizer, gpu):
        for epoch in range(10):
            total_loss = 0.0
            total_ssim = 0.0
            all_logits = []
            for minibatch, data in enumerate(train_loader, 1):
                selected_tensors = []
                for idx in data['label']:
                    selected_tensor = all_cleans[idx]
                    sliced_tensor = selected_tensor[:10]
                    selected_tensors.append(sliced_tensor)
                stacked_tensors = torch.stack(selected_tensors, axis=0)
                cleans=stacked_tensors
                aug_image = data['aug_image']
                aug_image = aug_image/255.0
                aug_image = aug_image.to(self.device)  
                # clean_image = data['clean_image'].to(self.device) 
                # print(images.shape)
                target = cleans.to(self.device)
                optimizer.zero_grad()
                
                # target = empty_tensor_images
                # print("target shape: ", aug_imag.shape)
                encoded, mean, log_var= self.encoder.forward2(aug_image)
                # print("target shape: ", encoded.shape)
                all_logits.append(encoded.detach().cpu())
                outputs = self.decoder.forward2(encoded)
                output_new=outputs.repeat(1, 10, 1, 1)
                output_new = output_new.unsqueeze(2)
                
                loss = self.criterion(output_new, target, mean, log_var)
                ssim_val=0
                for i in range(len(target)):
                    decoded_np = outputs[i].detach()
                    decoded_np = decoded_np.cpu()
                    decoded_np = decoded_np.numpy()
                    images_np = target[i][0].detach()
                    images_np = images_np.cpu()
                    images_np = images_np.numpy()
                    images_np = images_np/255.0
                    ssim_value = ssim(images_np.squeeze(), decoded_np.squeeze(), data_range = 1.0)
                    if ssim_value is not None and not math.isnan(ssim_value):
                        ssim_val += ssim(images_np.squeeze(), decoded_np.squeeze(), data_range = 1.0)
                        
                    else:
                        ssim_val+=0
                    # if(i%10==0):
                    #     print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch%10, loss.item(), ssim_val))
                
                total_loss += loss.item()
                total_ssim += (ssim_val.item()/len(target))
                loss.backward()
                optimizer.step()
            if epoch == 49:  
                self.save_models(epoch)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, total_loss / len(train_loader),total_ssim/len(train_loader)))
            all_logits = torch.cat(all_logits, dim=0)
            
            all_logits_np = all_logits.numpy()
            
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(all_logits_np)
            
            tsne_results_tensor = torch.tensor(tsne_results)
        
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_results_tensor[:, 0], tsne_results_tensor[:, 1])
            if(epoch%5==0):
                filename = f'VAE_epoch_{epoch}.png'
                plt.savefig(filename)
                plt.close()
            

class AE_TRAINED:
    def __init__(self, gpu=False):
       
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.grey = Grayscale()
        encoder_path = 'AE_encoder_epoch_49.pth'
        decoder_path = 'AE_decoder_epoch_49.pth'
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        
    def load_model(self, model, path):
        
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()

    def from_path(self, sample, original, type):
        sample = torchvision.io.read_image(sample)
        sample = sample/255.0
        sample = self.grey(sample)
        original = torchvision.io.read_image(original)
        original = original/255.0
        original = self.grey(original)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        encoded = self.encoder.forward(sample)
        decoded = self.decoder.forward(encoded)
        dc = decoded.detach().cpu().numpy()
        ori = original.detach().cpu().numpy()
        if type=="SSIM":
            return ssim(dc,ori, data_range = 255.0)
        
        return peak_signal_to_noise_ratio(dc, ori)

            

class VAE_TRAINED:
    def __init__(self, gpu=False):
       
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.grey = Grayscale()
        encoder_path = 'VAE_encoder_epoch_49.pth'
        decoder_path = 'VAE_decoder_epoch_49.pth'
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        
    def load_model(self, model, path):
        
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()

    def from_path(self, sample, original, type):
        sample = torchvision.io.read_image(sample)
        sample = sample/255.0
        sample = self.grey(sample)
        original = torchvision.io.read_image(original)
        original = original/255.0
        original = self.grey(original)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        encoded = self.encoder.forward2(sample)
        decoded = self.decoder.forward2(encoded)
        dc = decoded.detach().cpu().numpy()
        ori = original.detach().cpu().numpy()
        if type=="SSIM":
            return ssim(dc,ori, data_range = 255.0)
        
        return peak_signal_to_noise_ratio(dc, ori)

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