import torch
from torch import nn
from torch.nn import functional as F

def initialize_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Xavier initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)  # Xavier initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Encoder(nn.Module):
    def __init__(self, gene_num, latent_size, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(gene_num, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_size)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_size)
        # Apply Xavier initialization
        self.apply(initialize_weights_xavier)  

        
    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar
    

class Decoder(nn.Module):
    def __init__(self, latent_size, gene_num, hidden_dim=128):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, gene_num)
        # Apply Xavier initialization
        self.apply(initialize_weights_xavier)  
        
    def forward(self, z):
        h = F.leaky_relu(self.bn1(self.fc1(z)))
        reconstructed_x = self.fc2(h)  
        return reconstructed_x



class Primary_tumor_DNN(nn.Module):
    def __init__(self, num_latent, num_classes):
        super(Primary_tumor_DNN, self).__init__()
        self.fc1 = nn.Linear(num_latent, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        # self.bn2 = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        #  x = self.fc3(x)  
        return self.softmax(x)


class Metas_site_DNN(nn.Module):
    def __init__(self, num_latent, num_classes):
        super(Metas_site_DNN, self).__init__()
        self.fc1 = nn.Linear(num_latent, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        # self.bn2 = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        #  x = self.fc3(x)  
        return self.softmax(x)
    
class PreMet(nn.Module):
    def __init__(self, gene_num, latent_size, num_tumors, num_metas_sites, hidden_size):
        super(PreMet, self).__init__()
        self.encoder = Encoder(gene_num, latent_size, hidden_dim=hidden_size)
        self.decoder = Decoder(latent_size, gene_num, hidden_dim=hidden_size)
        self.tissue_nn = Primary_tumor_DNN(latent_size, num_tumors)
        self.site_nn = Metas_site_DNN(latent_size, num_metas_sites)
        
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decoder(z)
        
        # predict tumor type from latent
        tissue_output = self.tissue_nn(z)
        
        # predict metastasis site from latent
        site_output = self.site_nn(z)
        
        return reconstructed_x, mean, logvar, tissue_output, site_output 