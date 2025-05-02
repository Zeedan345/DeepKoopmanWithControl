from ntpath import join
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import odeint
# physics engine
import pybullet as pb
import pybullet_data
from scipy.io import loadmat, savemat
# Franka simulator
from franka_env import FrankaEnv

#data collect
def Obs(o):
    return np.concatenate((o[:3],o[7:]),axis=0)

class data_collecter():
    def __init__(self,env_name) -> None:
        self.env_name = env_name
        self.env =  FrankaEnv(render = False)
        self.Nstates = 17
        self.uval = 0.12
        self.udim = 7
        self.reset_joint_state = np.array(self.env.reset_joint_state)

    def collect_koopman_data(self,traj_num,steps):
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))
        for traj_i in range(traj_num):
            noise = (np.random.rand(7)-0.5)*2*0.2
            joint_init = self.reset_joint_state+noise
            joint_init = np.clip(joint_init,self.env.joint_low,self.env.joint_high)
            s0 = self.env.reset_state(joint_init)
            s0 = Obs(s0)
            u10 = (np.random.rand(7)-0.5)*2*self.uval
            train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                s0 = self.env.step(u10)
                s0 = Obs(s0)
                u10 = (np.random.rand(7)-0.5)*2*self.uval
                train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
        return train_data
        
#define network
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega
    
class Network(nn.Module):
    def __init__(self,encode_layers,Nkoopman,u_input_dim):
        super(Network,self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers)-1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i],encode_layers[layer_i+1])
            if layer_i != len(encode_layers)-2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Nkoopman = Nkoopman
        self.u_input_dim = u_input_dim
        self.lA = nn.Linear(Nkoopman,Nkoopman,bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_input_dim,Nkoopman,bias=False)

    def encode(self,x):
        return torch.cat([x,self.encode_net(x)],axis=-1)
    
    def forward(self,x,u):
        return self.lA(x)+self.lB(u)
    

class VAENetwork(nn.Module):
    def __init__(self, state_dim, u_dim, u_out_dim, hidden_dim=128):
        super(VAENetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + u_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, u_out_dim)
        self.fc_logvar = nn.Linear(hidden_dim, u_out_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + u_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, u_dim)
        )
        # self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        # self.B = nn.Linear(u_dim, latent_dim, bias=False)

        # self.A.weight.data = gaussian_init_(latent_dim, std=1)
        # U, _, V = torch.svd(self.A.weight.data)
        # self.A.weight.data = torch.mm(U, V.t()) * 0.9

    def encode(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        h = self.encoder(xu)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, u_hat):
        xu_hat = torch.cat([x, u_hat], dim=-1)
        return self.decoder(xu_hat)

    def forward(self, x, u):
        mu, logvar = self.encode(x, u)
        u_hat = self.reparameterize(mu, logvar)
        u_recon = self.decode(x, u_hat)
        return u_recon, mu, logvar, u_hat
    

#loss function
def Klinear_loss(data, net, control_net, mse_loss, u_dim=1, gamma=0.99, Nstate=4, all_loss=0, vae_weight=1.0, kl_weight=1.0):
    steps, train_traj_num, _ = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_current = net.encode(data[0, :, u_dim:])  # Initial lifted state z0
    beta = 1.0
    beta_sum = 0.0
    koopman_loss = torch.zeros(1, dtype=torch.float64).to(device)
    vae_loss = torch.zeros(1, dtype=torch.float64).to(device)
    
    for i in range(steps - 1):
        s_i = data[i, :, u_dim:]
        u_i = data[i, :, :u_dim]
        u_recon_i, mu_i, logvar_i, u_hat_i = control_net(s_i, u_i)
        step_vae, _, _ = Control_VAE_Loss(u_i, u_recon_i, mu_i, logvar_i, kl_weight)
        vae_loss += beta * step_vae
        X_current = net.forward(X_current, u_hat_i)
        if not all_loss:
            koopman_loss += beta * mse_loss(X_current[:, :Nstate], data[i + 1, :, u_dim:])
        else:
            Y = net.encode(data[i + 1, :, u_dim:])
            koopman_loss += beta * mse_loss(X_current, Y)
        beta_sum += beta
        beta *= gamma
    
    koopman_loss = koopman_loss / beta_sum
    vae_loss = vae_loss / beta_sum
    total_loss = koopman_loss + vae_weight * vae_loss
    return total_loss, koopman_loss, vae_loss


def Control_VAE_Loss(u, u_recon, mu, logvar, kl_weight = 1.0):
    recon_loss = F.mse_loss(u_recon, u, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    total_vae_loss = recon_loss + kl_weight * kl_loss
    return total_vae_loss, recon_loss, kl_loss

def Stable_loss(net,Nstate):
    x_ref = np.zeros(Nstate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_ref_lift = net.encode_only(torch.DoubleTensor(x_ref).to(device))
    loss = torch.norm(x_ref_lift)
    return loss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs()-torch.ones(1,dtype=torch.float64).to(device)
    mask = c>0
    loss = c[mask].sum()
    return loss

def train(env_name,train_steps = 300000,suffix="",all_loss=0,\
            encode_dim = 20,layer_depth=3,e_loss=1,gamma=0.5):
    np.random.seed(98)
    # Ktrain_samples = 1000
    # Ktest_samples = 1000
    Ktrain_samples = 50000
    Ktest_samples = 20000
    Ksteps = 10
    Kbatch_size = 512
    u_dim = 7
    #data prepare
    data_collect = data_collecter(env_name)
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Ksteps)
    print("test data ok!")
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples,Ksteps)
    print("train data ok!")
    # savemat('FrankaTrainingData.mat',{'Train_data':Ktrain_data,'Test_data':Ktest_data})
    # raise NotImplementedError
    in_dim = Ktest_data.shape[-1]-u_dim
    Nstate = in_dim
    layer_width = 128
    layers = [in_dim]+[layer_width]*layer_depth+[encode_dim]
    Nkoopman = in_dim+encode_dim
    u_out_dim = u_dim
    print("layers:",layers)
    net = Network(layers,Nkoopman,encode_dim)
    control_net = VAENetwork(in_dim, u_dim, encode_dim)
    # print(net.named_modules())
    eval_step = 1000
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda() 
        control_net.cuda()
    net.double()
    control_net.double()
    mse_loss = nn.MSELoss()
    all_params = list(net.parameters()) + list(control_net.parameters())
    optimizer = torch.optim.Adam(all_params,
                                    lr=learning_rate)
    # for name, param in all_params:
    #     print("model:",name,param.requires_grad)
    #train
    eval_step = 1000
    best_loss = 1000.0
    kl_weight = 0.01
    vae_weight = 1.0
    
    best_state_dict = {}
    subsuffix = suffix+"KK_"+env_name+"layer{}_edim{}_eloss{}_gamma{}_aloss{}".format(layer_depth,encode_dim,e_loss,gamma,all_loss)
    logdir = "Data/"+suffix+"/"+subsuffix
    if not os.path.exists( "Data/"+suffix):
        os.makedirs( "Data/"+suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    for i in range(train_steps):
        #K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:,Kindex[:Kbatch_size],:]

        total_loss, Kloss, vae_loss = Klinear_loss(X, net, control_net, mse_loss, u_dim, gamma, Nstate, all_loss, vae_weight, kl_weight)
        Eloss = Eig_loss(net)
        
        loss = total_loss+Eloss if e_loss else total_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        writer.add_scalar('Train/Kloss',Kloss,i)
        writer.add_scalar('Train/VAEloss',vae_loss,i)
        writer.add_scalar('Train/Eloss',Eloss,i)
        writer.add_scalar('Train/loss',loss,i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i+1) % eval_step ==0:
            #K loss
            total_loss, Kloss, vae_loss = Klinear_loss(X, net, control_net, mse_loss, u_dim, gamma, Nstate, all_loss, vae_weight, kl_weight)
            Eloss = Eig_loss(net) 
            loss = total_loss+Eloss if e_loss else total_loss
            Kloss = Kloss.detach().cpu().numpy()
            vae_loss = vae_loss.detach().cpu().numpy()
            Eloss = Eloss.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()
            writer.add_scalar('Eval/Kloss',Kloss,i)
            writer.add_scalar('Eval/VAEloss',vae_loss,i)
            writer.add_scalar('Eval/Eloss',Eloss,i)
            writer.add_scalar('Eval/loss',loss,i)
            if loss<best_loss:
                best_loss = loss
                saved_dict = {
                    'net_model': net.state_dict(),
                    'control_net_model': control_net.state_dict(),
                    'layer': layers
                }
                torch.save(saved_dict,"Data/"+subsuffix+".pth")
            print("Step:{} Total-Eval-loss{} K-loss:{} E-loss:{} VAE-Loss{}".format(i,loss,Kloss,Eloss, vae_loss))
            # print("-------------END-------------")
    print("END-best_loss{}".format(best_loss))
    

def main():
    train(args.env,suffix=args.suffix,all_loss=args.all_loss,\
        encode_dim=args.encode_dim,layer_depth=args.layer_depth,\
            e_loss=args.eloss,gamma=args.gamma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="Franka")
    parser.add_argument("--suffix",type=str,default="")
    parser.add_argument("--all_loss",type=int,default=1)
    parser.add_argument("--eloss",type=int,default=0)
    parser.add_argument("--gamma",type=float,default=0.8)
    parser.add_argument("--encode_dim",type=int,default=20)
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()

