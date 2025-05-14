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
        Kp = 2.0
        Kd = 0.1
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))
        for traj_i in range(traj_num):
            q_goal = np.random.uniform(self.env.joint_low, self.env.joint_high)
            
            noise = (np.random.rand(7)-0.5)*2*0.2

            joint_init = self.reset_joint_state+noise
            joint_init = np.clip(joint_init,self.env.joint_low,self.env.joint_high)

            s0 = self.env.reset_state(joint_init)
            s0 = Obs(s0)
            
            #u10 = (np.random.rand(7)-0.5)*2*self.uval
            u10 = np.zeros(self.udim)
            train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                full_s = self.env.get_state()
                q = full_s[6:13]
                qd = full_s[13:20]

                u10 = Kp * (q_goal - q) - Kd * qd
                u10 = np.clip(u10, -self.env.sat_val, self.env.sat_val)

                s0 = self.env.step(u10)
                s0 = Obs(s0)
                #u10 = (np.random.rand(7)-0.5)*2*self.uval
                train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
        return train_data
        
#define network
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega
    
class Network(nn.Module):
    def __init__(self,encode_layers,bilinear_layers,Nkoopman,u_dim, encode_dim):
        super(Network,self).__init__()
        #First We need the VAE encoder for state
        ELayers = OrderedDict()
        for layer_i in range(len(encode_layers)-1):
            ELayers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i],encode_layers[layer_i+1])
            if layer_i != len(encode_layers)-2:
                ELayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(ELayers)
        #self.state_fc_mu = nn.Linear(encode_layers[-1], encode_dim)
        #self.state_fc_logvar = nn.Linear(encode_layers[-1], encode_dim)
        #VAE decoder for state
        DLayers = OrderedDict()
        dims = [encode_dim] + encode_layers[1:] + [encode_layers[0]]
        for i in range(len(dims)-1):
            DLayers[f"linear_{i}"] = nn.Linear(dims[i], dims[i+1])
            if i < len(dims)-2:
                DLayers[f"relu_{i}"] = nn.ReLU()
        self.decode_net = nn.Sequential(DLayers)


        BELayers = OrderedDict()
        for layer_i in range(len(bilinear_layers)-1):
            BELayers["linear_{}".format(layer_i)] = nn.Linear(bilinear_layers[layer_i],bilinear_layers[layer_i+1])
            if layer_i != len(bilinear_layers)-2:
                BELayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.bilinear_net = nn.Sequential(BELayers)  
        self.control_fc_mu = nn.Linear(bilinear_layers[-1], u_dim)
        self.control_fc_logvar = nn.Linear(bilinear_layers[-1], u_dim)

        BDLayers = OrderedDict()
        uddims = bilinear_layers + [u_dim]
        for layer_i in range(len(uddims)-1):
            BDLayers["linear_{}".format(layer_i)] = nn.Linear(uddims[layer_i],uddims[layer_i+1])
            if layer_i != len(bilinear_layers)-2:
                BDLayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.control_decode_net = nn.Sequential(BDLayers)

        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman,Nkoopman,bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim,Nkoopman,bias=False)

    def encode_state(self,x):
        #h = self.encode_net(x)
        #mu = self.state_fc_mu(h)
        #logvar = self.state_fc_logvar(h)
        #z = self.reparameterize(mu, logvar)
        # encoded_state = torch.cat([x, z],axis=-1)
        # return encoded_state
        return torch.cat([x,self.encode_net(x)],axis=-1)
    
    def decode_state(self,encoded_state):
        return self.decode_net(encoded_state)
    
    def decode_control(self, x, u_hat):
        return self.control_decode_net(torch.cat([x, u_hat], axis = -1))
    
    def bicode(self,x,u):
        x_all = torch.cat([x,u],axis=-1)
        h = self.bilinear_net(x_all)
        mu = self.control_fc_mu(h)
        logvar = self.control_fc_logvar(h)
        encoded_control = self.reparameterize(mu, logvar)
        return encoded_control, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def forward(self,x,b):
        return self.lA(x)+self.lB(b)
    
def Klinear_loss(data,net,mse_loss,u_dim=1,gamma=0.99,Nstate=4,all_loss=0,detach=0):
    steps,train_traj_num,NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_current = net.encode_state(data[0,:,u_dim:])
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    Augloss = torch.zeros(1,dtype=torch.float64).to(device)
    #vae_loss = torch.zeros(1, dtype=torch.float64, device=device)
    control_vaeloss = torch.zeros(1, dtype=torch.float64, device=device)
    for i in range(steps-1):
        bilinear, u_mu, u_logvar = net.bicode(X_current[:,:Nstate],data[i,:,:u_dim])
        X_current = net.forward(X_current,bilinear)
        beta_sum += beta
        if not all_loss:
            loss += beta*mse_loss(X_current[:,:Nstate],data[i+1,:,u_dim:])
        else:
            Y = net.encode_state(data[i+1,:,u_dim:])
            loss += beta*mse_loss(X_current,Y)
        X_current_encoded = net.encode_state(X_current[:,:Nstate])
        #x_step_loss = State_VAE_Loss(data[i+1, :, u_dim:], net.decode_state(z_next), x_mu_next, x_logvar_next)
        #vae_loss += beta*x_step_loss
        u_step_loss, _, _ = Control_VAE_Loss(data[i,:,:u_dim], net.decode_control(data[i,:,u_dim:], bilinear), u_mu, u_logvar)
        control_vaeloss += beta*u_step_loss
        Augloss += mse_loss(X_current_encoded,X_current)
        beta *= gamma
    loss = loss/beta_sum
    Augloss = Augloss/beta_sum
    #vae_loss = vae_loss/beta_sum
    control_vaeloss = control_vaeloss/beta_sum
    return loss+0.5*Augloss + 0.4*control_vaeloss, Augloss, control_vaeloss


def State_VAE_Loss(u, u_recon, mu, logvar, kl_weight = 1.0):
    #recon_loss = F.mse_loss(u_recon, u, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    total_vae_loss = kl_weight * kl_loss
    #total_vae_loss = recon_loss + kl_weight * kl_loss
    return total_vae_loss

def Control_VAE_Loss(u, u_recon, mu, logvar, kl_weight = 0.4):
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
    blayers = [in_dim+u_dim]+[layer_width]*layer_depth
    Nkoopman = in_dim+encode_dim
    u_out_dim = u_dim
    print("layers:",layers)
    net = Network(layers,blayers, Nkoopman,u_dim, encode_dim)
    #control_net = VAENetwork(in_dim, u_dim, encode_dim)
    # print(net.named_modules())
    eval_step = 1000
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda() 
        #control_net.cuda()
    net.double()
    #control_net.double()
    mse_loss = nn.MSELoss()
    #all_params = list(net.parameters()) + list(control_net.parameters())
    all_params = list(net.parameters())
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
        Kloss, augloss, control_loss = Klinear_loss(X,net,mse_loss,u_dim,gamma,Nstate,all_loss)
        Eloss = Eig_loss(net)
        loss = Kloss+Eloss if e_loss else Kloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        writer.add_scalar('Train/Kloss',Kloss,i)
        writer.add_scalar('Train/Eloss',Eloss,i)
        # writer.add_scalar('Train/Dloss',Dloss,i)
        writer.add_scalar('Train/loss',loss,i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i+1) % eval_step ==0:
            #K loss
            with torch.no_grad():
                Kloss, augloss, control_loss = Klinear_loss(Ktest_data,net,mse_loss,u_dim,gamma,Nstate,all_loss=0)
                Eloss = Eig_loss(net)
                loss = Kloss
                Kloss = Kloss.detach().cpu().numpy()
                Eloss = Eloss.detach().cpu().numpy()
                augloss = augloss.detach().cpu().numpy()
                #vae_loss = vae_loss.detach().cpu().numpy()
                control_loss = control_loss.detach().cpu().numpy()
                # Dloss = Dloss.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy()
                writer.add_scalar('Eval/Kloss',Kloss,i)
                writer.add_scalar('Eval/Eloss',Eloss,i)
                writer.add_scalar('Eval/best_loss',best_loss,i)
                writer.add_scalar('Eval/loss',loss,i)
                if loss<best_loss:
                    best_loss = copy(Kloss)
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {'model':best_state_dict,'layer':layers,'blayer':blayers}
                    torch.save(Saved_dict,logdir+".pth")
                print("Step:{} Eval-loss{} K-loss:{} Augloss{} Control_Loss {}".format(i,loss,Kloss, augloss, control_loss))
                # print("-------------END-------------")
        writer.add_scalar('Eval/best_loss',best_loss,i)
        # if (time.process_time()-start_time)>=210*3600:
        #     print("time out!:{}".format(time.clock()-start_time))
        #     break
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

