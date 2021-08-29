# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import json
from tqdm import tqdm, trange

from layers import Summarizer, Generator, Discriminator, apply_weight_norm
from utils import TensorboardWriter
from feature_extraction import ResNetFeature


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):

        # Build Modules
        self.backbone = ResNetFeature()
        self.summarizer = Summarizer(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.generator = Generator(
            self.config.input_size).cuda()
        self.discriminator = Discriminator().cuda()
        self.model = nn.ModuleList([
            self.summarizer, self.generator, self.discriminator])

        if self.config.mode == 'train':
            # Build Optimizers
            self.s_e_optimizer = optim.Adam(
                list(self.summarizer.parameters()),
                lr=self.config.lr)
            self.g_optimizer = optim.Adam(
                list(self.generator.parameters()),
                lr=self.config.lr)
            self.d_optimizer = optim.Adam(
                list(self.discriminator.parameters()),
                lr=self.config.discriminator_lr)

            self.model.train()
            self.model.apply(apply_weight_norm)

            # Overview Parameters
            # print('Model Parameters')
            # for name, param in self.model.named_parameters():
            #     print('\t' + name + '\t', list(param.size()))

            # Tensorboard
            self.writer = TensorboardWriter(self.config.log_dir)

    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    def reconstruction_loss(self, h_origin, h_fake):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""
        return torch.norm(h_origin - h_fake, p=2)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""
        return torch.abs(torch.mean(scores) - self.config.summary_rate)

    def gan_loss(self, original_prob, fake_prob, uniform_prob):
        """Typical GAN loss + Classify uniformly scored features"""
        gan_loss = torch.mean(torch.log(original_prob) + torch.log(1 - fake_prob)
                              + torch.log(1 - uniform_prob))  # Discriminate uniform score
        return gan_loss

    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            s_e_loss_history = []
            d_loss_history = []
            c_loss_history = []
            for batch_i in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):
                original_data, _ = batch_i[1] # for the [0 , [data, label]]
                image_features = self.backbone(original_data.cuda())
                
                # [batch_size=1, seq_len, 2048]
                # [seq_len, 2048]
                image_features = image_features.view(-1, self.config.input_size)

                #---- Train sANN, Gen, Disc ----#
                if self.config.verbose:
                    tqdm.write('\nTraining sANN and generator and discriminator...')

                # [seq_len, 1, hidden_size]
                #original_features = .detach()).unsqueeze(1)

                scores, weighted_features = self.summarizer(image_features)
                _, uniform_features = self.summarizer(image_features, uniform=True)
                weighted_features, uniform_features = torch.unsqueeze(weighted_features, -1), torch.unsqueeze(uniform_features, -1)
                weighted_features, uniform_features = torch.unsqueeze(weighted_features, -1), torch.unsqueeze(uniform_features, -1)
                weighted_data = self.generator(weighted_features)
                uniform_data = self.generator(uniform_features)
                original_prob = self.discriminator(original_data.cuda()).squeeze()
                fake_prob = self.discriminator(weighted_data).squeeze()
                uniform_prob = self.discriminator(uniform_data).squeeze()
                print("unform: ", uniform_prob.shape)
                print('[original_p: %.3f][fake_p: %.3f][uniform_p: %.3f]'% (original_prob[0].item(), fake_prob[0].item(), uniform_prob[0].item()))

                reconstruction_loss = 0.1*self.reconstruction_loss(weighted_data, uniform_data)
                sparsity_loss = self.sparsity_loss(scores)
                gan_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)
                
                
                print('[recon loss: %.3f][sparsity loss: %.3f][gan loss: %.3f]'% (reconstruction_loss.item(), sparsity_loss.item(), gan_loss.item()))

                s_e_loss = reconstruction_loss + sparsity_loss
                d_loss = reconstruction_loss + gan_loss
                c_loss = -1 * gan_loss # Maximization
                
                if self.config.discriminator_slow_start < epoch_i:
                    self.c_optimizer.zero_grad()
                    c_loss.backward(retain_graph=True)
                    # Gradient cliping
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                    self.c_optimizer.step()
                
                self.s_e_optimizer.zero_grad()
                s_e_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                self.s_e_optimizer.step()
                
                self.d_optimizer.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                self.d_optimizer.step()
                
                
                
                s_e_loss_history.append(s_e_loss.data)
                d_loss_history.append(d_loss.data)
                c_loss_history.append(c_loss.data)
               
                if self.config.verbose:
                    tqdm.write('Plotting...')

                self.writer.update_loss(reconstruction_loss.data, step, 'recon_loss')
                self.writer.update_loss(sparsity_loss.data, step, 'sparsity_loss')
                self.writer.update_loss(gan_loss.data, step, 'gan_loss')

                #self.writer.update_loss(original_prob.data, step, 'original_prob')
                #self.writer.update_loss(fake_prob.data, step, 'fake_prob')
                #self.writer.update_loss(uniform_prob.data, step, 'uniform_prob')

                step += 1

            s_e_loss = torch.stack(s_e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_loss = torch.stack(c_loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')
            self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            self.writer.update_loss(c_loss, epoch_i, 'c_loss_epoch')

            # Save parameters at checkpoint
            ckpt_path = str(self.config.save_dir) + f'_epoch-{epoch_i}.pkl'
            print('Save parameters at ', ckpt_path)
            torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate(epoch_i)

            self.model.train()

    def evaluate(self, epoch_i):
        # checkpoint = self.config.ckpt_path
        # print(f'Load parameters from {checkpoint}')
        # self.model.load_state_dict(torch.load(checkpoint))

        self.model.eval()

        for data, label in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):

            # [seq_len, batch=1, 2048]
            image_features = self.backbone(original_data.cuda())

            # [seq_len]
            scores, _ = self.summarizer(image_features)

            scores = np.array(scores.data).tolist()
            score_save_path = self.config.score_dir.joinpath(f'{self.config.dataset}_{epoch_i}.npy')
            np.save(score_save_path, scores)

    def pretrain(self):
        pass


if __name__ == '__main__':
    pass
