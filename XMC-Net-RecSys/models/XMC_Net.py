import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
from .loss import XMC_IPSLoss

class XMC_Net(nn.Module):
    def __init__(self, inverse_propensity ,num_users, num_movies, num_items, embedding_size,
                 reg_c, reg_t, reg_tc, s_c, s_t, device='cpu',alpha=0.1,gamma=0.1 ):
        super(XMC_Net, self).__init__()

        self.user_e = nn.Embedding(num_users, embedding_size)
        self.tag_e = nn.Embedding(num_items, embedding_size)

        self.movie_b = nn.Embedding(num_movies, 1)
        self.user_b = nn.Embedding(num_users, 1)
        self.tag_b = nn.Embedding(num_items, 1)

        self.item_q = nn.Embedding(num_movies, 1)

        self.fc_tag1 = nn.Linear(2*embedding_size, 1)

        self.reg_c = reg_c
        self.reg_t = reg_t
        self.reg_tc = reg_tc
        self.s_c = s_c
        self.s_t = s_t

        self.apply(self._init_weights)

        self.loss_user_tag = XMC_IPSLoss(device)
        self.loss_t = nn.MSELoss()


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)


    def user_tag_forward(self, user, tag):
        user_embedding = self.user_e(user)
        tag_embedding = self.tag_e(tag)
        tag_input = torch.cat([user_embedding, tag_embedding], dim=1)
        tag_out = self.fc_tag1(tag_input)

        # preds = self.user_b(user)
        # preds += self.tag_b(tag)
        # preds += (user_embedding * tag_embedding).sum(dim=1, keepdim=True)
        # return preds.squeeze()
        return tag_out

    def rating_forward(self, user, item, tag_index):
        # user_embedding = self.user_e(user)
        # all_tag_embedding = self.tag_e(tag_index)
        # mov_alltag_embedding = torch.mean(all_tag_embedding, dim=1) #batch* bedding
        # moview_embedding = self.movie_e(item)
        # # new_item = (item_embedding + moview_embedding)/2
        # mov_input = torch.cat([user_embedding, mov_alltag_embedding, moview_embedding], dim=1)
        # mov_out = self.fc_mov1(mov_input)
        # return mov_out

        user_embedding = self.user_e(user)
        all_tag_embedding = self.tag_e(tag_index)
        item_embedding = torch.mean(all_tag_embedding, dim=1) #batch* bedding
        preds = self.user_b(user)
        preds += torch.mean(self.tag_b(tag_index), dim=1)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        preds += self.item_q(item)
        return preds.squeeze()



    def calculate_loss(self, weight1 , tag_user_list , tag_list, tag_label_list, ips_tag , ips_movie,rating_user_list, item_list, tag_index_list, rating_label_list):
        tag_output = self.user_tag_forward(tag_user_list, tag_list)
        loss1= self.loss_user_tag( tag_output, tag_label_list, ips_tag)

        rating_output = self.rating_forward(rating_user_list, item_list, tag_index_list)
        loss2 = self.loss_t(rating_output, rating_label_list)

        return loss2+loss1


    def predict(self, user, item):
        return self.user_tag_forward(user, item)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
