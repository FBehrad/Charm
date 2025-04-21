import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel, SwinForImageClassification, SwinModel
import yaml
import ml_collections
from torch.utils.data import DataLoader
from mixout.example_huggingface import mixout
from torchvision import models
import numpy as np
from resize_patch_embedding import Resized_patch_embedding
#from flexivit_pytorch import flexivit_small
import timm
from data_utils import image_to_patches, create_binary_mask, padding
import torch.nn.functional as F
# from mixed_res.patch_scorers.random_patch_scorer import RandomPatchScorer
# from mixed_res.quadtree_impl.quadtree_z_curve import ZCurveQuadtreeRunner
# from mixed_res.tokenization.patch_embed import FlatPatchEmbed, PatchEmbed
# from mixed_res.tokenization.tokenizers import QuadtreeTokenizer, VanillaTokenizer

class ScaleEmbs(nn.Module):
    """Adds learnable scale embeddings to the inputs."""

    def __init__(self, num_scales, hidden_size):
        super(ScaleEmbs, self).__init__()
        self.scale_emb_init = nn.init.normal_
        self.se = self.initialization(num_scales, hidden_size)

    def initialization(self, num_scales, hidden_size):
        se = nn.Parameter(torch.empty((1, num_scales, hidden_size)))
        se = self.scale_emb_init(se, mean=0.0, std=0.02)
        se = se.detach().numpy()
        return se

    def forward(self, inputs_positions):
        selected_pe = np.take(self.se[0], inputs_positions.unsqueeze(0), axis=0)
        return torch.tensor(selected_pe)


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_selection = config.data.patch_selection
        self.scales = config.data.scales
        self.patch_size = config.data.patch_size
        config = config.model
        if "dino" in config.encoder_name:
            model = AutoModel.from_pretrained(config.encoder_name)
            self.cls_token = model.embeddings.cls_token
        elif 'swin' in config.encoder_name:
            model = SwinModel.from_pretrained(config.encoder_name)
        else:
            model = ViTModel.from_pretrained(config.encoder_name)
            self.cls_token = model.embeddings.cls_token

        if self.patch_selection == 'original':
            self.patch_embedder = model.embeddings.patch_embeddings
            if config.flexiViT.enable:
                self.resized_version = Resized_patch_embedding(
                    [config.flexiViT.new_patch_size, config.flexiViT.new_patch_size],
                    3, self.patch_embedder.projection.out_channels, self.patch_embedder, True, True)
                self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            if 'swin' not in config.encoder_name:
                self.se = self.se_initialization(self.scales, model.embeddings.position_embeddings.shape[-1])  # scale embeddings
                self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            else:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size * 2))

            self.patch_embedder = model.embeddings.patch_embeddings.projection

        if config.masked:
            self.FNA_initialization()

        self.config = config

    def se_initialization(self, scales, size):
        se = nn.Parameter(torch.empty((1, scales, size)))  # scale embedding
        se = nn.init.normal_(se, mean=0.0, std=0.02)
        return se

    def add_scale_embed(self, input, masks):
        np_se = self.se.detach().cpu().numpy()[0]
        mask = masks[:, 1:].unsqueeze(0).cpu()
        scale_embed = torch.tensor(np.take(np_se, mask, axis=0)).to('cuda')[0]
        input = input + scale_embed
        return input

    def FNA_initialization(self):
        # input channels (4 channels: R, G, B, and the mask)
        num_input_channels = 4
        if self.patch_selection == 'original':
            patch_embedder = self.patch_embedder.projection
        else:
            patch_embedder = self.patch_embedder

        new_patch_embedding = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=patch_embedder.out_channels,
            kernel_size=patch_embedder.kernel_size,
            stride=patch_embedder.stride,
            padding=patch_embedder.padding,
            bias=True
        )

        with torch.no_grad():
            original_weights = patch_embedder.weight
            new_weights = torch.zeros_like(new_patch_embedding.weight)
            new_weights[:, :3, :, :] = original_weights

        new_patch_embedding.weight = nn.Parameter(new_weights)
        new_patch_embedding.bias = patch_embedder.bias

        # Replace the patch embedding layer in the model
        if self.patch_selection == 'original':
            self.patch_embedder.projection = new_patch_embedding
        else:
            self.patch_embedder = new_patch_embedding

        # self.model.embeddings.patch_embeddings.num_channels = num_input_channels

    def forward(self, patches, pos_embeds, masks):

        if self.patch_selection == 'original' and not self.config.flexiViT.enable:
            embedding = self.patch_embedder(patches)
        else:
            patches = patches[:, 1:, :, :, :]  # remove the zero padding for the cls token
            batch_size, l, c, W, H = patches.size()

            patches = patches.reshape(-1, c, W, H)

            if self.config.flexiViT.enable:
                embedding = self.resized_version(patches)
            else:
                embedding = self.patch_embedder(patches)
            embedding = embedding.reshape(batch_size, l, -1)

            if masks is not None:
                padding_mask = (masks == 9).int()
                masks[masks == 9] = 0  # similar to the MUSIQ paper
                seq_length = embedding.shape[1]
                mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
                # replace the masked visual tokens by mask_tokens
                mask = padding_mask[:, 1:].unsqueeze(-1).type_as(mask_tokens)
                embedding = embedding * (1.0 - mask) + mask_tokens * mask

            if 'swin' not in self.config.encoder_name:
                if self.scales != 1:
                    embedding = self.add_scale_embed(embedding, masks)
                cls_token = self.cls_token.expand(batch_size, -1, -1)
                embedding = torch.cat((cls_token, embedding), dim=1)
                embedding = embedding + pos_embeds

        return embedding


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if "dino" in config.encoder_name:
            model = AutoModel.from_pretrained(config.encoder_name)
            self.encoder = model.encoder
            self.layernorm = model.layernorm
        elif "vit-base" in config.encoder_name:
            model = ViTModel.from_pretrained(config.encoder_name)
            self.encoder = model.encoder
            self.layernorm = model.layernorm
        elif "vit-small" in config.encoder_name:
            model = ViTModel.from_pretrained(config.encoder_name)
            self.encoder = model.encoder
            self.layernorm = model.layernorm
        elif 'swin' in  config.encoder_name:
            model = SwinModel.from_pretrained(config.encoder_name)
            self.encoder = model.encoder
            self.layernorm = model.layernorm
        else:
            raise ValueError("Model is not supported.")

    def forward(self, encoder_input):
        encoder_output = self.encoder(encoder_input).last_hidden_state
        encoder_output = self.layernorm(encoder_output)
        return encoder_output


class MlpHead(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        if 'flexiViT' not in config.encoder_name:
            self.linear = nn.Linear(config.hidden_size * 2, config.num_classes)
            self.dropout = nn.Dropout(p=config.dropout_prob)
            self.flexiViT = False
        else:
            self.flexiViT = True
        if data == 'ava' or data == 'para':
            self.activation = nn.Softmax(dim=-1)
            self.multi = False
        elif data == 'aadb':
            if config.multi_class:
                self.sigmoid = nn.Sigmoid()
                self.tanh = nn.Tanh()
                self.multi = True
            else:
                self.activation = nn.Sigmoid()
                self.multi = False
        else:
            self.activation = nn.Sigmoid() #ReLU()
            self.multi = False

    def forward(self, hidden_states):
        if not self.flexiViT:
            cls_token_output = hidden_states[:, 0, :]
            patch_tokens = hidden_states[:, 1:, :]
            linear_input = torch.cat([cls_token_output, patch_tokens.mean(dim=1)], dim=1)
            linear_input = self.dropout(linear_input)
            linear_head = self.linear(linear_input)
        else:
            linear_head = hidden_states
        if self.multi:
            features = self.tanh(torch.cat((linear_head[:, :-3], linear_head[:, -2:]), dim=1))
            score = self.sigmoid(linear_head[:, -3])
            predictions = torch.cat((features[:, :-2], score.unsqueeze(1), features[:, -2:]), dim=1)
        else:
            predictions = self.activation(linear_head)
        return linear_input, predictions


class Model(torch.nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.data = config.data.dataset
        self.patch_selection = config.data.patch_selection
        self.config = config.model
        if self.patch_selection == 'original' and 'swin' in self.config.encoder_name:
            self.swin = SwinForImageClassification.from_pretrained(self.config.encoder_name).base_model
        elif self.patch_selection == 'original' and 'flexiViT' in self.config.encoder_name:
            self.flexiViT = timm.create_model('flexivit_small.1200ep_in1k', pretrained=True, num_classes=self.config.num_classes)
        else:
            self.patch_embeddings = PatchEmbeddings(config)
            self.encoder = Transformer(self.config)
            if self.config.mixout:
                self.encoder = mixout(self.encoder, self.config.mixout_prob)
        self.mlp_head = MlpHead(self.config, self.data)

    def forward(self, patches, pos_embeds, masks):

        if self.patch_selection == 'original' and 'swin' in self.config.encoder_name:
            encoder_output = self.swin(patches).last_hidden_state
        elif self.patch_selection == 'original' and 'flexiViT' in self.config.encoder_name:
            encoder_output = self.flexiViT(patches)
        else:
            embedding = self.patch_embeddings(patches, pos_embeds, masks)
            encoder_output = self.encoder(embedding)

        features, predictions = self.mlp_head(encoder_output)

        return features, predictions


if __name__ == '__main__':
    with open(r'config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = ml_collections.ConfigDict(config)
    config.model.encoder_name = 'microsoft/swin-small-patch4-window7-224'
    model = Model(config)
    print('Hi')
