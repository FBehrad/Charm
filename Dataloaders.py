import csv
import pandas as pd
import torch
from torchvision import transforms
import os
import math
from torch.utils.data import Dataset, DataLoader
from data_utils import add_padding, image_to_patches, select_patches, create_binary_mask, mask_to_patches, pad_or_crop
from data_utils import interpolate_positional_embedding, padding, random_flip, random_rotate, gray_scale, cropping, lcm
from data_utils import calculate_entropy, calculate_frequency, calculate_gradients, patch_selection_saliency
from data_utils import patch_selection_gradient, patch_selection_entropy_based, patch_selection_frequency_based
from data_utils import draw_red_boundaries_on_patches
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFile
import ml_collections
import yaml
import random
from transformers import AutoFeatureExtractor
import timm
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
torchvision.disable_beta_transforms_warning()

class AestheticDataset(Dataset):
    def __init__(self, config, is_train, split=None):
        self.config = config
        self.augmentation = config.data.augmentation
        self.is_train = is_train
        if is_train:
            self.img_folder = config.path.img_folder
            self.mask_folder = config.path.mask_folder
        else:
            self.img_folder = config.path.test_folder
            self.mask_folder = config.path.test_mask_folder

        self.prepare_ratings(config.data.dataset, split)
        self.patch_selection_strategy = config.data.patch_selection
        self.hidden_size = config.data.max_seq_len_from_original_res + 1  # 1 is for cls token
        self.patch_size = config.data.patch_size
        self.patch_stride = config.data.patch_stride
        self.factor = config.data.factor
        self.pos_embeds = {}
        self.load_pos_embed(config.model.encoder_name)
        self.interpolate_offset = 0.1
        self.num_scales = config.data.scales
        self.scale_factor = [self.factor] + [s * ((1 - self.factor) / (self.num_scales - 1)) + self.factor for s in
                                             range(self.num_scales)[1:]]
        self.scaled_patchsizes = [int(element * ((2 ** (self.num_scales - 1)) * self.patch_size)) for element in
                                  self.scale_factor]
        self.initial_hidden_size = config.data.initial_hidden_size
        if 'swin' in config.model.encoder_name:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model.encoder_name)
        if 'flexiViT' in config.model.encoder_name:
            model = timm.create_model(
                'flexivit_small.1200ep_in1k',
                pretrained=True
            )
            model = model.eval()
            data_config = timm.data.resolve_model_data_config(model)
            self.feature_extractor = timm.data.create_transform(**data_config, is_training=False)

    def prepare_ratings(self, dataset, split=None):
        if dataset == 'ava':
            self.ratings = self.prepare_ratings_ava()
            # self.ratings = self.prepare_ratings_high_agreement()
            self.data_list = os.listdir(self.img_folder)

        elif dataset == 'aadb':
            if self.config.model.multi_class:
                self.ratings = self.prepare_ratings_aadb_multiclass(split)
            else:
                self.ratings = self.prepare_ratings_aadb(split)
            self.data_list = list(self.ratings.keys())

        elif dataset == 'para':
            self.ratings = self.prepare_ratings_para()

        elif dataset == 'tad66k':
            self.ratings = self.prepare_ratings_tad66k()
            self.data_list = os.listdir(self.img_folder)

        elif dataset == 'baid':
            self.ratings = self.prepare_ratings_baid(split)

        elif dataset == 'spaq':
            self.ratings = self.prepare_ratings_spaq(split)

        elif dataset == 'koniq10k':
            self.ratings = self.prepare_ratings_koniq10k(split)

    def prepare_ratings_aadb(self, split):
        if split == 'train':
            address = os.path.join(self.config.path.ratings_path, r'imgListTrainRegression_score.txt')
        else:
            address = os.path.join(self.config.path.ratings_path, r'imgListTestNewRegression_score.txt')
        ratings = {}
        with open(address, 'r') as file:
            for line in file:
                image_name, score = line.strip().split()
                score = float(score)
                ratings[image_name] = score
        return ratings

    def prepare_ratings_baid(self, split):
        if split == 'train':
            address = os.path.join(self.config.path.ratings_path, r'train_set.csv')
        else:
            address = os.path.join(self.config.path.ratings_path, r'test_set.csv')
        ratings = pd.read_csv(address)
        ratings = ratings.set_index('image')['score'].to_dict()
        self.data_list = list(ratings.keys())
        return ratings

    def prepare_ratings_aadb_multiclass(self, split):
        path_dir = os.listdir(self.config.path.ratings_path)
        if split == 'train':
            train = []
            val = []
            for filename in path_dir:
                if filename.startswith('imgListTrainRegression_'):
                    train.append(filename)
                elif filename.startswith('imgListValidationRegression_'):
                    val.append(filename)
            labels = []
            ratings = {}
            for file1, file2 in zip(train, val):
                labels.append(file1.split('_')[1].split('.')[0])
                with open(os.path.join(self.config.path.ratings_path, file1), 'r') as file:
                    for line in file:
                        image_name, score = line.strip().split()
                        score = float(score)
                        if image_name not in ratings:
                            ratings[image_name] = []
                        ratings[image_name].append(score)
                with open(os.path.join(self.config.path.ratings_path, file2), 'r') as file:
                    for line in file:
                        image_name, score = line.strip().split()
                        score = float(score)
                        if image_name not in ratings:
                            ratings[image_name] = []
                        ratings[image_name].append(score)
            # print(labels)
        else:
            test = []
            for filename in path_dir:
                if filename.startswith('imgListTestNewRegression_'):
                    test.append(filename)

            labels = []
            ratings = {}
            for file1 in test:
                labels.append(file1.split('_')[1].split('.')[0])
                with open(os.path.join(self.config.path.ratings_path, file1), 'r') as file:
                    for line in file:
                        image_name, score = line.strip().split()
                        score = float(score)
                        if image_name not in ratings:
                            ratings[image_name] = []
                        ratings[image_name].append(score)
            # print(labels)

        return ratings

    def prepare_ratings_ava(self):
        ratings = {}
        with open(self.config.path.ratings_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                id = row[1]
                rates = [int(x) for x in row[2:-3]]
                ratings[str(id)] = rates
        return ratings

    def prepare_ratings_spaq(self, split):
        annotations = pd.read_excel(self.config.path.ratings_path)
        ratings = annotations.set_index('Image name')['MOS'].to_dict()
        annotation_path = '\\'.join(self.config.path.ratings_path.split('\\')[:-1])
        if split == 'train':
            with open(os.path.join(annotation_path, 'train_split.csv'), "r") as f:
                reader = csv.reader(f)
                self.data_list = list(reader)[0]
        else:
            with open(os.path.join(annotation_path, 'test_split.csv'), "r") as f:
                reader = csv.reader(f)
                self.data_list = list(reader)[0]
        return ratings

    def prepare_ratings_koniq10k(self, split):
        annotations = pd.read_csv(self.config.path.ratings_path)
        ratings = annotations.set_index('image_name')['MOS'].to_dict()
        annotation_path = '\\'.join(self.config.path.ratings_path.split('\\')[:-1])
        if split == 'train':
            with open(os.path.join(annotation_path, 'train_split.csv'), "r") as f:
                reader = csv.reader(f)
                self.data_list = list(reader)[0]
        else:
            with open(os.path.join(annotation_path, 'test_split.csv'), "r") as f:
                reader = csv.reader(f)
                self.data_list = list(reader)[0]
        return ratings

    def prepare_ratings_para(self):
        if self.is_train:
            annotations = pd.read_csv(os.path.join(self.config.path.ratings_path, 'PARA-GiaaTrain.csv'))
        else:
            annotations = pd.read_csv(os.path.join(self.config.path.ratings_path, 'PARA-GiaaTest.csv'))
        self.data_list = os.listdir(self.img_folder)
        image_names = annotations['imageName']
        session_id = annotations['sessionId']
        imgs = session_id + '_' + image_names
        imgs = imgs.values.tolist()
        labels = annotations.iloc[:, 3:12]
        ratings = {}
        for index, name in enumerate(imgs):
            if name in self.data_list:
                scores = labels.iloc[index].values.tolist()
                ratings[name] = scores
            else:
                continue
        return ratings

    def prepare_ratings_tad66k(self):
        ratings = {}
        with open(self.config.path.ratings_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    data = row[0].split(',')
                    id = data[0]
                    rate = data[1]
                    ratings[str(id)] = float(rate)
        return ratings

    def prepare_pos_embed(self, coarse_mask, fine_mask, size, dim):

        coarse_mask = 1 - coarse_mask  # to show the areas that are not selected

        fine_size = (size[0] * self.factor, size[1] * self.factor)
        if fine_size not in self.pos_embeds:
            patch_pos_embed, class_pos_embed = interpolate_positional_embedding(self.pos_embed, fine_size,
                                                                                self.interpolate_offset, dim)
            self.pos_embeds[fine_size] = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
            fine_pos_embed = patch_pos_embed[:, fine_mask.type(torch.bool), :]
            fine_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), fine_pos_embed),
                                       dim=1)  # we need to add the clas pos embed
        else:
            patch_pos_embed = self.pos_embeds[fine_size][:, 1:, :]
            class_pos_embed = self.pos_embeds[fine_size][:, 0, :]
            fine_pos_embed = patch_pos_embed[:, fine_mask.type(torch.bool), :]
            fine_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), fine_pos_embed), dim=1)

        if size not in self.pos_embeds:
            patch_pos_embed, class_pos_embed = interpolate_positional_embedding(self.pos_embed, size,
                                                                                self.interpolate_offset,
                                                                                dim)
            self.pos_embeds[size] = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
            coarse_pos_embed = patch_pos_embed[:, coarse_mask.type(torch.bool), :]
        else:
            patch_pos_embed = self.pos_embeds[size][:, 1:, :]
            coarse_pos_embed = patch_pos_embed[:, coarse_mask.type(torch.bool), :]

        final_pos_embed = torch.cat([fine_pos_embed, coarse_pos_embed], dim=1)
        return final_pos_embed

    def random_drop(self, input, pos_embed, mask_ms=None):
        elements_to_remove = input.shape[0] - self.hidden_size
        indices_to_remove = random.sample(range(1, input.shape[0]), elements_to_remove)  # to keep the cls token
        mask = torch.ones(input.shape[0], dtype=torch.bool)
        mask[indices_to_remove] = False
        input = input[mask, :, :, :]
        pos_embed = pos_embed[mask, :]
        if mask_ms is not None:
            mask_ms = mask_ms[mask]
            return input, pos_embed, mask_ms
        else:
            return input, pos_embed

    def normal_flow(self, original_image):
        original_image = add_padding(original_image, self.patch_size)
        channels, H, W = original_image.size()
        n_crops_w = math.ceil(W / self.patch_size)
        n_crops_H = math.ceil(H / self.patch_size)
        size = (n_crops_H, n_crops_w)

        image_patches = image_to_patches(original_image, self.patch_size, self.patch_stride)
        input = torch.stack(image_patches)

        # visualize_patches(image_patches, 32, 10).save(r'C:\Users\gestaltrevision\Pictures\sample_1.jpg')
        # visualize_patches(depth_patches, 32, 10).save(r'C:\Users\gestaltrevision\Pictures\sample_1_depth.jpg')

        input = torch.cat((torch.zeros(1, input.shape[1], input.shape[2], input.shape[3]),
                           input), dim=0)  # instead of CLS token

        if size not in self.pos_embeds:
            patch_pos_embed, class_pos_embed = interpolate_positional_embedding(self.pos_embed, size, 0.1,
                                                                                self.pos_embed.shape[-1])
            pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
            self.pos_embeds[size] = pos_embed
        else:
            pos_embed = self.pos_embeds[size]

        pos_embed = pos_embed.squeeze(0)
        if input.shape[0] != pos_embed.shape[0]:
            raise ValueError("Pos embedding length doesn't match the tokens length.")

        if input.shape[0] < self.hidden_size:
            mask = torch.zeros(input.shape[0])
            padded_area = self.hidden_size - input.shape[0]
            pad = [9] * padded_area
            mask = torch.cat((mask, torch.Tensor(pad)), 0)
            input = padding(input, self.hidden_size)
            pos_embed = padding(pos_embed.unsqueeze(-1).unsqueeze(-1), self.hidden_size).squeeze(-1).squeeze(-1)

        elif input.shape[0] > self.hidden_size:
            input, pos_embed = self.random_drop(input, pos_embed)
            # input = input[:self.hidden_size, :, :, :]
            # pos_embed = pos_embed[:self.hidden_size, :]
            mask = torch.zeros(self.hidden_size)

        else:
            mask = torch.zeros(self.hidden_size)

        return input, pos_embed, mask

    def prepare_pos_embed_ms(self, masks, dim):
        fine_pos_embeds = []
        for i, fine_mask in enumerate(masks):
            fine_size = fine_mask.size()
            if fine_size not in self.pos_embeds:
                patch_pos_embed, class_pos_embed = interpolate_positional_embedding(self.pos_embed, fine_size,
                                                                                    self.interpolate_offset, dim)
                self.pos_embeds[fine_size] = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
                fine_pos_embed = patch_pos_embed[:, fine_mask.type(torch.bool).flatten(), :]
            else:
                patch_pos_embed = self.pos_embeds[fine_size][:, 1:, :]
                class_pos_embed = self.pos_embeds[fine_size][:, 0, :]
                fine_pos_embed = patch_pos_embed[:, fine_mask.type(torch.bool).flatten(), :]

            fine_pos_embeds.append(fine_pos_embed)

        fine_pos_embeds = torch.cat(fine_pos_embeds, dim=1)

        final_pos_embed = torch.cat([class_pos_embed.unsqueeze(0), fine_pos_embeds], dim=1)
        return final_pos_embed

    def calculate_importance(self, patch_selection, image_patches, patch_size, patch_stride, mask=None):
        if patch_selection == 'frequency':
            importance = calculate_frequency(image_patches)
        elif patch_selection == 'entropy':
            importance = calculate_entropy(image_patches)
        elif patch_selection == 'gradient':
            importance = calculate_gradients(image_patches)
        elif patch_selection == 'saliency':
            mask_patches = image_to_patches(mask, patch_size, patch_stride)
            mask_tensors = torch.stack(mask_patches)
            mask_tensors = mask_tensors.reshape(mask_tensors.shape[0], -1)
            # salient_part = torch.any(mask_tensors != 0, dim=1)
            # salient_indices = torch.where(salient_part)[0].tolist()
            salient_part = mask_tensors.sum(1) > (mask_tensors.max() // 2)
            importance = torch.where(salient_part)[0].tolist()
        else:
            importance = [0] * len(image_patches)

        return importance

    def patch_selection(self, patch_selection, importance, n_patches, scale, total_patches):
        if patch_selection == 'random':
            selected_indices = random.sample(total_patches, n_patches)
        elif patch_selection == 'frequency':
            selected_indices = patch_selection_frequency_based(importance, n_patches, scale, self.num_scales)
        elif patch_selection == 'entropy':
            selected_indices = patch_selection_entropy_based(importance, n_patches, scale, self.num_scales)
        elif patch_selection == 'gradient':
            selected_indices = patch_selection_gradient(importance, n_patches, scale, self.num_scales)
        elif patch_selection == 'saliency':
            if len(importance) >= n_patches:
                selected_indices = patch_selection_saliency(importance, n_patches)
            else:
                if set(total_patches).isdisjoint(set(importance)):
                    selected_indices = random.sample(total_patches, n_patches)
                else:
                    non_salient = list(set(total_patches) - set(importance))
                    extra = n_patches - len(importance)
                    extra_patches = random.sample(non_salient, extra)
                    selected_indices = extra_patches + importance
        else:
            raise ValueError(f'{patch_selection} patch selection is not supported.')

        return selected_indices


    def highResPreserve_ms(self, image, mask=None):
        image = pad_or_crop(image, lcm(self.scaled_patchsizes))
        if mask is not None:
            mask = pad_or_crop(mask, lcm(self.scaled_patchsizes))
            if mask.size()[1:] != image.size()[1:]:
                raise ValueError('Image size and mask size do not match.')

        # To make all of them a multiple of the patch size, we use the following line.
        patch_sizes = [x + (self.patch_size - x % self.patch_size) if x % self.patch_size != 0 else x for x in
                       self.scaled_patchsizes]
        patch_strides = [x + (self.patch_size - x % self.patch_size) if x % self.patch_size != 0 else x for x in
                         self.scaled_patchsizes]

        image_patches = image_to_patches(image, patch_sizes[-1], patch_strides[-1])

        # detecting the important indices in patches
        importance = self.calculate_importance(self.patch_selection_strategy, image_patches, patch_sizes[-1],
                                               patch_strides[-1], mask)

        n_patch_per_col = image.size()[-1] // patch_sizes[-1]
        n_patch_per_row = image.size()[-2] // patch_sizes[-1]

        ratio = 1 / self.num_scales

        n_patches = int((self.initial_hidden_size * ratio) / ((2 ** (self.num_scales - 1)) ** 2))
        # n_token_per_scale = (len(image_patches) // self.num_scales)
        # ratio_h = patch_sizes[-1] // patch_sizes[0]
        # n_patches_h = n_token_per_scale // ratio_h ** 2  # number of patches in the highest resolution
        # level n : Highest resolution : For example: 64 between 16, 32, 64
        locals()[f'selected_indices_l{self.num_scales - 1}'] = self.patch_selection(self.patch_selection_strategy,
                                                                                    importance,
                                                                                    n_patches, self.num_scales - 1,
                                                                                    range(len(image_patches)))

        # To visualize the selected area
        # transform = T.ToPILImage()
        # image_pil = transform(image)
        # output = draw_red_boundaries_on_patches(image_pil,  locals()[f'selected_indices_l{self.num_scales - 1}'], patch_sizes[-1])
        # output.show()

        locals()[f'patches_l{self.num_scales - 1}'] = []
        for i in (sorted(locals()[f'selected_indices_l{self.num_scales - 1}'])):
            patches = image_to_patches(image_patches[i], self.patch_size, self.patch_stride)
            locals()[f'patches_l{self.num_scales - 1}'].extend(patches)

        locals()[f'mask_l{self.num_scales - 1}'] = [self.num_scales - 1] * len(
            locals()[f'patches_l{self.num_scales - 1}'])

        # level 1 ... n - 1
        remaining_patches = range(len(image_patches))
        intermediate_patches = []
        intermediate_masks = []
        selected_indices = []
        for i in range(self.num_scales):
            if i == 0 or i == self.num_scales - 1:
                continue
            else:
                p = 0
                remaining_patches = list(
                    set(remaining_patches) - set(locals()[f'selected_indices_l{self.num_scales - 1}']))

                # ratio = patch_sizes[i] // patch_sizes[0]
                # n_patches = n_token_per_scale // ratio ** 2

                locals()[f'selected_indices_l{i}'] = self.patch_selection(self.patch_selection_strategy, importance,
                                                                          n_patches, i, remaining_patches)

                for index in sorted(locals()[f'selected_indices_l{i}']):
                    a = F.interpolate(image_patches[index].unsqueeze(0),
                                      size=(self.scaled_patchsizes[i], self.scaled_patchsizes[i]),
                                      mode='bicubic').squeeze(0)
                    patch = image_to_patches(a, self.patch_size, self.patch_stride)
                    intermediate_patches.extend(patch)
                    p = p + len(patch)

                locals()[f'mask_l{i}'] = [i] * p
                intermediate_masks.extend(locals()[f'mask_l{i}'])
                selected_indices.extend(locals()[f'selected_indices_l{i}'])

        selected_indices = selected_indices + locals()[f'selected_indices_l{self.num_scales - 1}']

        # Level 0
        p = 0
        selected_indices_l0 = [x for x in range(0, len(image_patches)) if
                               x not in selected_indices]
        remaining_patches = []
        for index in sorted(selected_indices_l0):
            a = F.interpolate(image_patches[index].unsqueeze(0),
                              size=(self.scaled_patchsizes[0], self.scaled_patchsizes[0]),
                              mode='bicubic').squeeze(0)
            patch = image_to_patches(a, self.patch_size, self.patch_stride)
            remaining_patches.extend(patch)
            p = p + len(patch)

        mask_l0 = [0] * p

        final = remaining_patches + intermediate_patches + locals()[
            f'patches_l{self.num_scales - 1}']  # level 0, ..., n

        mask_ms = mask_l0 + intermediate_masks + locals()[f'mask_l{self.num_scales - 1}']
        final = torch.stack(final)
        # print(final.size())
        # pos embed
        masks = []
        for i in range(self.num_scales):
            p = patch_sizes[i] // self.patch_size
            mask = create_binary_mask((3, p * n_patch_per_row, p * n_patch_per_col), p,
                                      locals()[f'selected_indices_l{i}'])
            masks.append(mask)

        pos_embeds = self.prepare_pos_embed_ms(masks, self.pos_embed.shape[-1]).squeeze(0)

        final_tensor = final

        final_tensor = torch.cat((torch.zeros(1, final_tensor.shape[1], final_tensor.shape[2], final_tensor.shape[3]),
                                  final_tensor), dim=0)  # instead of CLS token
        mask_ms.insert(0, 0)  # for cls token

        if final_tensor.shape[0] != pos_embeds.shape[0]:
            raise ValueError("Pos embedding length doesn't match the tokens length.")

        if final_tensor.shape[0] < self.hidden_size:
            input = padding(final_tensor, self.hidden_size)
            pos_embeds = padding(pos_embeds.unsqueeze(-1).unsqueeze(-1), self.hidden_size).squeeze(-1).squeeze(-1)

            padded_area = self.hidden_size - final_tensor.shape[0]
            pad = [9] * padded_area
            mask_ms = mask_ms + pad
            mask = torch.Tensor(mask_ms)

        elif final_tensor.shape[0] > self.hidden_size:
            input, pos_embeds, mask = self.random_drop(final_tensor, pos_embeds, torch.Tensor(mask_ms))
        else:
            input = final_tensor
            mask = torch.Tensor(mask_ms)

        return input, pos_embeds, mask

    def preprare_patches(self, original_image, saliency_mask=None):
        channels, H, W = original_image.size()
        n_crops_w = math.ceil(W / self.patch_size)
        n_crops_H = math.ceil(H / self.patch_size)

        if n_crops_H * n_crops_w <= self.hidden_size or self.config.model.flexiViT.enable:
            if self.config.model.flexiViT.enable:
                self.patch_size = self.config.model.flexiViT.new_patch_size
                self.patch_stride = self.config.model.flexiViT.new_patch_size
            input, pos_embed, mask = self.normal_flow(original_image)
        else:
            input, pos_embed, mask = self.highResPreserve_ms(original_image, saliency_mask)

        return input, pos_embed, mask

    def preprare_patches_test(self, original_image, depth=None):
        channels, H, W = original_image.size()
        n_crops_w = math.ceil(W / self.patch_size)
        n_crops_H = math.ceil(H / self.patch_size)
        size = (n_crops_H, n_crops_w)

        original_image = add_padding(original_image, self.patch_size)
        image_patches = image_to_patches(original_image, self.patch_size, self.patch_stride)
        input_image = torch.stack(image_patches)

        if depth is not None:
            depth = add_padding(depth, self.patch_size)
            depth_patches = image_to_patches(depth, self.patch_size, self.patch_stride)
            input_depth = torch.stack(depth_patches)
            input = torch.cat((input_image, input_depth), dim=1)
        else:
            input = input_image

        input = torch.cat((torch.zeros(1, input.shape[1], input.shape[2], input.shape[3]),
                           input), dim=0)  # instead of CLS token

        if size not in self.pos_embeds:
            patch_pos_embed, class_pos_embed = interpolate_positional_embedding(self.pos_embed, size, 0.1,
                                                                                self.pos_embed.shape[-1])
            pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
            self.pos_embeds[size] = pos_embed
        else:
            pos_embed = self.pos_embeds[size]

        pos_embed = pos_embed.squeeze(0)
        if input.shape[0] != pos_embed.shape[0]:
            raise ValueError("Pos embedding length doesn't match the tokens length.")

        mask = torch.zeros(input.shape[0])

        return input, pos_embed, mask

    def __len__(self):
        return len(self.data_list)

    def augment(self, image, depth=None):
        if self.augmentation == 'HF' or self.augmentation == 'all':
            if random.random() < 0.5:
                image, depth = random_flip(image, depth)

        if self.augmentation == 'RR' or self.augmentation == 'all':
            if random.random() < 0.5:
                image, depth = random_rotate(image, depth)

        # if self.augmentation == 'GS' or self.augmentation == 'all':
        #     if random.random() < 0.5:
        #         image = gray_scale(image)

        return image, depth

    def load_pos_embed(self, model_name):
        if model_name == 'facebook/dinov2-small':
            path = "pos_embeds/dino_small.pt"
        elif model_name == 'facebook/dinov2-large':
            path = "pos_embeds/dino_large.pt"
        elif model_name == 'facebook/dinov2-base':
            path = "pos_embeds/dino_base.pt"
        elif model_name == 'facebook/dino-vits16':
            path = 'pos_embeds/dino_vit16.pt'
        elif model_name == 'google/vit-base-patch32-224-in21k':
            path = 'pos_embeds/vit_base_32_224.pt'
        elif model_name == 'google/vit-base-patch32-384':
            path = 'pos_embeds/vit_base_32_384.pt'
        elif model_name == 'WinKawaks/vit-small-patch16-224':
            path = 'pos_embeds/vit_small_16_224.pt'
        else:
            path = 'pos_embeds/vit_base_16_224.pt'
        self.pos_embed = torch.load(path).detach()

    def prepare_ratings_high_agreement(self):
        ratings = {}
        with open(self.config.path.ratings_path_high_agreement, newline='') as csvfile:
            data = csvfile.readlines()
            data = data[1:]
            for line in data:
                id = line.split(',')[0]
                rates = [int(x.strip('"[ \r\n ]')) for x in line.split(',')[3:]]
                ratings[id] = rates
            # reader = csv.reader(csvfile, delimiter=' ')
            # for row in reader:
            #     id = row[1]
            #     rates = [int(x) for x in row[2:-3]]
            #     ratings[str(id)] = rates

        return ratings

    def random_crop(self, img, mask=None, output_size=(224, 224)):

        max_y_start = img.shape[1] - output_size[0] + 1
        max_x_start = img.shape[2] - output_size[1] + 1

        y_start = torch.randint(0, max_y_start, size=(1,))
        x_start = torch.randint(0, max_x_start, size=(1,))
        cropped_image = img[:, y_start:y_start + output_size[0], x_start:x_start + output_size[1]]
        if mask is not None:
            cropped_mask = mask[:, y_start:y_start + output_size[0], x_start:x_start + output_size[1]]
        else:
            cropped_mask = None

        return cropped_image, cropped_mask

    def center_crop(self, image, crop_size=(224, 224)):
        height, width = image.shape[1:]
        crop_height, crop_width = crop_size
        top_left_y = (height - crop_height) // 2
        top_left_x = (width - crop_width) // 2
        cropped_image = image[:, top_left_y:top_left_y + crop_height, top_left_x:top_left_x + crop_width]
        return cropped_image

    def resize(self, img, desired_shortest_edge=256):
        height, width = img.shape[1:]
        shortest_edge = min(height, width)
        if shortest_edge == desired_shortest_edge:
            return img
        elif shortest_edge < desired_shortest_edge:
            diff_w = ((desired_shortest_edge - width)//2 )+ 1
            diff_h = ((desired_shortest_edge - height) // 2 )+ 1
            if diff_h < 0:
                diff_h = 0
            if diff_w < 0:
                diff_w = 0
            transform = transforms.Pad((diff_w, diff_h))
            img = transform(img)
        else:
            if height > width:
                scale_factor = height / width
                new_width = desired_shortest_edge
                new_height = desired_shortest_edge * scale_factor
            else:
                scale_factor = width / height
                new_height = desired_shortest_edge
                new_width = desired_shortest_edge * scale_factor

            resize = transforms.Resize((int(new_height), int(new_width)), antialias=True)
            # Apply the transformation to your image tensor
            img = resize(img)
        return img

    def read_label(self, img_name):
        if self.config.data.dataset == 'ava':
            label = torch.Tensor(self.ratings[img_name.split('.')[0]])
        elif self.config.data.dataset == 'aadb':
            if self.config.model.multi_class:
                label = torch.Tensor(self.ratings[img_name])
            else:
                label = self.ratings[img_name]
        elif self.config.data.dataset == 'para':
            label = torch.Tensor(self.ratings[img_name])
        elif self.config.data.dataset == 'tad66k' or self.config.data.dataset == 'baid':
            label = torch.Tensor([self.ratings[
                                      img_name]]) / 10  # (torch.Tensor([self.ratings[img_name]]) - min(self.ratings.values())) / (
            # max(self.ratings.values()) - min(self.ratings.values()))
        elif self.config.data.dataset == 'spaq':
            label = torch.Tensor([self.ratings[
                                      img_name]]) / 100

        elif self.config.data.dataset == 'koniq10k':
            label = torch.Tensor([self.ratings[
                                      img_name]]) / 5
        else:
            raise ValueError('Dataset is not supported.')
        return label

    def normalize(self, img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean, std)
        new_img = transform_norm(img)
        return new_img

    def __getitem__(self, idx):
        img_name = self.data_list[idx]
        img_path = os.path.join(self.img_folder, img_name)

        image_pil = Image.open(img_path)
        image_pil = image_pil.convert("RGB")

        if self.patch_selection_strategy == 'saliency':
            mask_name = img_path.split('\\')[-1].split('.')[0] + '_mask.jpg'
            mask_path = os.path.join(self.mask_folder, mask_name)
            mask = Image.open(mask_path)
            mask = mask.convert("L")
            mask = mask.point(lambda p: 255 if p > 0.5 else 0)  # making the mask binary
        else:
            mask = None

        if self.config.data.dataset == 'para' or self.config.data.dataset == 'spaq':
            width, height = image_pil.size
            if max(width, height) > 1024:
                max_size = 1024
                ratio = min(max_size / width, max_size / height)

                new_width = int(width * ratio)
                new_height = int(height * ratio)

                image_pil = image_pil.resize((new_width, new_height))
                if self.patch_selection_strategy == 'saliency':
                    mask = mask.resize((new_width, new_height))

        # if image.size[0] != image.size[1]: # to change the aspect ratio while keeping high-resolution details
        #     final_size= max(image.size[0], image.size[1])
        #     image, depth = self.resize(image, depth, (final_size, final_size))

        # width, height = image.size  # to keep the aspect ratio while losing high-resolution details
        # ratio = max(width, height) / min(width, height)
        # new_smaller_dim = 256
        # new_larger_dim = int(new_smaller_dim * ratio)
        # if width > height:
        #     size = (new_smaller_dim, new_larger_dim)
        # else:
        #     size = (new_larger_dim, new_smaller_dim)
        #
        # image, depth = self.resize(image, depth, size ) #(256, 256)) # to change both

        if self.is_train and self.augmentation is not None:
            image_pil, mask = self.augment(image_pil, mask)

        # image_pil.thumbnail((512, 512)) # for testing padding effect
        # image_pil = ImageOps.pad(image_pil, (512, 512), method=Image.Resampling.LANCZOS, color=(0, 0, 0))

        image = transforms.ToTensor()(image_pil)

        if self.patch_selection_strategy == 'saliency':
            mask = transforms.ToTensor()(mask)

            if image.shape[1:] != mask.shape[1:]:
                raise Exception('Image and Depth map should have the same size !')

        if self.patch_selection_strategy == 'original' and not self.config.model.flexiViT.enable and not self.config.data.muller:

            if 'swin' in self.config.model.encoder_name:
                inputs = self.feature_extractor(images=image_pil, return_tensors="pt")
                input = inputs.data['pixel_values'][0]
            elif 'flexiViT' in self.config.model.encoder_name:
                input = self.feature_extractor(image_pil)
            # input = self.resize(image)
            # input, _ = self.random_crop(input, output_size=(240, 240))  # self.center_crop(input)
            # input = self.normalize(input)
            else:
                input = self.resize(image, 384)
                input, _ = self.random_crop(input, output_size=(384, 384))  # self.center_crop(input)
                # input = self.normalize(input)

            pos_embed = torch.tensor([])
            mask_tokens = torch.tensor([])
        elif self.patch_selection_strategy == 'original' and not self.config.model.flexiViT.enable and self.config.data.muller:
            input = image
            pos_embed = torch.tensor([])
            mask_tokens = torch.tensor([])
        else:
            input, pos_embed, mask_tokens = self.preprare_patches(image, mask)

        # input = self.normalize(input)

        label = self.read_label(img_name)

        return input, pos_embed, mask_tokens, label


def prepare_train_test_split(path, annotation_path):
    annotation_path = '\\'.join(annotation_path.split('\\')[:-1])
    if not os.path.exists(os.path.join(annotation_path, 'train_split.csv')):
        image_files = os.listdir(path)
        random.shuffle(image_files)
        split_index = int(len(image_files) * 0.8)
        train_images = image_files[:split_index]
        test_images = image_files[split_index:]
        train_add = annotation_path + f'\\train_split.csv'
        test_add = annotation_path + f'\\test_split.csv'
        with open(train_add, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(train_images)
        with open(test_add, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(test_images)

def prepare_dataloaders(config):
    if config.data.dataset == 'aadb' or config.data.dataset == 'baid':
        im_dataset = AestheticDataset(config, is_train=True, split='train')
        train_dataset, val_dataset = torch.utils.data.random_split(im_dataset,
                                                                   [int(len(im_dataset) * config.training.train_size),
                                                                    len(im_dataset) - int(
                                                                        len(im_dataset) * config.training.train_size)])
        test_dataset = AestheticDataset(config, is_train=False, split='test')

    elif config.data.dataset in ['ava', 'para', 'tad66k']:
        im_dataset = AestheticDataset(config, is_train=True)
        train_dataset, val_dataset = torch.utils.data.random_split(im_dataset,
                                                                   [int(len(im_dataset) * config.training.train_size),
                                                                    len(im_dataset) - int(
                                                                        len(im_dataset) * config.training.train_size)])

        test_dataset = AestheticDataset(config, is_train=False)
    elif config.data.dataset in ['spaq', 'koniq10k']:
        prepare_train_test_split(config.path.img_folder, config.path.ratings_path)
        im_dataset = AestheticDataset(config, is_train=True, split='train')
        train_dataset, val_dataset = torch.utils.data.random_split(im_dataset,
                                                                   [int(len(im_dataset) * config.training.train_size),
                                                                    len(im_dataset) - int(
                                                                        len(im_dataset) * config.training.train_size)])
        test_dataset = AestheticDataset(config, is_train=False, split='test')
    else:
        raise ValueError(f'{config.data.dataset} is not supported. Please choose ava, para, tad66k, or aadb.')

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                              pin_memory=True
                              , num_workers=config.training.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=config.training.shuffle_data,
                            pin_memory=True,
                            num_workers=config.training.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                             shuffle=config.training.shuffle_data, drop_last=True,
                             pin_memory=True,
                             num_workers=config.training.num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return dataloaders


if __name__ == '__main__':
    with open(r'config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = ml_collections.ConfigDict(config)

    # dataloaders = prepare_dataloaders(config)
    #
    # for index, sample in enumerate(dataloaders['train']):
    #     print("hi")

    dataset = AestheticDataset(config, True)

    img_path = r'D:\Datasets\AADB\train\farm1_308_20158286555_a3bb34e65e_b.jpg'

    image_pil = Image.open(img_path)
    image_pil = image_pil.convert("RGB")
    image = transforms.ToTensor()(image_pil)
    a, b, c = dataset.preprare_patches(image, None)
