import csv
from loss import Metrics
from PIL import ImageFile
import torch
import os
from torch.utils.data import DataLoader
import ml_collections
import yaml
ImageFile.LOAD_TRUNCATED_IMAGES = True
from Dataloaders import AestheticDataset
from tqdm import tqdm
import numpy as np


def evaluate(config):
    rank = 'cuda'

    if config.data.dataset == 'ava':
        config.model.num_classes = 10
        score_values = torch.arange(1, config.model.num_classes + 1, dtype=torch.float32, device=rank)
        dataset = AestheticDataset(config, is_train=False)
        data_list = dataset.data_list
        ava = True
        para = False
    elif config.data.dataset in ['aadb', 'tad66k', 'baid', 'spaq', 'koniq10k']:
        if config.model.multi_class:
            config.model.num_classes = 12
        else:
            config.model.num_classes = 1
        ava = False
        para = False
        dataset = AestheticDataset(config, is_train=True, split='test')
        data_list = dataset.data_list
    elif config.data.dataset == 'para':
        config.model.num_classes = 9
        ava = False
        para = True
        dataset = AestheticDataset(config, is_train=False)
        score_values = torch.arange(1, 5.5, step=0.5, dtype=torch.float32, device=rank)
        data_list = dataset.data_list
    else:
        raise ValueError('Dataset is not supported. Choose one of these options: 1) ava 2) aadb 3) para 4) tad66k 5)baid 6) spaq 7) koniq10k')

    custom_dataloader = DataLoader(dataset, batch_size=config.training.batch_size,
                                   shuffle=False)

    checkpoint = torch.load(config.path.warm_start_checkpoint)
    epoch = checkpoint['epoch']
    print(f'Checkpoint from epoch {epoch}')

    if config.path.warm_start_checkpoint.endswith("best_model.pth"):
        model = checkpoint['model']
    else:
        model = checkpoint['model_state_dict']

    model = model.to(rank)
    model.config.update(config.model)
    model.patch_embeddings.scales = config.data.scales
    model.mlp_head.flexiViT = config.model.flexiViT.enable
    model.mlp_head.multi = config.model.multi_class
    metrics = Metrics(config.data.dataset)
    preds_list = []
    labels_list = []
    with torch.no_grad():

        for index, sample in tqdm(enumerate(custom_dataloader)):
            inputs, pos_embeds, masks, labels = sample
            inputs, pos_embeds, masks, labels = inputs.to(rank), pos_embeds.to(rank), masks.to(rank), labels.to(rank)

            if ava or para:
                labels = labels / labels.sum(dim=1, keepdim=True)
            outputs = model(inputs, pos_embeds, masks)

            if 'ava' in config.path.warm_start_checkpoint and config.data.dataset != 'ava':
                score_values = torch.arange(1, 11, dtype=torch.float32, device=rank)
                outputs = torch.sum(outputs * score_values, dim=-1) / 10


            if ava or para:
                outputs = torch.sum(outputs * score_values, dim=-1)
                labels = torch.sum(labels * score_values, dim=-1)
                preds_list.extend(outputs.data.cpu().numpy())
                labels_list.extend(labels.data.cpu().numpy())
            else:
                preds_list.extend(outputs.data.cpu().numpy())
                labels_list.extend(labels.data.cpu().numpy())

        preds_list = np.squeeze(np.asarray(preds_list))
        labels_list = np.squeeze(np.asarray(labels_list))

        plcc, srcc, accuracy, mse, mae = metrics.calculate(preds_list, labels_list)

    print(f"PLCC:{plcc}")
    print(f"SRCC:{srcc}")
    print(f"ACC:{accuracy}")

    destination = r'C:\Users\gestaltrevision\PycharmProjects\MUSIQ_pytorch\Main\Best_models\Para\original'
    with open(os.path.join(destination, "original.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "Prediction", "Actual Label"])
        for i, (image_path, prediction, real_label) in enumerate(zip(data_list, preds_list, labels_list)):
            writer.writerow([image_path, prediction, real_label])


if __name__ == '__main__':
    with open(r'config_eval.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = ml_collections.ConfigDict(config)

    num_gpus = torch.cuda.device_count()
    evaluate(config)
