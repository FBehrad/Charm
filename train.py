import torch
import yaml
import ml_collections
from Dataloaders import prepare_dataloaders
from model import Model
from tqdm import tqdm
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from utils import prepare_path_writer, prepare_config, prepare_optimizers, prepare_scheduler
from loss import Metrics, prepare_loss
import numpy as np
import wandb


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # https://discuss.pytorch.org/t/gpu-utilisation-low-but-memory-usage-high/140025/2

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12350'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_on_gpu(config, checkpoint_folder):
    rank = 'cuda'
    torch.cuda.empty_cache()

    if config.data.dataset == 'ava':
        config.model.num_classes = 10
        dataloaders = prepare_dataloaders(config)
        ava = True
        para = False
    elif config.data.dataset in ['aadb', 'tad66k', 'baid', 'spaq', 'koniq10k']:
        if config.model.multi_class:
            config.model.num_classes = 12
        else:
            config.model.num_classes = 1
        ava = False
        para = False
        dataloaders = prepare_dataloaders(config)
    elif config.data.dataset == 'para':
        config.model.num_classes = 9
        ava = False
        para = True
        dataloaders = prepare_dataloaders(config)
    else:
        raise ValueError('Dataset is not supported. Choose one of these options: 1) ava 2) aadb 3) para 4) tad66k 5)baid 6) spaq 7) koniq10k')
    # Initialize the process group for distributed training
    # setup(rank, world_size)

    model = Model(config)

    if config.training.warm_start:
        checkpoint = torch.load(config.path.warm_start_checkpoint)
        if config.path.warm_start_checkpoint.endswith("best_model.pth"):
            model = checkpoint['model']
        else:
            model = checkpoint['model_state_dict'] #.module

    model = model.to(rank)
    # model = DDP(model, device_ids=[rank])#, find_unused_parameters=True)

    model.train()

    if config.model.frozen_vit:
        if config.training.warm_start:
            for param in model.module.module.patch_embeddings.parameters(): # chain(model.module.encoder.parameters(), model.module.patch_embeddings.parameters()):
                param.requires_grad = False
        else:
            for param in model.module.patch_embeddings.parameters(): # chain(model.module.encoder.parameters(), model.module.patch_embeddings.parameters()):
                param.requires_grad = False

    if config.training.LLDR.enable:
        # https://discuss.pytorch.org/t/two-learning-rate-schedulers-one-optimizer/68136/2
        optimizer_patch_embedding, optimizer_encoder, optimizer_mlp = prepare_optimizers(model, config)
    else:
        optimizer = prepare_optimizers(model, config)

    if config.training.LLDR.enable:
        scheduler_patch, scheduler_encoder, scheduler_mlp = prepare_scheduler(config,
                                                                              optimizer_patch_embedding,
                                                                              optimizer_encoder,
                                                                              optimizer_mlp)
    else:
        scheduler = prepare_scheduler(config, optimizer)

    loss_func = prepare_loss(config.data.dataset)
    metrics = Metrics(config.data.dataset)

    best_plcc = 0 #float('inf')

    # scaler = GradScaler()

    if config.training.warm_start:
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Warm start: resume training from epoch {epoch}')
    else:
        epoch = 0
    print('\nPlease be patient the first epoch is the longest one and we are preparing position embeddings :) ')
    # with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True, record_shapes=True) as prof:
    for epoch in tqdm(range(epoch, config.training.num_epochs)):
        print(f'\nEpoch {epoch}:')

        if config.training.LLDR.enable:
            optimizer_patch_embedding.zero_grad(set_to_none=True)
            optimizer_encoder.zero_grad(set_to_none=True)
            optimizer_mlp.zero_grad(set_to_none=True)

        else:
            optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

        if epoch == 0:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n---------- Total number of trainable parameters: {total_params} ----------")

        if ava:
            score_values = torch.arange(1, config.model.num_classes + 1, dtype=torch.float32, device=rank)

        if para:
            score_values = torch.arange(1, 5.5, step=0.5, dtype=torch.float32, device=rank)

        total_loss = 0.0
        preds_list_train = []
        labels_list_train = []
        # end = time.time()

        for index, sample in enumerate(dataloaders['train']):

            # print(time.time() - end)
            inputs, pos_embeds, masks, labels = sample
            inputs, pos_embeds, masks, labels = inputs.to(rank), pos_embeds.to(rank), masks.to(rank), labels.to(rank)

            if ava or para:
                labels = labels / labels.sum(dim=1, keepdim=True)


            outputs = model(inputs, pos_embeds, masks)

            # if config.training.warm_start and config.data.dataset != 'ava':
            #     score_values = torch.arange(1, 11, dtype=torch.float32, device=rank)
            #     outputs = torch.sum(outputs * score_values, dim=-1) / 10
            #     if config.data.dataset == 'aadb':
            #         outputs = outputs.unsqueeze(1)

            if config.data.dataset == 'aadb' and not config.model.multi_class:
                outputs = outputs.squeeze(1)

            loss = loss_func(outputs, labels.float())
            loss = loss / config.training.accumulation_steps

            if config.training.LLDR.enable:
                optimizer_patch_embedding.zero_grad(set_to_none=True)
                optimizer_encoder.zero_grad(set_to_none=True)
                optimizer_mlp.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

            loss.backward()

            if ava or para:
                outputs = torch.sum(outputs * score_values, dim=-1)
                labels = torch.sum(labels * score_values, dim=-1)
                preds_list_train.extend(outputs.data.cpu().numpy())
                labels_list_train.extend(labels.data.cpu().numpy())
            else:
                if config.model.multi_class:
                    preds_list_train.extend((outputs[:, -3].data.cpu().numpy()))
                    labels_list_train.extend((labels[:, -3].data.cpu().numpy()))
                else:
                    preds_list_train.extend(outputs.data.cpu().numpy())
                    labels_list_train.extend(labels.data.cpu().numpy())


            # scaler.scale(loss).backward()

            if (index + 1) % config.training.accumulation_steps == 0:
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                if config.training.gradient_clipping:
                    # scaler.unscale_(optimizer)
                    # Apply gradient clipping
                    nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                if config.training.LLDR.enable:
                    optimizer_patch_embedding.step()
                    optimizer_encoder.step()
                    optimizer_mlp.step()
                else:
                    optimizer.step()
                # scheduler.step()
                # scaler.step(optimizer)
                # scaler.update()
                # optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

            total_loss += loss.item()
            # wandb.log({"Loss/train": loss.item()}, step=index + epoch * (len(dataloaders['train'])))

        preds_list_train = np.squeeze(np.asarray(preds_list_train))
        labels_list_train = np.squeeze(np.asarray(labels_list_train))
        plcc, srcc, accuracy, mse, mae = metrics.calculate(preds_list_train, labels_list_train)
        wandb.log({"Metrics/ACC train": accuracy, 'epoch':epoch})
        wandb.log({"Metrics/mse train": mse, 'epoch':epoch})
        wandb.log({"Metrics/mae train": mae, 'epoch':epoch})
        wandb.log({"Metrics/PLCC train": plcc, 'epoch':epoch})
        wandb.log({"Metrics/SRCC train": srcc, 'epoch':epoch})
        print('\nSRCC on train set: %.4f.' % (srcc))
        print('PLCC on train set: %.4f.' % (plcc))
        avg_loss = total_loss / len(dataloaders['train'])
        wandb.log({"Loss/average train": avg_loss, 'epoch':epoch})

        # Validation after each epoch
        total_val_loss = 0.0
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for index, sample in enumerate(dataloaders['val']):

                inputs, pos_embeds, masks, labels = sample
                inputs, pos_embeds, masks, labels = inputs.to(rank), pos_embeds.to(rank), masks.to(rank) ,labels.to(rank)

                if ava or para:
                    labels = labels / labels.sum(dim=1, keepdim=True)


                outputs = model(inputs, pos_embeds, masks)

                # if config.training.warm_start and config.data.dataset != 'ava':
                #     score_values = torch.arange(1, 11, dtype=torch.float32, device=rank)
                #     outputs = torch.sum(outputs * score_values, dim=-1) / 10
                #     if config.data.dataset == 'aadb':
                #         outputs = outputs.unsqueeze(1)

                if config.data.dataset == 'aadb' and not config.model.multi_class:
                    outputs = outputs.squeeze(1)
                val_loss = loss_func(outputs, labels)

                total_val_loss += val_loss

                if ava or para:
                    outputs = torch.sum(outputs * score_values, dim=-1)
                    labels = torch.sum(labels * score_values, dim=-1)
                    preds_list.extend(outputs.data.cpu().numpy())
                    labels_list.extend(labels.data.cpu().numpy())
                else:
                    if config.model.multi_class:
                        preds_list.extend((outputs[:, -3].data.cpu().numpy()))
                        labels_list.extend((labels[:, -3].data.cpu().numpy()))
                    else:
                        preds_list.extend(outputs.data.cpu().numpy())
                        labels_list.extend(labels.data.cpu().numpy())


        avg_val_loss = total_val_loss / len(dataloaders['val'])
        wandb.log({"Loss/average validation": avg_val_loss, 'epoch':epoch})

        preds_list = np.squeeze(np.asarray(preds_list))
        labels_list = np.squeeze(np.asarray(labels_list))

        plcc, srcc, accuracy, mse, mae = metrics.calculate(preds_list, labels_list)
        wandb.log({"Metrics/ACC val": accuracy, 'epoch':epoch})
        wandb.log({"Metrics/mse val": mse, 'epoch':epoch})
        wandb.log({"Metrics/mae val": mae, 'epoch':epoch})
        wandb.log({"Metrics/PLCC val": plcc, 'epoch':epoch})
        wandb.log({"Metrics/SRCC val": srcc, 'epoch':epoch})
        print('SRCC on val set: %.4f.' % (srcc))
        print('PLCC on val set: %.4f.' % (plcc))


        if config.training.LLDR.enable:
            current_lr = optimizer_patch_embedding.param_groups[0]['lr']
            wandb.log({"Learning Rate/Patch embedding": current_lr, 'epoch':epoch})
            current_lr = optimizer_encoder.param_groups[0]['lr']
            wandb.log({"Learning Rate/Encoder": current_lr, 'epoch':epoch})
            current_lr = optimizer_mlp.param_groups[0]['lr']
            wandb.log({"Learning Rate/MLP Head": current_lr, 'epoch':epoch})
        else:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"Learning Rate": current_lr, 'epoch':epoch})

        if (epoch % config.training.test_epochs) == 0:
            total_test_loss = 0.0
            preds_list = []
            labels_list = []
            with torch.no_grad():

                for index, sample in enumerate(dataloaders['test']):

                    inputs, pos_embeds, masks, labels = sample
                    inputs, pos_embeds, masks, labels = inputs.to(rank), pos_embeds.to(rank), masks.to(rank), labels.to(rank)

                    if ava or para:
                        labels = labels / labels.sum(dim=1, keepdim=True)

                    outputs = model(inputs, pos_embeds, masks)

                    # if config.training.warm_start and config.data.dataset != 'ava':
                    #     score_values = torch.arange(1, 11, dtype=torch.float32, device=rank)
                    #     outputs = torch.sum(outputs * score_values, dim=-1) / 10
                    #     if config.data.dataset == 'aadb':
                    #         outputs = outputs.unsqueeze(1)

                    if config.data.dataset == 'aadb' and not config.model.multi_class:
                        outputs = outputs.squeeze(1)

                    test_loss = loss_func(outputs, labels)
                    total_test_loss += test_loss

                    if ava or para:
                        outputs = torch.sum(outputs * score_values, dim=-1)
                        labels = torch.sum(labels * score_values, dim=-1)
                        preds_list.extend(outputs.data.cpu().numpy())
                        labels_list.extend(labels.data.cpu().numpy())
                    else:
                        if config.model.multi_class:
                            preds_list.extend((outputs[:, -3].data.cpu().numpy()))
                            labels_list.extend((labels[:, -3].data.cpu().numpy()))
                        else:
                            preds_list.extend(outputs.data.cpu().numpy())
                            labels_list.extend(labels.data.cpu().numpy())


            avg_test_loss = total_test_loss / len(dataloaders['test'])
            wandb.log({"Loss/average test": avg_test_loss, 'epoch':epoch})

            preds_list = np.squeeze(np.asarray(preds_list))
            labels_list = np.squeeze(np.asarray(labels_list))
            plcc, srcc, accuracy, mse, mae = metrics.calculate(preds_list, labels_list)
            wandb.log({"Metrics/ACC test": accuracy, 'epoch':epoch})
            wandb.log({"Metrics/mse test": mse, 'epoch':epoch})
            wandb.log({"Metrics/mae test": mae, 'epoch':epoch})
            wandb.log({"Metrics/PLCC test": plcc, 'epoch':epoch})
            wandb.log({"Metrics/SRCC test": srcc, 'epoch':epoch})
            print('SRCC on test set: %.4f.' % (srcc))
            print('PLCC on test set: %.4f.' % (plcc))

        if (epoch % config.training.save_model_steps) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model,
                # 'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{checkpoint_folder}/model_epoch_{epoch}.pth")
        if plcc > best_plcc:
            checkpoint = {
                'epoch': epoch,
                'model': model,
                # 'optimizer_state_dict': optimizer.state_dict(),
            }
            best_plcc = plcc
            torch.save(checkpoint, f"{checkpoint_folder}/best_model.pth")

        if config.training.LLDR.enable:
            scheduler_patch.step()
            scheduler_encoder.step()
            scheduler_mlp.step()
        else:
            scheduler.step()  # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    # with open("profiler_results_0.txt", "w") as f:
    #     f.write(prof.key_averages().table())
    # writer.close()
    # cleanup()


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = ml_collections.ConfigDict(config)

    config = prepare_config(config)

    print(f"Initial random seed : {torch.initial_seed()}")

    # torch.manual_seed() to reproduce the results
    num_gpus = torch.cuda.device_count()
    path = prepare_path_writer()

    device_ids = list(range(num_gpus))

    log_folder = 'tracking'
    os.makedirs(log_folder, exist_ok=True)
    log_folder = os.path.join(log_folder, path)
    os.makedirs(log_folder, exist_ok=True)

    checkpoint_folder = "checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_folder = os.path.join(checkpoint_folder, path)
    os.makedirs(checkpoint_folder, exist_ok=True)

    with open(os.path.join(log_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config.to_dict(), f)
    wandb.init(project="Charm", entity='fatemehbehrad', config={**config})
    wandb.require("core")
    wandb.define_metric("epoch")
    wandb.define_metric("Metrics/*", step_metric="epoch")
    wandb.define_metric("Loss/*", step_metric="epoch")
    wandb.define_metric("Learning Rate", step_metric="epoch")

    train_on_gpu(config, checkpoint_folder)
    # mp.spawn(train_on_gpu,
    #          args=(num_gpus, config, checkpoint_folder, log_folder),
    #          nprocs=num_gpus,
    #          join=True
    #          )