import os
from datetime import datetime
from torch.optim import SGD, Adam, AdamW
import torch.optim as optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def prepare_path_writer():
    now = datetime.now()
    dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    return dir


def prepare_config(config):
    if "dinov2-small" in config.model.encoder_name or "vit-small" in config.model.encoder_name or 'flexiViT' in config.model.encoder_name:
        config.model.hidden_size = 384
    elif "dinov2-base" in config.model.encoder_name or "vit-base" in config.model.encoder_name or "swin" in config.model.encoder_name:
        config.model.hidden_size = 768
    elif "dinov2-large" in config.model.encoder_name:
        config.model.hidden_size = 1024
    else:
        raise ValueError("Model is not supported.")

    if config.data.scales == 2:
        config.data.initial_hidden_size = 512
    else:
        config.data.initial_hidden_size = 768  # used for calculating the number of patches in the multi-scale approach

    if config.model.multi_class and config.data.dataset != 'aadb':
        raise ValueError('Multiclass setup only supports AADB dataset.')

    return config


def prepare_optimizers(model, config):
    if config.training.optimizer == 'sgd':
        if config.training.LLDR.enable:
            optimizer_patch_embedding = SGD(model.patch_embeddings.parameters(),
                                            lr=float(config.training.LLDR.max_lr_patch),
                                            weight_decay=config.training.LLDR.weight_decay_patch)
            optimizer_encoder = SGD(model.encoder.parameters(),
                                    lr=float(config.training.LLDR.max_lr_encoder),
                                    weight_decay=config.training.LLDR.weight_decay_encoder)
            optimizer_mlp = SGD(model.mlp_head.parameters(),
                                lr=float(config.training.LLDR.max_lr_mlphead),
                                weight_decay=config.training.LLDR.weight_decay_mlp)

            return optimizer_patch_embedding, optimizer_encoder, optimizer_mlp

        else:
            optimizer = SGD(model.parameters(), lr=float(config.training.learning_rate.lr),
                            momentum=float(config.training.momentum),
                            weight_decay=config.training.weight_decay)
            return optimizer

    elif config.training.optimizer == 'Adam':
        if config.training.LLDR.enable:
            optimizer_patch_embedding = Adam(model.patch_embeddings.parameters(),
                                             lr=float(config.training.LLDR.max_lr_patch),
                                             weight_decay=config.training.LLDR.weight_decay_patch)
            optimizer_encoder = Adam(model.encoder.parameters(),
                                     lr=float(config.training.LLDR.max_lr_encoder),
                                     weight_decay=config.training.LLDR.weight_decay_encoder)
            optimizer_mlp = Adam(model.mlp_head.parameters(),
                                 lr=float(config.training.LLDR.max_lr_mlphead),
                                 weight_decay=config.training.LLDR.weight_decay_mlp)

            return optimizer_patch_embedding, optimizer_encoder, optimizer_mlp

        else:
            optimizer = Adam(model.parameters(), lr=float(config.training.learning_rate.lr),
                             weight_decay=config.training.weight_decay)
            return optimizer

    elif config.training.optimizer == 'AdamW':
        if config.training.LLDR.enable:
            optimizer_patch_embedding = AdamW(model.patch_embeddings.parameters(),
                                              lr=float(config.training.LLDR.max_lr_patch),
                                              weight_decay=config.training.LLDR.weight_decay_patch)
            optimizer_encoder = AdamW(model.encoder.parameters(),
                                      lr=float(config.training.LLDR.max_lr_encoder),
                                      weight_decay=config.training.LLDR.weight_decay_encoder)
            optimizer_mlp = AdamW(model.mlp_head.parameters(),
                                  lr=float(config.training.LLDR.max_lr_mlphead),
                                  weight_decay=config.training.LLDR.weight_decay_mlp)

            return optimizer_patch_embedding, optimizer_encoder, optimizer_mlp

        else:
            optimizer = AdamW(model.parameters(), lr=float(config.training.learning_rate.lr),
                              weight_decay=config.training.weight_decay)

            return optimizer


    else:
        raise ValueError('Only Adam and sgd optimizers are supported.')


def prepare_scheduler(config, optimizer_patch_embedding, optimizer_encoder=None, optimizer_mlp=None):
    if config.training.LLDR.enable:
        # if config.data.dataset == 'aadb':
        #     train_path = os.path.join(config.path.img_folder, 'train')
        # else:

        cycle_mult = config.training.LLDR.cosine_scheduler_patch.cycle_mult
        min_lr = config.training.LLDR.cosine_scheduler_patch.min_lr
        warmup = config.training.LLDR.cosine_scheduler_patch.warm_up
        gamma = config.training.LLDR.cosine_scheduler_patch.gamma

        scheduler_patch = CosineAnnealingWarmupRestarts(optimizer_patch_embedding,
                                                        first_cycle_steps=config.training.num_epochs,
                                                        cycle_mult=cycle_mult,
                                                        max_lr=float(config.training.LLDR.max_lr_patch),
                                                        min_lr=min_lr,
                                                        warmup_steps=warmup,
                                                        gamma=gamma)

        cycle_mult = config.training.LLDR.cosine_scheduler_encoder.cycle_mult
        min_lr = config.training.LLDR.cosine_scheduler_encoder.min_lr
        warmup = config.training.LLDR.cosine_scheduler_encoder.warm_up
        gamma = config.training.LLDR.cosine_scheduler_encoder.gamma

        scheduler_encoder = CosineAnnealingWarmupRestarts(optimizer_encoder,
                                                          first_cycle_steps=config.training.num_epochs,
                                                          cycle_mult=cycle_mult,
                                                          max_lr=float(config.training.LLDR.max_lr_encoder),
                                                          min_lr=min_lr,
                                                          warmup_steps=warmup,
                                                          gamma=gamma)

        cycle_mult = config.training.LLDR.cosine_scheduler_mlp.cycle_mult
        min_lr = config.training.LLDR.cosine_scheduler_mlp.min_lr
        warmup = config.training.LLDR.cosine_scheduler_mlp.warm_up
        gamma = config.training.LLDR.cosine_scheduler_mlp.gamma

        scheduler_mlp = CosineAnnealingWarmupRestarts(optimizer_mlp,
                                                      first_cycle_steps=config.training.num_epochs,
                                                      cycle_mult=cycle_mult,
                                                      max_lr=float(config.training.LLDR.max_lr_mlphead),
                                                      min_lr=min_lr,
                                                      warmup_steps=warmup,
                                                      gamma=gamma)

        return scheduler_patch, scheduler_encoder, scheduler_mlp
    else:
        if config.training.learning_rate.scheduler.type == 'cosine':
            cycle_mult = config.training.learning_rate.scheduler.cosine.cycle_mult
            min_lr = config.training.learning_rate.scheduler.cosine.min_lr
            warmup = config.training.learning_rate.scheduler.cosine.warm_up
            gamma = config.training.learning_rate.scheduler.cosine.gamma

            # if config.data.dataset == 'aadb':
            #     train_path = os.path.join(config.path.img_folder, 'train')
            # else:
            train_path = config.path.img_folder

            num_train_samples = len(os.listdir(train_path)) * config.training.train_size
            steps_per_epoch = (num_train_samples // config.training.batch_size) + (
                    num_train_samples % config.training.batch_size > 0)
            warmup_steps = int(steps_per_epoch * warmup)
            first_cycle_steps = int(steps_per_epoch * config.training.num_epochs)

            # https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
            scheduler = CosineAnnealingWarmupRestarts(optimizer_patch_embedding,
                                                      first_cycle_steps=config.training.num_epochs,
                                                      cycle_mult=cycle_mult,
                                                      max_lr=float(config.training.learning_rate.lr),
                                                      min_lr=min_lr,
                                                      warmup_steps=warmup,
                                                      gamma=gamma)
            return scheduler

        elif config.training.learning_rate.scheduler.type == 'step_lr':
            milestones = config.training.learning_rate.scheduler.step_lr.milestones
            gamma = config.training.learning_rate.scheduler.step_lr.gamma
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer_patch_embedding, milestones=milestones, gamma=gamma)

            return scheduler
        else:
            raise ValueError("Only cosine annealing and step lr schedulers are supported. ")
