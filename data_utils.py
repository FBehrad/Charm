from PIL import Image, ImageDraw, ImageFile
import torch
import torch.nn as nn
import math
import random
from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True


def add_padding(img, patch_size):
    channels, height, width = img.size()

    if width % patch_size != 0:
        padding_width = (patch_size - (width % patch_size)) % patch_size
    else:
        padding_width = 0

    if height % patch_size != 0:
        padding_height = (patch_size - (height % patch_size)) % patch_size
    else:
        padding_height = 0

    padding = torch.nn.ZeroPad2d((0, padding_width, 0, padding_height))
    padded_img = padding(img)

    return padded_img


def image_to_patches(image, patch_size, stride=None):
    channels, height, width = image.size()
    if stride is None:
        stride = patch_size

    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[:, y:y + patch_size, x:x + patch_size]
            patches.append(patch)
        if width % patch_size != 0:
            # Add remaining parts along the right edge
            x = width - patch_size
            remaining_patch = image[:, y:y + patch_size, x:width]
            patches.append(remaining_patch)

    return patches

def mask_to_patches(mask, patch_size, stride=None):
    height, width = mask.size()
    if stride is None:
        stride = patch_size

    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = mask[y:y+patch_size, x:x+patch_size]
            if torch.all(patch == 0):
                patches.append(0)
            else:
                patches.append(1)
        if width % patch_size != 0:
            # Add remaining parts along the right edge
            x = width - patch_size
            remaining_patch = mask[:, :, y:y+patch_size, x:width]
            if torch.all(remaining_patch == 0):
                patches.append(0)
            else:
                patches.append(1)
    return patches


def visualize_patches(patches, patch_size, margin):
    # Calculate the size of the output image
    output_width = len(patches) * (patch_size + margin) - margin
    output_height = patch_size

    # Create a blank image for visualization
    output_image = Image.new("RGB", (output_width, output_height), color="white")
    draw = ImageDraw.Draw(output_image)

    # Paste each patch into the output image with a margin
    current_x = 0
    transform = transforms.ToPILImage()
    for patch in patches:
        patch = transform(patch)
        output_image.paste(patch, (current_x, 0))
        current_x += patch_size + margin

    return output_image


def reconstruct_image(patches, image_size, patch_size):
    patches_per_row = (image_size[0] + patch_size - 1) // patch_size
    num_rows = (len(patches) + patches_per_row - 1) // patches_per_row

    reconstructed_image = Image.new("RGB", image_size, color="white")
    for row in range(num_rows):
        for col in range(patches_per_row):
            patch_index = row * patches_per_row + col

            if patch_index < len(patches):
                patch = patches[patch_index]
                x = col * patch_size
                y = row * patch_size
                reconstructed_image.paste(patch, (x, y))
    return reconstructed_image


def select_patches(image_patches, depth_patches, num_selected_patches):
    mean_values = torch.stack([torch.mean(patch) for patch in depth_patches])
    min_value = torch.min(mean_values)
    max_value = torch.max(mean_values)
    normalized_tensor = (mean_values - min_value) / (max_value - min_value)

    patch_prob_index_tuples = list(enumerate(zip(image_patches, normalized_tensor)))
    sorted_patches = sorted(patch_prob_index_tuples, key=lambda x: x[1][1])
    # situation 1: random based on the probabilities
    # selected_patch_indices = sorted(random.sample(sorted_patches, num_selected_patches))

    # situation 2: keep top probabilities + random based on probabilities
    # idx = -num_selected_patches // 2
    # top_half_patches = sorted_patches[idx:]
    # remaining = num_selected_patches - len(top_half_patches)
    # remaining_patches = sorted(random.sample(sorted_patches[:idx], remaining))
    # selected_patch_indices = top_half_patches + remaining_patches

    # situation 3: keep top probabilities
    idx = -num_selected_patches
    top_half_patches = sorted_patches[idx:]
    selected_patch_indices = top_half_patches

    selected_patches = [patch for _, (patch, _) in selected_patch_indices]
    selected_indices = [index for index, _ in selected_patch_indices]

    return selected_patches, selected_indices


def create_binary_mask(image_size, patch_size, selected_patch_indices):
    channels, height, width = image_size
    start_index = torch.tensor(sorted(selected_patch_indices)) * patch_size
    row_num = start_index // width
    row_start = row_num * patch_size
    row_end = (row_num + 1) * patch_size

    column_first = start_index % width
    column_end = column_first + patch_size

    num_patches = len(row_start)
    mask = torch.zeros((height, width), dtype=torch.bool)

    for i in range(num_patches):
        row_indices = torch.arange(row_start[i], row_end[i]).unsqueeze(1)
        col_indices = torch.arange(column_first[i], column_end[i])
        mask[row_indices, col_indices] = True
    return mask


def interpolate_positional_embedding(pos_embed, size,interpolate_offset, dim):
    w0, h0 = size
    w0, h0 = w0 + interpolate_offset, h0 + interpolate_offset
    N = pos_embed.shape[1] - 1
    pos_embed = pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    sqrt_N = math.sqrt(N)
    sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
        scale_factor=(sx, sy),
        mode="bicubic",
        align_corners=False,
    )
    assert int(w0) == patch_pos_embed.shape[-2]
    assert int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed, class_pos_embed


def padding(x, max_seq_len):
    n_crops, c, W, H = x.size()
    padding = torch.zeros(max_seq_len, c, W, H, dtype=x.dtype, device=x.device)
    x = torch.cat([x, padding], dim=0)
    x = x[:max_seq_len, :, :, :]
    return x


def random_flip(image, mask):
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if mask is not None:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask


def random_rotate(image, mask):
    angles = [90, 180, 270]
    angle = random.choice(angles)
    rotated_image = image.rotate(angle, expand=True)
    if mask is not None:
        rotated_mask = mask.rotate(angle, expand=True)
    else:
        rotated_mask = None
    return rotated_image, rotated_mask


def gray_scale(image):
    gray_image = image.convert('L')
    gray_image = gray_image.convert('RGB')
    return gray_image



def lcm(arr):
    l=reduce(lambda x,y:(x*y)//math.gcd(x,y),arr)
    return l


def cropping(image, target_size):
    channels, height, width = image.shape
    crop_width = width % target_size
    crop_height = height % target_size

    end_x = width - crop_width
    end_y = height - crop_height

    cropped_image = image[:, :end_y, :end_x]

    return cropped_image


def pad_or_crop(image, size):
    channels, height, width = image.shape

    crop_width = width % size
    crop_height = height % size
    if crop_height == 0:
        crop_height = 1
    if crop_width == 0:
        crop_width = 1
    total_pixel_crop = crop_height * crop_width

    if width % size != 0:
        padding_width = (size - (width % size)) % size
    else:
        padding_width = 1
    if height % size != 0:
        padding_height = (size - (height % size)) % size
    else:
        padding_height = 1
    total_pixel_pad = padding_height * padding_width

    if total_pixel_pad > total_pixel_crop:
        new_image = cropping(image, size)
    else:
        new_image = add_padding(image, size)

    return new_image


def visualize(image_tensor):
    image_numpy = (image_tensor.numpy().transpose((1, 2, 0)) + 1) / 2
    plt.imshow(image_numpy)
    plt.show()


def calculate_blur_metric(patch):
    # ref: https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
    fft_image = torch.fft.fft2(patch)
    fft_shifted = torch.fft.fftshift(fft_image)
    magnitude_spectrum = 20 * torch.log10(torch.abs(fft_shifted))
    blur_metric = torch.mean(magnitude_spectrum)
    return blur_metric


def calculate_and_sort_blur_metrics(image_tensors):
    blur_metrics_with_indices = []
    for i, image in enumerate(image_tensors):
        blur_metric = calculate_blur_metric(image)
        # if torch.isfinite(blur_metric):
        blur_metrics_with_indices.append((blur_metric, i))

    blur_metrics_with_indices.sort(key=lambda x: x[0])
    return blur_metrics_with_indices


def calculate_frequency(image_patches):
    frequencies = calculate_and_sort_blur_metrics(image_patches)
    return frequencies


def calculate_entropy(image_tensors):
    entropy_with_indices = []
    for i, image in enumerate(image_tensors):
        flattened_image = image.reshape(-1)
        hist = torch.histc(flattened_image, bins=256, min=0, max=1)
        hist = hist / hist.sum()
        entropy = -torch.sum(hist * torch.log2(hist + 1e-9))
        entropy_with_indices.append((entropy, i))

    entropy_with_indices.sort(key=lambda x: x[0])
    return entropy_with_indices


def patch_selection_entropy_based(entropies, n_patches, scale, num_scales):
    if scale == num_scales - 1:
        bluriness, indices = zip(*entropies[-2 * n_patches:])
    else:
        bluriness, indices = zip(*entropies[-4 * n_patches:-2 * n_patches])
    selected_indices = random.sample(indices, n_patches)  # To add stochasticity
    return selected_indices


def patch_selection_frequency_based(frequencies, n_patches, scale, num_scales):
    if scale == num_scales - 1:
        bluriness, indices = zip(*frequencies[-2 * n_patches:])
    else:
        bluriness, indices = zip(*frequencies[-4 * n_patches:-2 * n_patches])
    selected_indices = random.sample(indices, n_patches)  # To add stochasticity
    return selected_indices


def calculate_gradient(image_tensor):
    if image_tensor.shape[0] == 3:
        image_tensor = image_tensor.mean(dim=0, keepdim=True)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(image_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(image_tensor, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return torch.sum(torch.abs(grad_mag))


def calculate_gradients(image_tensors):
    gradients_with_indices = []
    for i, image in enumerate(image_tensors):
        gradient = calculate_gradient(image)
        gradients_with_indices.append((gradient, i))

    gradients_with_indices.sort(key=lambda x: x[0])
    return gradients_with_indices


def patch_selection_gradient(gradients, n_patches, scale, num_scales):
    if scale == num_scales - 1:
        grads, indices = zip(*gradients[-2 * n_patches:])
    else:
        grads, indices = zip(*gradients[-4 * n_patches:-2 * n_patches])
    selected_indices = random.sample(indices, n_patches)  # To add stochasticity
    return selected_indices


def patch_selection_saliency(salient_indices, n_patches):
    selected_indices = random.sample(salient_indices, n_patches)
    return selected_indices


def draw_red_boundaries_on_patches(image, patch_indices, patch_size):
    # image : PIL image
    image_array = np.array(image)
    height, width, channels = image_array.shape

    for index in patch_indices:
        x, y = index % (width // patch_size), index // (width // patch_size)
        start_x, start_y = x * patch_size, y * patch_size
        end_x, end_y = start_x + patch_size, start_y + patch_size

        patch = image_array[start_y:end_y, start_x:end_x, :]

        border_width = 3

        # Top and bottom borders
        patch[:border_width, :, :] = [255, 0, 0]
        patch[-border_width:, :, :] = [255, 0, 0]

        # Left and right borders
        patch[:, :border_width, :] = [255, 0, 0]
        patch[:, -border_width:, :] = [255, 0, 0]

        # Replace the patch in the image array
        image_array[start_y:end_y, start_x:end_x, :] = patch

    output_image = Image.fromarray(image_array)
    return output_image
