"""Main script for Diffusion-TTA"""
import os
# 设置cuda可见
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["WANDB_API_KEY"] = 'a4905e045d361b6bba94c0cbe0be213eb13ebb5a'
import copy
import random
import warnings

import wandb
# os.environ["WANDB_API_KEY"] = 'a4905e045d361b6bba94c0cbe0be213eb13ebb5a'
# os.environ['WANDB_DISABLED'] = 'true'
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict
from mergedeep import merge
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True
from copy import deepcopy
from dataset.catalog import DatasetCatalog
from diff_tta import utils, engine
from diff_tta.vis_utils import (
    visualize_classification_with_image,
    visualize_diffusion_loss,
    visualize_classification_improvements,
)
from diff_tta.models import build
# 归一化函数
mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()
def normalize_image(image_tensor):
    normalized_image = (image_tensor - mean) / std
    return normalized_image

# 反归一化函数
def denormalize_image(image_tensor):
    denormalized_image = image_tensor * std + mean
    # denormalized_image = torch.clamp(denormalized_image, 0, 1)
    return denormalized_image

def logit_to_entropy(logits):
    """Convert logits to probabilities and compute entropy."""
    # 应用softmax函数将logit转换为概率分布
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    # 计算熵
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value
import torch.nn as nn

import numpy as np


def softmax(logits):
    """计算softmax概率"""
    exp_logits = np.exp(logits - np.max(logits))  # 减去最大值以防止数值溢出
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def kl_divergence(logits1, logits2):
    """计算两个logit数组的KL散度"""
    # 计算softmax概率
    probs1 = softmax(logits1)
    probs2 = softmax(logits2)

    # 使用公式计算KL散度
    # KL(P || Q) = Σ P(i) * log(P(i) / Q(i))
    kl_div = np.sum(probs1 * np.log(probs1 / probs2), axis=-1)
    return kl_div


# # 假设logits1和logits2是两个numpy数组，包含了分类网络的预测logit
# logits1 = np.array([3.0, 1.0, 0.2])
# logits2 = np.array([2.0, 1.5, 0.3])
#
# # 计算KL散度
# kl_div = kl_divergence(logits1, logits2)
#
# # 输出结果
# print("KL Divergence:", kl_div)
criterion = nn.CrossEntropyLoss()
def tta_one_epoch(config, dataloader, tta_model, optimizer, scaler, autoencoder, image_renormalizer):

    """Perform test time adaptation over the entire dataset.

    Args:
        config: configuration object for hyper-parameters.
        dataloader: The dataloader for the dataset.
        tta_model: A test-time adaptation wrapper model.
        optimizer: A gradient-descent optimizer for updating classifier.
        scaler: A gradient scaler used jointly with optimizer.
        autoencoder: A pre-trained autoencoder model (e.g. VQVAE).
        image_renormalizer: An object for renormalizing images.
    """

    cwd = config.cwd
    discrete_sampling_accuracy = []
    dataloader, one_data = dataloader

    tta_model.eval()

    # Keep a copy of the original model state dict, so that we can reset the
    # model after each image
    tta_class_state_dict = copy.deepcopy(tta_model.state_dict())

    # Enlarge batch size by accumulating gradients over multiple iterations
    config.tta.gradient_descent.train_steps = (
        config.tta.gradient_descent.train_steps
        * config.tta.gradient_descent.accum_iter
    )

    # Start iterations
    start_index = 0
    last_index = len(dataloader.dataset)
    from purifier import get_purifier, re_runer
    purifier = get_purifier()
    purifier.cuda().eval()
    import matplotlib.pyplot as plt
    before_s = []
    after_s = []

    rr = re_runer()
    all_save_entropy = []
    for img_ind in range(start_index, last_index):
        # Enable/disable to upload visualization to wandb
        visualize = (
            (config.log_freq > 0 and img_ind % config.log_freq == 0)
            or img_ind == last_index - 1
        )

        # The dictionary for visualization
        wandb_dict = {}

        # Fetch data from the dataset
        print(f"\n\n Example: {img_ind}/{last_index} \n\n")
        batch = dataloader.dataset[img_ind]
        batch = engine.preprocess_input(batch, config.gpu)

        # We will classify before and after test-time adaptation via
        # gradient descent. We run tta_model.evaluate(batch, after_tta=True) to
        # save the classification results

        # Step 1: Predict pre-TTA classification. The results are saved in
        # `before_tta_stats_dict` and `tta_model.before_tta_acc`
        before_tta_stats_dict = tta_model.evaluate(batch, before_tta=True)
        # if int(after_tta_stats_dict['after_tta_correct']) :
        #     continue
        # 修改这部分，把tta改成robust 图像去噪
        # 核心在于如何去噪，
        """去噪算法----begin"""
        image = batch["test_image_disc"]  # different cls label input to the purifier and classifier ,then conditional image compute the entropy of each class
        ori_image = deepcopy(image)
        tmp_image = deepcopy(ori_image)
        tmp_image = denormalize_image(tmp_image)
        # plt.imshow(image[0].cpu().numpy().transpose(1, 2, 0))
        # plt.savefig('2.png')
        # y = batch['class_idx']
        resized_image = torch.nn.functional.interpolate(tmp_image, size=(256, 256), mode='bilinear')
        save_y_entropy = []
        before_logit = (before_tta_stats_dict['before_tta_logits']).numpy()
        sorted_list = np.argsort(before_logit)[0, ::-1]
        # 从logit中获取概率最大的类别
        plt.imshow(resized_image[0].cpu().numpy().transpose(1, 2, 0))
        plt.savefig(f'tmp1/ori_{img_ind}.png')
        for y in sorted_list[:10]:
            y = torch.tensor([y]).cuda()
            rr_i = (rr.image_editing_sample((resized_image.cuda()-0.5)*2, y)).detach()
            # purifier_image = purifier(resized_image)
            purifier_image = torch.nn.functional.interpolate(rr_i, size=(image.shape[-2], image.shape[-1]), mode='bilinear')
            purifier_image = purifier_image / 2 + 0.5
            plt.imshow(purifier_image[0].cpu().numpy().transpose(1, 2, 0))
            plt.savefig(f'tmp1/pur_{img_ind}_{y}.png')
            purifier_image = normalize_image(purifier_image)

            # purifier_image =
            diff_rate = []
            for ii in [1.0]:

                batch['test_image_disc'] = ori_image * (1-ii) + ii*purifier_image
                """去噪算法----end"""
                # print(image.shape)

                # Step 2: TTA by gradient descent
                # losses, after_tta_outputs = engine.tta_one_image_by_gradient_descent(
                #     batch, tta_model, optimizer, scaler,
                #     autoencoder, image_renormalizer, config,
                #     before_tta_stats_dict['pred_topk_idx']
                # )

                # Step 3: Predict post-TTA classification. The results are saved in
                # `after_tta_stats_dict` and `tta_model.after_tta_acc`
                after_tta_stats_dict = tta_model.evaluate(batch, after_tta=True)
                after_logit = (after_tta_stats_dict['after_tta_logits']).numpy()

                # compute the KL div

                diff_rate.append([ii, logit_to_entropy(after_logit), after_logit])

            # bf = int(before_tta_stats_dict['before_tta_correct'])
            # af = int(after_tta_stats_dict['after_tta_correct'])
            # before_s.append(bf)
            # after_s.append(af)
            # if int(after_tta_stats_dict['after_tta_correct']) > int(before_tta_stats_dict['before_tta_correct']):
            #     print(img_ind)
            # after_logit = (after_tta_stats_dict['after_tta_logits']).numpy()
            # before_logit = (before_tta_stats_dict['before_tta_logits']).numpy()
            save_y_entropy.append([y.cpu(), diff_rate])
            # print(logit_to_entropy(before_logit), logit_to_entropy(after_logit))
            # print(criterion(torch.from_numpy(before_logit).cpu(), y.cpu()), criterion(torch.from_numpy(after_logit).cpu(), y.cpu()))
            # print(bf, af)
        # print(before_s)
        # print(after_s)
        # before_acc = np.mean(before_s)
        # after_acc = np.mean(after_s)
        # print(f"before acc: {before_acc}, after acc: {after_acc}")
        # Reload the original model state dict
        if not config.tta.online:
            tta_model.load_state_dict(tta_class_state_dict)
            optimizer = build.load_optimizer(config, tta_model)
        all_save_entropy.append([batch['class_idx'], save_y_entropy])
        # if visualize:
        #     # wandb_dict is updated in-place
        #     wandb_dict = visualize_classification_with_image(
        #         batch, config, dataloader.dataset,
        #         before_tta_stats_dict["before_tta_logits"],
        #         before_tta_stats_dict["before_tta_topk_idx"],
        #         before_tta_stats_dict["before_tta_pred_class_idx"],
        #         before_tta_stats_dict["before_tta_topk_class_idx"],
        #         wandb_dict
        #     )

            # wandb_dict = visualize_diffusion_loss(losses, config, wandb_dict)
    # compute acc
    before_acc = np.mean(before_s)
    after_acc = np.mean(after_s)
    print(f"before acc: {before_acc}, after acc: {after_acc}")
        # Plot accuracy curve every image
        # wandb_dict = visualize_classification_improvements(
        #     tta_model.before_tta_acc, tta_model.after_tta_acc,
        #     before_tta_stats_dict["before_tta_correct"].float(),
        #     after_tta_stats_dict["after_tta_correct"].float(),
        #     wandb_dict
        # )
        #
        # # Save the results to the disck
        # wandb_run_name = wandb.run.name
        # stats_folder_name = f'stats/{wandb_run_name}/'
        # os.makedirs(stats_folder_name, exist_ok=True)

        # if config.save_results:
        #     stats_dict = {}
        #     stats_dict['accum_iter'] = config.tta.gradient_descent.accum_iter
        #     stats_dict['filename'] = batch['filepath']
        #     stats_dict['losses'] = losses
        #     stats_dict['gt_idx'] = batch['class_idx'][0]
        #     stats_dict = merge(stats_dict, before_tta_stats_dict, after_tta_stats_dict)
        #     file_index = int(batch['index'].squeeze())
        #     store_filename = f"{stats_folder_name}/{file_index:06d}.p"
        #     pickle.dump(stats_dict, open(store_filename, 'wb'))
        #
        # wandb.log(wandb_dict, step=img_ind)


def get_dataset(config):
    """Instantiate the dataset object."""
    Catalog = DatasetCatalog(config)

    dataset_dict = getattr(Catalog, config.input.dataset_name)

    target = dataset_dict['target']
    params = dataset_dict['train_params']
    if config.input.dataset_name == "ObjectNetSubsetNew":
        params.update({'use_dit': config.model.use_dit})
    dataset = utils.instantiate_from_config(dict(target=target, params=params))

    return dataset


@hydra.main(config_path="diff_tta/config", config_name="config2")
def run(config):
    with open_dict(config):
        config.log_dir = os.getcwd()
        print(f"Logging files in {config.log_dir}")
        config.cwd = get_original_cwd()
        config.gpu = None if config.gpu < 0 else config.gpu

    # Hydra automatically changes the working directory, but we stay at the
    # project directory.
    os.chdir(config.cwd)

    print(OmegaConf.to_yaml(config))

    if config.input.dataset_name == "ObjectNetDataset":
        config.input.use_objectnet = True

    if config.seed is not None:
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    run_worker(config)


def run_worker(config):

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    # wandb.init(project=config.wandb.project, config=config, mode=config.wandb.mode)

    # 读取imagenet train数据，把数据保存起来，方便调用
    tmp_config = deepcopy(config)
    tmp_config.input.dataset_name = "ImageNetOneDataset"
    tmp_dataset = get_dataset(tmp_config)
    tmp_dataloader = torch.utils.data.DataLoader(
        tmp_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )
    one_data = {}
    for bat in tqdm(tmp_dataloader):
        img = bat["test_image_disc"]
        img_idx = bat['class_idx']
        one_data[int(img_idx)] = img


    print("=> Loading dataset")
    dataset = get_dataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )

    # create model
    print("=> Creating model ")
    model, autoencoder, image_renormalizer = (
        build.create_models(config, dataset.classes, dataset.class_names)
    )
    optimizer = build.load_optimizer(config, model)
    scaler = torch.cuda.amp.GradScaler()

    tta_one_epoch(config, (dataloader, one_data), model, optimizer, scaler,
                  autoencoder, image_renormalizer)


if __name__ == '__main__':
    run()
