"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of examples.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
examples that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
from inception import InceptionV3
from image_folder import make_dataset

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有安装 tqdm，提供一个模拟版本
    def tqdm(x): return x

# 创建命令行参数解析器
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')  # 批处理大小
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))  # Inception特征的维度
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')  # 使用的GPU设备
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated examples or '
                          'to .npz statistic files'))  # 输入路径

def imread(filename):
    """
    将图像文件加载为 (height, width, 3) 的 uint8 ndarray，并调整大小为 (229, 229)
    """
    return np.asarray(Image.open(filename).convert('RGB').resize((229, 229), Image.BILINEAR), dtype=np.uint8)[..., :3]

def get_activations(files, model, batch_size=50, dims=2048, cuda=False):
    """计算所有示例的 pool_3 层的激活值。
    
    参数：
    -- files       : 图像文件路径列表
    -- model       : Inception 模型实例
    -- batch_size  : 模型一次处理的示例批量大小。
                     确保样本数量是批量大小的倍数，否则一些样本将被忽略。
    -- dims        : Inception 返回的特征维度
    -- cuda        : 如果设置为 True，则使用 GPU
    
    返回：
    -- 一个 numpy 数组，维度为 (num examples, dims)，包含
       在将查询张量传递给 Inception 时的激活值。
    """
    model.eval()  # 设置模型为评估模式
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))  # 如果批量大小大于数据大小，调整批量大小
        batch_size = len(files)
    pred_arr = np.empty((len(files), dims))  # 初始化预测数组
    for i in tqdm(range(0, len(files), batch_size)):
        start = i
        end = i + batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])  # 读取图像并转换为浮点数
        # 重塑为 (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255  # 将像素值归一化到 [0, 1]
        batch = torch.from_numpy(images).type(torch.FloatTensor)  # 转换为 PyTorch 张量
        if cuda:
            batch = batch.cuda()  # 如果使用GPU，将张量移到GPU
        pred = model(batch)[0]  # 通过模型获取激活值
        # 如果模型输出不是标量，则应用全局空间平均池化。
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))  # 进行池化
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)  # 将结果存入数组
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算 Frechet 距离的 Numpy 实现。
    
    Frechet 距离用于比较两个多元高斯分布 X_1 ~ N(mu_1, C_1)
    和 X_2 ~ N(mu_2, C_2)：
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    
    稳定版本由 Dougal J. Sutherland 提供。
    
    参数：
    -- mu1   : 包含生成样本的 Inception 网络激活的 Numpy 数组。
    -- mu2   : 在代表性数据集上预计算的激活样本均值。
    -- sigma1: 生成样本的激活的协方差矩阵。
    -- sigma2: 在代表性数据集上预计算的激活的协方差矩阵。
    
    返回：
    --   : Frechet 距离。
    """
    mu1 = np.atleast_1d(mu1)  # 确保 mu1 是一维数组
    mu2 = np.atleast_1d(mu2)  # 确保 mu2 是一维数组
    sigma1 = np.atleast_2d(sigma1)  # 确保 sigma1 是二维数组
    sigma2 = np.atleast_2d(sigma2)  # 确保 sigma2 是二维数组
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'  # 确保均值向量长度相同
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'  # 确保协方差矩阵维度相同
    diff = mu1 - mu2  # 计算均值差
    # 计算协方差矩阵的平方根
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():  # 检查协方差矩阵是否有限
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps  # 添加小的偏移量以确保数值稳定性
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # 检查协方差矩阵是否有虚部
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))  # 抛出错误
        covmean = covmean.real  # 取实部
    tr_covmean = np.trace(covmean)  # 计算协方差矩阵的迹
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)  # 返回 Frechet 距离

def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    cuda=False):
    """计算用于 FID 的统计信息。
    
    参数：
    -- files       : 图像文件路径列表
    -- model       : Inception 模型实例
    -- batch_size  : 示例 numpy 数组分成的批次大小。
                     合理的批量大小取决于硬件。
    -- dims        : Inception 返回的特征维度
    -- cuda        : 如果设置为 True，则使用 GPU
    
    返回：
    -- mu    : Inception 模型 pool_3 层激活的均值。
    -- sigma : Inception 模型 pool_3 层激活的协方差矩阵。
    """
    act = get_activations(files, model, batch_size, dims, cuda)  # 获取激活值
    mu = np.mean(act, axis=0)  # 计算均值
    sigma = np.cov(act, rowvar=False)  # 计算协方差矩阵
    return mu, sigma

def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    """根据路径计算统计信息。
    
    参数：
    -- path       : 文件路径
    -- model      : Inception 模型实例
    -- batch_size : 批量大小
    -- dims       : 特征维度
    -- cuda       : 是否使用 GPU
    
    返回：
    -- mu    : 均值
    -- sigma : 协方差矩阵
    """
    if path.endswith('.npz'):
        f = np.load(path)  # 加载 .npz 文件
        m, s = f['mu'][:], f['sigma'][:]  # 提取均值和协方差
        f.close()
    elif path.endswith('.txt'):
        files, file_size = make_dataset(path)  # 从文本文件生成文件路径
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)  # 计算统计
    else:
        path = pathlib.Path(path)  # 将路径转换为 Path 对象
        files = list(path.glob('*.jpg')) + list(path.glob('*.png')) + \
                list(path.glob('*.JPEG')) + list(path.glob('*.jpeg'))  # 获取所有图像文件
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)  # 计算统计
    return m, s  # 返回均值和协方差矩阵

def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """计算两个路径之间的 FID 值。
    
    参数：
    -- paths      : 文件路径列表
    -- batch_size : 批量大小
    -- cuda       : 是否使用 GPU
    -- dims       : 特征维度
    
    返回：
    -- fid_value  : 计算得到的 FID 值
    """
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)  # 检查路径有效性
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]  # 获取特征维度对应的块索引
    model = InceptionV3([block_idx])  # 创建 Inception 模型实例
    if cuda:
        model.cuda()  # 如果使用GPU，将模型移到GPU
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda)  # 计算第一个路径的统计信息
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda)  # 计算第二个路径的统计信息
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)  # 计算 FID 值
    return fid_value  # 返回 FID 值

def main():
    args = parser.parse_args()  # 解析命令行参数
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 设置可见的 GPU 设备
    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims)  # 计算 FID 值
    print('FID: ', fid_value)  # 打印 FID 值

if __name__ == '__main__':
    main()  # 运行主函数
