import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

class VectorQuantiser(nn.Module):
    """
    向量量化的改进版本，具有动态初始化未优化的“死”点。
    
    参数：
    num_embed: 代码本条目的数量
    embed_dim: 代码本条目的维度
    beta: 约束损失的权重
    distance: 查找最近代码的距离度量
    anchor: 采样锚点的方法
    first_batch: 如果为真，则为我们模型的离线版本
    contras_loss: 如果为真，则使用对比损失以进一步提高性能
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()
        self.num_embed = num_embed  # 代码本条目数量
        self.embed_dim = embed_dim  # 代码本条目维度
        self.beta = beta  # 约束损失权重
        self.distance = distance  # 距离度量方法
        self.anchor = anchor  # 采样锚点的方法
        self.first_batch = first_batch  # 是否为离线版本
        self.contras_loss = contras_loss  # 是否使用对比损失
        self.decay = 0.99  # 衰减因子
        self.init = False  # 初始化标志
        self.pool = FeaturePool(self.num_embed, self.embed_dim)  # 特征池
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)  # 嵌入层
        # 初始化嵌入权重
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))  # 嵌入概率缓冲区

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # 确保与Gumbel接口兼容
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        
        # 重塑 z -> (batch, height, width, channel) 并展平
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)  # 展平为二维张量
        
        # 计算距离
        if self.distance == 'l2':
            # 计算 L2 距离
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # 计算余弦距离
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()  # 对 z 进行归一化
            normed_codebook = F.normalize(self.embedding.weight, dim=1)  # 对代码本进行归一化
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))
        
        # 编码
        sort_distance, indices = d.sort(dim=1)  # 按距离排序
        # 查找最近点的索引
        encoding_indices = indices[:, -1]  # 取最近的索引
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)  # 初始化编码张量
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)  # 将最近的编码设置为1
        
        # 量化并反展平
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)  # 量化后的输出
        
        # 计算嵌入损失
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        # 保持梯度
        z_q = z + (z_q - z).detach()  # 保持 z 的梯度
        # 重新调整形状以匹配原始输入形状
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        
        # 计算平均概率和困惑度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  # 计算困惑度
        min_encodings = encodings  # 最小编码
        
        # 在线聚类重新初始化未优化的点
        if self.training:
            # 计算代码条目的平均使用情况
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)  # 更新嵌入概率
            # 基于平均使用情况的衰减
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # 最近采样
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1, :]]  # 获取最近的特征
                # 基于特征池的随机采样
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # 基于概率的随机采样
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]  # 随机选择特征
                
                # 更新嵌入权重
                decay = torch.exp(-(self.embed_prob * self.num_embed * 10) / (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True  # 如果是第一批，设置初始化标志为真
            
            # 对比损失
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.num_embed)):, :].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0) * 1 / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07  # 计算对比损失
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss += contra_loss  # 将对比损失加入总损失
        
        return z_q, loss, (perplexity, min_encodings, encoding_indices)  # 返回量化结果、损失、困惑度和编码索引


class FeaturePool():
    """
    该类实现了一个特征缓冲区，用于存储先前编码的特征。
    此缓冲区使我们能够使用历史生成特征初始化代码本，而不是使用最新编码器生成的特征。
    """
    def __init__(self, pool_size, dim=64):
        """
        初始化 FeaturePool 类
        
        参数：
        pool_size(int) -- 特征缓冲区的大小
        """
        self.pool_size = pool_size  # 设置缓冲区大小
        if self.pool_size > 0:
            self.nums_features = 0  # 当前特征数量
            self.features = (torch.rand((pool_size, dim)) * 2 - 1) / pool_size  # 初始化特征缓冲区

    def query(self, features):
        """
        从池中返回特征
        """
        self.features = self.features.to(features.device)  # 将特征移动到相同的设备
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size:  # 如果批量大小足够大，直接更新整个代码本
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]  # 随机选择特征
                self.nums_features = self.pool_size  # 更新特征数量
            else:
                # 如果批量大小不足，则仅存储以便下次更新
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features  # 存储特征
                self.nums_features = num  # 更新特征数量
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]  # 随机选择特征
            else:
                random_id = torch.randperm(self.pool_size)  # 随机打乱索引
                self.features[random_id[:features.size(0)]] = features  # 更新特征池
        return self.features  # 返回特征池中的特征
