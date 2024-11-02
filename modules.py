import torch
import torch.nn as nn
import torch.nn.functional as F
from quantise import VectorQuantiser  # 导入向量量化模块

# 定义残差块
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        # 定义残差块的结构
        self._block = nn.Sequential(
            nn.ReLU(True),  # 激活函数
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(True),  # 激活函数
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1)  # 1x1卷积层
        )
    def forward(self, x):
        return x + self._block(x)  # 残差连接

# 定义残差堆叠模块
class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers  # 残差层数量
        # 创建多个 Residual 块
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)  # 依次通过每个残差层
        return F.relu(x)  # 返回激活后的结果

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        # 定义卷积层
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        # 添加残差堆叠
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        # 前向传播
        x = self._conv_1(inputs)  # 通过第一个卷积层
        x = F.relu(x)  # 激活
        x = self._conv_2(x)  # 通过第二个卷积层
        x = F.relu(x)  # 激活
        x = self._conv_3(x)  # 通过第三个卷积层
        return self._residual_stack(x)  # 通过残差堆叠

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, output_channels):
        super(Decoder, self).__init__()
        # 定义解码器的卷积层
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        # 添加残差堆叠
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        # 转置卷积层用于上采样
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=output_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        # 前向传播
        x = self._conv_1(inputs)  # 通过卷积层
        x = self._residual_stack(x)  # 通过残差堆叠
        x = self._conv_trans_1(x)  # 上采样
        x = F.relu(x)  # 激活
        return self._conv_trans_2(x)  # 返回最终输出

# 定义模型
class Model(nn.Module):
    def __init__(self, input_dim, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost=0.25, distance='l2', 
                 anchor='closest', first_batch=False, contras_loss=True):
        super(Model, self).__init__()
        # 初始化编码器
        self._encoder = Encoder(input_dim, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        # 预处理卷积层
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        # 向量量化模块
        self._vq_vae = VectorQuantiser(num_embeddings, embedding_dim, commitment_cost, distance=distance, 
                                       anchor=anchor, first_batch=first_batch, contras_loss=contras_loss)
        # 初始化解码器
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens,
                                input_dim)

    def encode(self, x):
        # 编码过程
        z_e_x = self._encoder(x)  # 通过编码器
        z_e_x = self._pre_vq_conv(z_e_x)  # 通过预处理卷积层
        loss, quantized, perplexity, _ = self._vq_vae(z_e_x)  # 向量量化
        return loss, quantized, perplexity  # 返回损失、量化结果和困惑度

    def forward(self, x):
        # 前向传播过程
        z = self._encoder(x)  # 通过编码器
        z = self._pre_vq_conv(z)  # 通过预处理卷积层
        quantized, loss, (perplexity, encodings, _) = self._vq_vae(z)  # 向量量化
        x_recon = self._decoder(quantized)  # 通过解码器重建输入
        return x_recon, loss, perplexity, encodings  # 返回重建结果、损失、困惑度和编码
