# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroSoftmax(nn.Module):

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x



class SelfAttention(nn.Module):

    def __init__(self, in_dims=128, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()

        self.norm = nn.LayerNorm(in_dims) 

        self.embedding = nn.Linear(in_dims, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.scaled_factor = torch.sqrt(torch.Tensor([d_model // num_heads])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(0.1)

        self.num_heads = num_heads

    def split_heads(self, x):

        # x [batch_size seq_len d_model]

        x = x.reshape(*x.shape[:-1], self.num_heads, x.shape[-1] // self.num_heads).contiguous()

        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]

    def forward(self, x, mask=False, multi_head=False):

        #  x:(256*N*128)
        x = self.norm(x)
        embeddings = self.embedding(x)  #temp:20*N*64
        query = self.query(embeddings)  #temp:20*N*64
        key = self.key(embeddings)  #temp:20*N*64

        if multi_head:
            query = self.split_heads(query)  # 256 num_heads N d_model
            key = self.split_heads(key)  # 256 num_heads N d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)

        attention = self.softmax(attention / self.scaled_factor)
        attention = self.attn_dropout(attention)

        if mask is True:
            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)
        # Modifications added: 1. scaled_factor divides by an extra num_heads. 2. attention adds a dropout of 0.1
        return attention



class SpatialTemporalFusion(nn.Module):

    def __init__(self, obs_len=8):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        # 20*4*9*6*6
        shape = x.size()
        x = x.reshape(-1, shape[2], shape[3], shape[4])
        x = self.conv(x) + self.shortcut(x)
        return x.reshape(shape)


class InteractionMask(nn.Module):

    def __init__(self, number_asymmetric_conv_layer=7, temporal_channels=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_temporal_interaction, threshold=0.5):

        for j in range(self.number_asymmetric_conv_layer):
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        temporal_zero = torch.zeros_like(temporal_interaction_mask, device='cuda')

        temporal_interaction_mask = torch.where(temporal_interaction_mask > threshold, temporal_interaction_mask,
                                                temporal_zero)

        return temporal_interaction_mask


class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x

        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        shape = x.size()
        x = x.reshape(-1, shape[1], shape[2], shape[3])
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)

        return x2.reshape(shape) + shortcut


class SparseWeightedAdjacency(nn.Module):

    def __init__(self, in_dims=128, embedding_dims=64, num_heads=4, dropout=0):
        super(SparseWeightedAdjacency, self).__init__()

        # dense interaction
        # self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims)
        self.temporal_attention = SelfAttention(in_dims, embedding_dims, num_heads=num_heads)

        # attention fusion
        # self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)

        # interaction mask
        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=7, temporal_channels=num_heads
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()

    def forward(self, graph, identity):
        temporal_graph = graph  # (256 N 128)

        # output:(256*num_heads*N*N)   (256*N*d_model)      input:256*N*128
        dense_temporal_interaction = self.temporal_attention(temporal_graph, multi_head=True)

        ts_interaction = dense_temporal_interaction

        temporal_mask = self.interaction_mask(ts_interaction)  #

        # self-connected
        # spatial_mask = spatial_mask + identity[0].unsqueeze(1).unsqueeze(0)
        temporal_mask = temporal_mask + identity.unsqueeze(0).unsqueeze(0)

        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)

        return normalized_temporal_adjacency_matrix


class SparseGraphConvolution(nn.Module):

    def __init__(self, in_dims=16, embedding_dims=16, gcn_layer_num=1, dropout=0.0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.temporal_sparse_gcn = nn.ModuleList()
        self.temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims, dropout))
        for i in range(gcn_layer_num-1):
            self.temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims, dropout))

    def forward(self, graph, normalized_temporal_adjacency_matrix):
        # graph [256 num_windows  128]
        # _matrix [batch num_heads num_windows num_windows]
        tem_graph = graph.unsqueeze(1)  # (num_p 1 seq_len 128)

        #256*T*num_head*N*16
        x = tem_graph

        for gcn in self.temporal_sparse_gcn:
            x = gcn(x, normalized_temporal_adjacency_matrix)

        return x


class GraphConvolution(nn.Module):

    def __init__(self, in_dims=128, embedding_dims=16, dropout=0.0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout

    def forward(self, graph, adjacency):
        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout, training=self.training)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]


class PreNormResMLP1(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, in_dim)

        self.norm2 = nn.LayerNorm(in_dim)
        self.fc3 = nn.Linear(in_dim, 64)
        self.out = nn.Linear(64, 1)

        self.drop = nn.Dropout(0.10)

    def forward(self, x):

        h = self.fc1(self.norm1(x))
        h = F.relu(h)
        h = self.drop(h)
        h = self.fc2(h)

        x = x + h

        x = self.fc3(self.norm2(x))
        x = F.relu(x)
        x = self.drop(x)

        return self.out(x)

class MagnitudeEstimator_mean_gate_stable(nn.Module):
    def __init__(self, num_heads=4, input_dim=32, hidden_size=128):
        super().__init__()

        # =================================================
        # Temporal-domain GRU
        # =================================================
        self.temporal_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # =================================================
        # Frequency-domain GRU
        # =================================================
        self.spec_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # =================================================
        # Attention Pooling
        # =================================================
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.spec_attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # =================================================
        # Stable Gate
        # Key optimizations:
        # 1. LayerNorm
        # 2. Small network
        # 3. bias initialized to a negative value (conservative at first)
        # =================================================
        self.gate_fc1 = nn.Linear(hidden_size * 4, 64)
        self.gate_fc2 = nn.Linear(64, hidden_size * 2)

        self.gate_norm = nn.LayerNorm(hidden_size * 4)
        self.gate_drop = nn.Dropout(0.05)

        # Initially biased toward average fusion
        nn.init.constant_(self.gate_fc2.bias, -1.0)

        self.regressor = PreNormResMLP1(hidden_size * 4)

    def forward(self, gcn_temporal_features, gcn_spec_features):

        # =================================================
        # Multi-head average
        # Input:  (B, H, T, F)
        # Output: (B, T, F)
        # =================================================
        t = gcn_temporal_features.mean(dim=1)
        s = gcn_spec_features.mean(dim=1)

        # =================================================
        # Temporal branch
        # =================================================
        t_out, _ = self.temporal_gru(t)

        tw = torch.softmax(self.temporal_attn(t_out), dim=1)
        t_context = torch.sum(tw * t_out, dim=1)

        # =================================================
        # Frequency branch
        # =================================================
        s_out, _ = self.spec_gru(s)

        sw = torch.softmax(self.spec_attn(s_out), dim=1)
        s_context = torch.sum(sw * s_out, dim=1)

        # =================================================
        # Gate fusion (stable version)
        # =================================================
        fusion_raw = torch.cat([t_context, s_context], dim=1)

        g = self.gate_norm(fusion_raw)
        g = F.relu(self.gate_fc1(g))
        g = self.gate_drop(g)
        g = torch.sigmoid(self.gate_fc2(g))   # (B,256)

        # Temporal weight / frequency weight
        t_gate = g
        s_gate = 1.0 - g

        t_final = t_gate * t_context
        s_final = s_gate * s_context

        fusion = torch.cat([t_final, s_final], dim=1)

        # =================================================
        # Output
        # =================================================
        out = self.regressor(fusion)

        return out


class DualPool(nn.Module):
    """
    Input:  (B, C, T)
    Output: (B, 2C)
    """

    def __init__(self, hidden_size=128):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.no = nn.LayerNorm(hidden_size)

    def forward(self, x):
        avg_pool = self.avg_pool(x).squeeze(-1)
        max_pool = self.max_pool(x).squeeze(-1)
        return self.no(torch.cat([avg_pool, max_pool], dim=1))


class MagRegressor(nn.Module):
    """
    Conv1d -> Pool -> Conv1d -> Pool -> BiLSTM -> Dense(2)
    Input: (B, C=3, L=3000)
    Output: (B, 2)  -> [pred_magnitude, s] where s is log-variance
    """
    def __init__(self, window=100, stride=50, n_fft=100, hop_length=50, fc=20, filters=(32, 64),
                 embedding_dims=64, num_heads=4, dropout=0.1):
        super().__init__()
        # Temporal window parameters
        self.window = window
        self.stride = stride
        # Frequency-domain window parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("freq_windows", torch.hann_window(n_fft))
        self.max_bin = int(fc * self.n_fft / 100) + 1

        self.temp_conv = nn.Sequential(
            nn.Conv1d(3, filters[0], kernel_size=3, padding=1),
            nn.GroupNorm(4, filters[0]),
            nn.LeakyReLU(),
            nn.Conv1d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.GroupNorm(8, filters[1]),
            nn.LeakyReLU()
        )

        self.spec_conv = nn.Sequential(
            nn.Conv1d(3, filters[0], kernel_size=3, padding=1),
            nn.GroupNorm(4, filters[0]),
            nn.LeakyReLU(),
            nn.Conv1d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.GroupNorm(8, filters[1]),
            nn.LeakyReLU()
        )

        self.pool = DualPool()

        self.temp_SparseWeightedAdjacency = SparseWeightedAdjacency(
            in_dims=filters[1]*2, embedding_dims=embedding_dims, num_heads=num_heads)

        self.temp_stsgcn = SparseGraphConvolution(
            in_dims=filters[1]*2, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.spec_SparseWeightedAdjacency = SparseWeightedAdjacency(
            in_dims=filters[1]*2, embedding_dims=embedding_dims, num_heads=num_heads)

        self.spec_stsgcn = SparseGraphConvolution(
            in_dims=filters[1]*2, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.Estimator = MagnitudeEstimator_mean_gate_stable(num_heads=num_heads,input_dim=embedding_dims // num_heads,hidden_size=128)

    def forward(self, x):
        # x: (B, 3, 3000),
        # 1. Temporal Windows
        temp_windows = x.unfold(dimension=2, size=self.window, step=self.stride)      # x_window: (B, 3, num_windows, 100)
        b, c, n, l = temp_windows.size()
        temp_identity = torch.eye(n).cuda()
        temp_windows = temp_windows.reshape(-1, c, l)  # (B * num_windows, 3, 100)

        # 2. STFT Spectrogram
        x_stft = x.reshape(b * c, -1)   # (B*C, T)

        spec_windows = torch.stft(
            x_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.freq_windows,
            return_complex=True,
            center=True
        )

        # (B*C, F, N)
        spec_windows = torch.abs(spec_windows[:, :self.max_bin, :])
        spec_windows = spec_windows.reshape(b, c, spec_windows.shape[1], spec_windows.shape[2])
        _, _, l, n = spec_windows.size()        # (B, C, F, num_windows)
        spec_identity = torch.eye(n).cuda()
        spec_windows = spec_windows.permute(0, 3, 1, 2).reshape(-1, c, l)  # (B * num_windows, 3, 100)

        # 3. CNN Feature Extraction
        temporal_feature = self.temp_conv(temp_windows)  # (B*59, 64, 100)
        # temporal_feature = self.HybridFrontEnd(x_windows)
        spec_feature = self.spec_conv(spec_windows)

        temporal_feature = self.pool(temporal_feature)
        temporal_feature = temporal_feature.reshape(b, -1, temporal_feature.shape[-1])

        spec_feature = self.pool(spec_feature)
        spec_feature = spec_feature.reshape(b, -1, spec_feature.shape[-1])

        # 4. GCN
        temp_windows_adjacency_matrix = self.temp_SparseWeightedAdjacency(temporal_feature, temp_identity)
        gcn_temporal_features = self.temp_stsgcn(temporal_feature, temp_windows_adjacency_matrix)

        spec_windows_adjacency_matrix = self.spec_SparseWeightedAdjacency(spec_feature, spec_identity)
        gcn_spec_features = self.spec_stsgcn(spec_feature, spec_windows_adjacency_matrix)

        # 5. Magnitude Estimation
        out = self.Estimator(gcn_temporal_features, gcn_spec_features)
        return out
