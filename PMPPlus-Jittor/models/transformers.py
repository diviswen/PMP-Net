import jittor as jt
from jittor import init, nn
from models.misc.ops import knn, index_points, gather_operation, grouping_operation

class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def execute(self, x, pos):
        """
        Args:
            x: Tensor, (B, c, 2048)
            pos: Tensor, (B, 2048, 3)
        """
        identity = x
        x_bcn = self.linear_start(x)
        b, dim, n = x_bcn.shape
        pos_bcn = pos.transpose(0, 2, 1)
        idx_knn = knn(pos_bcn, self.n_knn)

        key = self.conv_key(x_bcn)
        value = self.conv_value(x_bcn)
        query = self.conv_query(x_bcn)

        # key = index_points(key.transpose(0, 2, 1), idx_knn).transpose(0, 3, 1, 2)  # (b, c, n, n_knn)
        key = grouping_operation(key, idx_knn)
        # print('key.shape', key.shape)
        qk_rel = query.reshape((b, -1, n, 1)) - key


        pos_rel = pos_bcn.reshape((b, -1, n, 1)) - \
                  grouping_operation(pos_bcn, idx_knn)
                  # index_points(pos, idx_knn).transpose(0, 3, 1, 2)
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = nn.softmax(attention, dim=-1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = (value * attention).sum(dim=-1)
        y = self.linear_end(agg)

        return y+identity





