import pytorch_lightning as pl
import torch
import torch.nn as nn
from onnx2torch import convert


class AttnHead(pl.LightningModule):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_norm_Constant_1,
        layer_norm_weight,
        layer_norm_bias,
        c_attn_MatMul,
        c_proj_MatMul,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.layer_norm_Constant_1 = nn.Parameter(layer_norm_Constant_1)
        self.layer_norm_weight = nn.Parameter(layer_norm_weight)
        self.layer_norm_bias = nn.Parameter(layer_norm_bias)
        self.c_attn_MatMul = nn.Parameter(c_attn_MatMul)
        self.c_proj_MatMul = nn.Parameter(c_proj_MatMul)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor):
        B, T, C = x.size()
        assert C % 4 == 0

        # layer norm
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.layer_norm_Constant_1)
        x = (x * self.layer_norm_weight) + self.layer_norm_bias

        x = torch.matmul(x, self.c_attn_MatMul)
        q, k, v = x.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        h_0_attn_c_proj_mat_mul = torch.matmul(y, self.c_proj_MatMul)
        return h_0_attn_c_proj_mat_mul


class MLP(pl.LightningModule):
    def __init__(
        self,
        layer_norm_Constant_1,
        layer_norm_weight,
        layer_norm_bias,
        c_fc_MatMul,
        act_Constant,
        act_Constant_1,
        act_Constant_2,
        act_Constant_3,
        c_proj_MatMul,
    ):
        super().__init__()
        self.layer_norm_Constant_1 = nn.Parameter(layer_norm_Constant_1)
        self.layer_norm_weight = nn.Parameter(layer_norm_weight)
        self.layer_norm_bias = nn.Parameter(layer_norm_bias)
        self.c_fc_MatMul = nn.Parameter(c_fc_MatMul)
        self.act_Constant = nn.Parameter(act_Constant)
        self.act_Constant_1 = nn.Parameter(act_Constant_1)
        self.act_Constant_2 = nn.Parameter(act_Constant_2)
        self.act_Constant_3 = nn.Parameter(act_Constant_3)
        self.c_proj_MatMul = nn.Parameter(c_proj_MatMul)

    def forward(self, x: torch.Tensor):
        # layer norm
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.Constant_1_output_0)
        x = (x * self.layer_norm_weight) + self.layer_norm_bias

        y = torch.matmul(x, self.c_fc_MatMul)
        h_0_mlp_act_mul_1 = y * y * y
        h_0_mlp_act_mul_2 = self.act_Constant * h_0_mlp_act_mul_1
        h_0_mlp_act_add = y + h_0_mlp_act_mul_2
        h_0_mlp_act_mul_3 = self.act_Constant_1 * h_0_mlp_act_add
        h_0_mlp_act_tanh = torch.tanh(h_0_mlp_act_mul_3)
        h_0_mlp_act_add_1 = self.act_Constant_2 + h_0_mlp_act_tanh
        h_0_mlp_act_mul_4 = y * h_0_mlp_act_add_1
        h_0_mlp_act_mul_5 = self.act_Constant_3 * h_0_mlp_act_mul_4

        h_0_mlp_c_proj_mat_mul = torch.matmul(h_0_mlp_act_mul_5, self.c_proj_MatMul)
        x = x + h_0_mlp_c_proj_mat_mul
        return x


class Block(pl.LightningModule):
    def __init__(self, attn_kwargs: dict, mlp_kwargs: dict):
        super().__init__()
        self.attn = AttnHead(**attn_kwargs)
        self.mlp = MLP(**mlp_kwargs)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor):
        x = x + self.attn(x, pos_emb)
        x = x + self.mlp(x)
        return x


class StateModel(pl.LightningModule):
    def __init__(self, onnx_path: str):
        super().__init__()
        onnx_model = convert(onnx_path)
        self.heads = nn.Sequential(
            Block(
                attn_kwargs=dict(
                    n_embd=128,
                    n_head=4,
                    layer_norm_Constant_1=getattr(onnx_model, "h/0/attn/layer_norm/Constant_1").value,
                    layer_norm_weight=onnx_model.initializers.onnx_initializer_7,
                    layer_norm_bias=onnx_model.initializers.onnx_initializer_8,
                    c_attn_MatMul=onnx_model.initializers.onnx_initializer_9,
                    c_proj_MatMul=onnx_model.initializers.onnx_initializer_11,
                ),
                mlp_kwargs=dict(
                    layer_norm_Constant_1=getattr(onnx_model, "h/0/mlp/layer_norm/Constant_1").value,
                    layer_norm_weight=onnx_model.initializers.onnx_initializer_12,
                    layer_norm_bias=onnx_model.initializers.onnx_initializer_13,
                    c_fc_MatMul=onnx_model.initializers.onnx_initializer_14,
                    act_Constant=getattr(onnx_model, "h/0/mlp/act/Constant").value,
                    act_Constant_1=getattr(onnx_model, "h/0/mlp/act/Constant_1").value,
                    act_Constant_2=getattr(onnx_model, "h/0/mlp/act/Constant_2").value,
                    act_Constant_3=getattr(onnx_model, "h/0/mlp/act/Constant_3").value,
                    c_proj_MatMul=onnx_model.initializers.onnx_initializer_15,
                )
            ),
            Block(
                attn_kwargs=dict(
                    n_embd=128,
                    n_head=4,
                    layer_norm_Constant_1=getattr(onnx_model, "h/1/attn/layer_norm/Constant_1").value,
                    layer_norm_weight=onnx_model.initializers.onnx_initializer_16,
                    layer_norm_bias=onnx_model.initializers.onnx_initializer_17,
                    c_attn_MatMul=onnx_model.initializers.onnx_initializer_18,
                    c_proj_MatMul=onnx_model.initializers.onnx_initializer_19,
                ),
                mlp_kwargs=dict(
                    layer_norm_Constant_1=getattr(onnx_model, "h/1/mlp/layer_norm/Constant_1").value,
                    layer_norm_weight=onnx_model.initializers.onnx_initializer_20,
                    layer_norm_bias=onnx_model.initializers.onnx_initializer_21,
                    c_fc_MatMul=onnx_model.initializers.onnx_initializer_22,
                    act_Constant=getattr(onnx_model, "h/1/mlp/act/Constant").value,
                    act_Constant_1=getattr(onnx_model, "h/1/mlp/act/Constant_1").value,
                    act_Constant_2=getattr(onnx_model, "h/1/mlp/act/Constant_2").value,
                    act_Constant_3=getattr(onnx_model, "h/1/mlp/act/Constant_3").value,
                    c_proj_MatMul=onnx_model.initializers.onnx_initializer_23,
                )
            ),
            Block(
                attn_kwargs=dict(
                    n_embd=128,
                    n_head=4,
                    layer_norm_Constant_1=getattr(onnx_model, "h/2/attn/layer_norm/Constant_1").value,
                    layer_norm_weight=onnx_model.initializers.onnx_initializer_24,
                    layer_norm_bias=onnx_model.initializers.onnx_initializer_25,
                    c_attn_MatMul=onnx_model.initializers.onnx_initializer_26,
                    c_proj_MatMul=onnx_model.initializers.onnx_initializer_27,
                ),
                mlp_kwargs=dict(
                    layer_norm_Constant_1=getattr(onnx_model, "h/2/mlp/layer_norm/Constant_1").value,
                    layer_norm_weight=onnx_model.initializers.onnx_initializer_28,
                    layer_norm_bias=onnx_model.initializers.onnx_initializer_29,
                    c_fc_MatMul=onnx_model.initializers.onnx_initializer_30,
                    act_Constant=getattr(onnx_model, "h/2/mlp/act/Constant").value,
                    act_Constant_1=getattr(onnx_model, "h/2/mlp/act/Constant_1").value,
                    act_Constant_2=getattr(onnx_model, "h/2/mlp/act/Constant_2").value,
                    act_Constant_3=getattr(onnx_model, "h/2/mlp/act/Constant_3").value,
                    c_proj_MatMul=onnx_model.initializers.onnx_initializer_31,
                )
            ),
            Block(
                attn_kwargs=dict(
                    n_embd=128,
                    n_head=4,
                    layer_norm_Constant_1=getattr(onnx_model, "h/3/attn/layer_norm/Constant_1").value,
                    layer_norm_weight=onnx_model.initializers.onnx_initializer_32,
                    layer_norm_bias=onnx_model.initializers.onnx_initializer_33,
                    c_attn_MatMul=onnx_model.initializers.onnx_initializer_34,
                    c_proj_MatMul=onnx_model.initializers.onnx_initializer_35,
                ),
                mlp_kwargs=dict(
                    layer_norm_Constant_1=getattr(onnx_model, "h/3/mlp/layer_norm/Constant_1").value,
                    layer_norm_weight=onnx_model.initializers.onnx_initializer_36,
                    layer_norm_bias=onnx_model.initializers.onnx_initializer_37,
                    c_fc_MatMul=onnx_model.initializers.onnx_initializer_38,
                    act_Constant=getattr(onnx_model, "h/3/mlp/act/Constant").value,
                    act_Constant_1=getattr(onnx_model, "h/3/mlp/act/Constant_1").value,
                    act_Constant_2=getattr(onnx_model, "h/3/mlp/act/Constant_2").value,
                    act_Constant_3=getattr(onnx_model, "h/3/mlp/act/Constant_3").value,
                    c_proj_MatMul=onnx_model.initializers.onnx_initializer_39,
                )
            ),
        )
        self.wt_embedding = nn.Linear(4, 64)
        self.wt_embedding.weight.data = onnx_model.initializers.onnx_initializer_3.T
        self.wt_embedding.bias.data = onnx_model.initializers.onnx_initializer_4

        self.wt2_embedding = nn.Embedding(1024, 64)
        self.wt2_embedding.weight.data = onnx_model.initializers.onnx_initializer_5

        self.wp_embedding = nn.Embedding(20, 128)
        self.wp_embedding.weight.data = onnx_model.initializers.onnx_initializer_6

        self.layer_norm_constant_1 = nn.Parameter(getattr(onnx_model, "layer_norm_f/Constant_1").value)
        self.layer_norm_weight = nn.Parameter(onnx_model.initializers.onnx_initializer_40)
        self.layer_norm_bias = nn.Parameter(onnx_model.initializers.onnx_initializer_41)

        self.lm_head = nn.Parameter(onnx_model.initializers.onnx_initializer_42)

    def forward(self, states, tokens):
        device = states.device
        b, t, _ = states.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        num_emb = self.wt_embedding(states)  # numerical "embeddings"
        tok_emb = self.wt2_embedding(tokens)  # token embeddings
        emb = torch.cat((num_emb, tok_emb), dim=-1)
        pos_emb = self.wp_embedding(pos)  # position embeddings of shape (t, n_embd)

        x = emb + pos_emb
        for head in self.heads:
            x = head(x, pos_emb)

        # layer norm
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.layer_norm_constant_1)
        x = (x * self.layer_norm_weight) + self.layer_norm_bias

        # lm head
        x = torch.matmul(x, self.lm_head)
        return x


if __name__ == "__main__":
    onnx_path = "models/tinyphysics.onnx"
    model = StateModel(onnx_path)
