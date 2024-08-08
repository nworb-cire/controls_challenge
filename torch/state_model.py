import torch
import torch.nn as nn
from onnx2torch import convert


class AttnHead(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor):
        b, t, N = x.size()
        assert N % 4 == 0

        # layer norm
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.Constant_1_output_0)
        x = (x * self.layer_norm_weight) + self.layer_norm_bias

        initializers_onnx_initializer_7 = self.initializers.onnx_initializer_7
        initializers_onnx_initializer_8 = self.initializers.onnx_initializer_8
        initializers_onnx_initializer_9 = self.initializers.onnx_initializer_9

        x = (x * initializers_onnx_initializer_7) + initializers_onnx_initializer_8
        x = torch.matmul(x, initializers_onnx_initializer_9)
        q, k, v = x.split(self.n_embd, dim=2)
        h_0_attn_cast_1 = N // 4
        h_0_attn_unsqueeze = getattr(self, "h/0/attn/Unsqueeze")(b)
        h_0_attn_unsqueeze_1 = getattr(self, "h/0/attn/Unsqueeze_1")(t)
        h_0_attn_constant_5 = getattr(self, "h/0/attn/Constant_5")()
        h_0_attn_unsqueeze_2 = getattr(self, "h/0/attn/Unsqueeze_2")(h_0_attn_cast_1)
        h_0_attn_concat = torch.concat((
            h_0_attn_unsqueeze,
            h_0_attn_unsqueeze_1,
            h_0_attn_constant_5,
            h_0_attn_unsqueeze_2,
        ), dim=0)
        h_0_attn_unsqueeze_3 = getattr(self, "h/0/attn/Unsqueeze_3")(b)
        h_0_attn_unsqueeze_4 = getattr(self, "h/0/attn/Unsqueeze_4")(t)
        h_0_attn_constant_6 = getattr(self, "h/0/attn/Constant_6")()
        h_0_attn_unsqueeze_5 = getattr(self, "h/0/attn/Unsqueeze_5")(h_0_attn_cast_1)
        h_0_attn_concat_1 = getattr(self, "h/0/attn/Concat_1")(
            h_0_attn_unsqueeze_3,
            h_0_attn_unsqueeze_4,
            h_0_attn_constant_6,
            h_0_attn_unsqueeze_5,
        )
        h_0_attn_unsqueeze_6 = getattr(self, "h/0/attn/Unsqueeze_6")(b)
        h_0_attn_unsqueeze_7 = getattr(self, "h/0/attn/Unsqueeze_7")(t)
        h_0_attn_constant_7 = getattr(self, "h/0/attn/Constant_7")()
        h_0_attn_unsqueeze_8 = getattr(self, "h/0/attn/Unsqueeze_8")(h_0_attn_cast_1)
        h_0_attn_concat_2 = getattr(self, "h/0/attn/Concat_2")(
            h_0_attn_unsqueeze_6,
            h_0_attn_unsqueeze_7,
            h_0_attn_constant_7,
            h_0_attn_unsqueeze_8,
        )
        h_0_attn_reshape = getattr(self, "h/0/attn/Reshape")(q, h_0_attn_concat)
        h_0_attn_reshape_1 = getattr(self, "h/0/attn/Reshape_1")(k, h_0_attn_concat_1)
        h_0_attn_transpose = getattr(self, "h/0/attn/Transpose")(h_0_attn_reshape_1)
        h_0_attn_reshape_2 = getattr(self, "h/0/attn/Reshape_2")(v, h_0_attn_concat_2)
        h_0_attn_transpose_1 = getattr(self, "h/0/attn/Transpose_1")(h_0_attn_reshape_2)
        initializers_onnx_initializer_10 = self.initializers.onnx_initializer_10
        h_0_attn_gather_3 = getattr(self, "h/0/attn/Gather_3")(initializers_onnx_initializer_10, pos_emb)
        h_0_attn_transpose_2 = getattr(self, "h/0/attn/Transpose_2")(h_0_attn_reshape)
        h_0_attn_mat_mul = getattr(self, "h/0/attn/MatMul")(h_0_attn_transpose, h_0_attn_transpose_2)
        h_0_attn_constant_8 = getattr(self, "h/0/attn/Constant_8")()
        h_0_attn_mul = getattr(self, "h/0/attn/Mul")(h_0_attn_mat_mul, h_0_attn_constant_8)
        h_0_attn_not = getattr(self, "h/0/attn/Not")(h_0_attn_gather_3)
        h_0_attn_cast_2 = getattr(self, "h/0/attn/Cast_2")(h_0_attn_not)
        h_0_attn_constant_9 = getattr(self, "h/0/attn/Constant_9")()
        h_0_attn_where = getattr(self, "h/0/attn/Where")(h_0_attn_cast_2, h_0_attn_constant_9, h_0_attn_mul)
        h_0_attn_softmax = getattr(self, "h/0/attn/Softmax")(h_0_attn_where)
        h_0_attn_mat_mul_1 = getattr(self, "h/0/attn/MatMul_1")(h_0_attn_softmax, h_0_attn_transpose_1)
        h_0_attn_transpose_3 = getattr(self, "h/0/attn/Transpose_3")(h_0_attn_mat_mul_1)
        h_0_attn_unsqueeze_9 = getattr(self, "h/0/attn/Unsqueeze_9")(b)
        h_0_attn_unsqueeze_10 = getattr(self, "h/0/attn/Unsqueeze_10")(t)
        h_0_attn_unsqueeze_11 = getattr(self, "h/0/attn/Unsqueeze_11")(N)
        h_0_attn_concat_3 = getattr(self, "h/0/attn/Concat_3")(h_0_attn_unsqueeze_9, h_0_attn_unsqueeze_10, h_0_attn_unsqueeze_11)
        h_0_attn_reshape_3 = getattr(self, "h/0/attn/Reshape_3")(h_0_attn_transpose_3, h_0_attn_concat_3)
        initializers_onnx_initializer_11 = self.initializers.onnx_initializer_11
        h_0_attn_c_proj_mat_mul = getattr(self, "h/0/attn/c_proj/MatMul")(h_0_attn_reshape_3, initializers_onnx_initializer_11)
        return h_0_attn_c_proj_mat_mul


class MLP(nn.Module):
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
        self.layer_norm_Constant_1 = layer_norm_Constant_1
        self.layer_norm_weight = layer_norm_weight
        self.layer_norm_bias = layer_norm_bias
        self.c_fc_MatMul = c_fc_MatMul
        self.act_Constant = act_Constant
        self.act_Constant_1 = act_Constant_1
        self.act_Constant_2 = act_Constant_2
        self.act_Constant_3 = act_Constant_3
        self.c_proj_MatMul = c_proj_MatMul

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


class Block(nn.Module):
    def __init__(self, n_embd: int, attn_kwargs: dict, mlp_kwargs: dict):
        super().__init__()
        # self.attn = AttnHead(n_embd, **attn_kwargs)
        self.mlp = MLP(**mlp_kwargs)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor):
        x = x + self.attn(x, pos_emb)
        x = x + self.mlp(x)
        return x


class StateModel(nn.Module):
    def __init__(self, onnx_path: str):
        super().__init__()
        onnx_model = convert(onnx_path)
        self.heads = nn.Sequential(
            Block(
                n_embd=128,
                attn_kwargs=dict(),
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
                n_embd=128,
                attn_kwargs=dict(),
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
                n_embd=128,
                attn_kwargs=dict(),
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
                n_embd=128,
                attn_kwargs=dict(),
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

    def forward(self, states, tokens):
        device = states.device
        b, t, _ = states.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        num_emb = self.transformer.wt_embedding(states)  # numerical "embeddings"
        tok_emb = self.transformer.wt2_embedding(tokens)  # token embeddings
        emb = torch.cat((num_emb, tok_emb), dim=-1)
        pos_emb = self.transformer.wp_embedding(pos)  # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(emb + pos_emb)
        x = self.heads(x, pos_emb)


        layer_norm_f_constant_1 = getattr(self, "layer_norm_f/Constant_1")()
        initializers_onnx_initializer_40 = self.initializers.onnx_initializer_40
        initializers_onnx_initializer_41 = self.initializers.onnx_initializer_41

        # layer norm
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + layer_norm_f_constant_1)
        x = (x * initializers_onnx_initializer_40) + initializers_onnx_initializer_41

        # lm head
        lm_head = self.initializers.onnx_initializer_42
        x = torch.matmul(x, lm_head)
        return x


if __name__ == "__main__":
    onnx_path = "models/tinyphysics.onnx"
    model = StateModel(onnx_path)
