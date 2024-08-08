import torch
import torch.nn as nn


class AttnHead(nn.Module):
    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor):
        b, t, N = x.size()
        h_0_attn_layer_norm_reduce_mean = getattr(self, "h/0/attn/layer_norm/ReduceMean")(x)
        h_0_attn_layer_norm_sub = getattr(self, "h/0/attn/layer_norm/Sub")(x, h_0_attn_layer_norm_reduce_mean)
        h_0_attn_layer_norm_constant = getattr(self, "h/0/attn/layer_norm/Constant")()
        h_0_attn_layer_norm_pow = getattr(self, "h/0/attn/layer_norm/Pow")(h_0_attn_layer_norm_sub, h_0_attn_layer_norm_constant)
        h_0_attn_layer_norm_reduce_mean_1 = getattr(self, "h/0/attn/layer_norm/ReduceMean_1")(h_0_attn_layer_norm_pow)
        h_0_attn_layer_norm_constant_1 = getattr(self, "h/0/attn/layer_norm/Constant_1")()
        h_0_attn_layer_norm_add = getattr(self, "h/0/attn/layer_norm/Add")(h_0_attn_layer_norm_reduce_mean_1, h_0_attn_layer_norm_constant_1)
        h_0_attn_layer_norm_sqrt = getattr(self, "h/0/attn/layer_norm/Sqrt")(h_0_attn_layer_norm_add)
        h_0_attn_layer_norm_div = getattr(self, "h/0/attn/layer_norm/Div")(h_0_attn_layer_norm_sub, h_0_attn_layer_norm_sqrt)
        initializers_onnx_initializer_7 = self.initializers.onnx_initializer_7
        h_0_attn_layer_norm_mul = getattr(self, "h/0/attn/layer_norm/Mul")(h_0_attn_layer_norm_div, initializers_onnx_initializer_7)
        initializers_onnx_initializer_8 = self.initializers.onnx_initializer_8
        h_0_attn_layer_norm_add_1 = getattr(self, "h/0/attn/layer_norm/Add_1")(h_0_attn_layer_norm_mul, initializers_onnx_initializer_8)
        initializers_onnx_initializer_9 = self.initializers.onnx_initializer_9
        h_0_attn_c_attn_mat_mul = getattr(self, "h/0/attn/c_attn/MatMul")(h_0_attn_layer_norm_add_1, initializers_onnx_initializer_9)
        h_0_attn_constant_3 = getattr(self, "h/0/attn/Constant_3")()
        h_0_attn_split = getattr(self, "h/0/attn/Split")(h_0_attn_c_attn_mat_mul, h_0_attn_constant_3)
        h_0_attn_constant_4 = getattr(self, "h/0/attn/Constant_4")()
        h_0_attn_div = getattr(self, "h/0/attn/Div")(N, h_0_attn_constant_4)
        h_0_attn_cast = getattr(self, "h/0/attn/Cast")(h_0_attn_div)
        h_0_attn_cast_1 = getattr(self, "h/0/attn/Cast_1")(h_0_attn_cast)
        h_0_attn_unsqueeze = getattr(self, "h/0/attn/Unsqueeze")(b)
        h_0_attn_unsqueeze_1 = getattr(self, "h/0/attn/Unsqueeze_1")(t)
        h_0_attn_constant_5 = getattr(self, "h/0/attn/Constant_5")()
        h_0_attn_unsqueeze_2 = getattr(self, "h/0/attn/Unsqueeze_2")(h_0_attn_cast_1)
        h_0_attn_concat = getattr(self, "h/0/attn/Concat")(
            h_0_attn_unsqueeze,
            h_0_attn_unsqueeze_1,
            h_0_attn_constant_5,
            h_0_attn_unsqueeze_2,
        )
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
        getitem = h_0_attn_split[1]
        h_0_attn_reshape = getattr(self, "h/0/attn/Reshape")(getitem, h_0_attn_concat)
        getitem_1 = h_0_attn_split[0]
        h_0_attn_reshape_1 = getattr(self, "h/0/attn/Reshape_1")(getitem_1, h_0_attn_concat_1)
        h_0_attn_transpose = getattr(self, "h/0/attn/Transpose")(h_0_attn_reshape_1)
        getitem_2 = h_0_attn_split[2]
        h_0_attn_reshape_2 = getattr(self, "h/0/attn/Reshape_2")(getitem_2, h_0_attn_concat_2)
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
    def __init__(self):
        super().__init__()
        # TODO: read these from the ONNX file
        self.Constant_1_output_0 = None
        self.layer_norm_weight = None
        self.layer_norm_bias = None

        self.c_fc_MatMul = None

        self.act_Constant = None
        self.act_Constant_1 = None
        self.act_Constant_2 = None
        self.act_Constant_3 = None

        self.c_proj_MatMul = None

    def forward(self, x: torch.Tensor):
        # layer norm
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
    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor):
        x = x + self.attn(x, pos_emb)
        x = x + self.mlp(x)
        return x


class StateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.Sequential(
            Block(),
            Block(),
            Block(),
            Block(),
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

        x = x - x.mean(dim=-1, keepdim=True)

        layer_norm_f_constant_1 = getattr(self, "layer_norm_f/Constant_1")()
        initializers_onnx_initializer_40 = self.initializers.onnx_initializer_40
        initializers_onnx_initializer_41 = self.initializers.onnx_initializer_41

        # layer norm
        x = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + layer_norm_f_constant_1)
        x = (x * initializers_onnx_initializer_40) + initializers_onnx_initializer_41

        # lm head
        lm_head = self.initializers.onnx_initializer_42
        x = torch.matmul(x, lm_head)
        return x
