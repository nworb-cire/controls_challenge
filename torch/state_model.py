import torch.nn as nn


class StateModel(nn.Module):
    def forward(self, input_1, input_2):
        initializers_onnx_initializer_0 = self.initializers.onnx_initializer_0
        identity_128 = self.Identity_128(initializers_onnx_initializer_0)
        initializers_onnx_initializer_1 = self.initializers.onnx_initializer_1
        identity_129 = self.Identity_129(initializers_onnx_initializer_1)
        initializers_onnx_initializer_2 = self.initializers.onnx_initializer_2
        identity_130 = self.Identity_130(initializers_onnx_initializer_2)
        shape = self.Shape(input_1)
        constant = self.Constant()
        gather = self.Gather(shape, constant)
        constant_1 = self.Constant_1()
        cast = self.Cast(gather)
        constant_2 = self.Constant_2()
        range_1 = self.Range(constant_1, cast, constant_2)
        initializers_onnx_initializer_3 = self.initializers.onnx_initializer_3
        wt_embedding_mat_mul = getattr(self, "wt_embedding/MatMul")(
            input_1, initializers_onnx_initializer_3
        )
        initializers_onnx_initializer_4 = self.initializers.onnx_initializer_4
        wt_embedding_add = getattr(self, "wt_embedding/Add")(
            initializers_onnx_initializer_4, wt_embedding_mat_mul
        )
        initializers_onnx_initializer_5 = self.initializers.onnx_initializer_5
        wt2_embedding_gather = getattr(self, "wt2_embedding/Gather")(
            initializers_onnx_initializer_5, input_2
        )
        unsqueeze = self.Unsqueeze(range_1)
        constant_4 = self.Constant_4()
        unsqueeze_1 = self.Unsqueeze_1(gather)
        concat = self.Concat(constant_4, unsqueeze_1)
        reshape = self.Reshape(unsqueeze, concat)
        initializers_onnx_initializer_6 = self.initializers.onnx_initializer_6
        wp_embedding_gather = getattr(self, "wp_embedding/Gather")(
            initializers_onnx_initializer_6, reshape
        )
        concat_1 = self.Concat_1(wt_embedding_add, wt2_embedding_gather)
        add = self.Add(concat_1, wp_embedding_gather)
        h_0_attn_shape = getattr(self, "h/0/attn/Shape")(add)
        h_0_attn_constant = getattr(self, "h/0/attn/Constant")()
        h_0_attn_gather = getattr(self, "h/0/attn/Gather")(
            h_0_attn_shape, h_0_attn_constant
        )
        h_0_attn_shape_1 = getattr(self, "h/0/attn/Shape_1")(add)
        h_0_attn_constant_1 = getattr(self, "h/0/attn/Constant_1")()
        h_0_attn_gather_1 = getattr(self, "h/0/attn/Gather_1")(
            h_0_attn_shape_1, h_0_attn_constant_1
        )
        h_0_attn_shape_2 = getattr(self, "h/0/attn/Shape_2")(add)
        h_0_attn_constant_2 = getattr(self, "h/0/attn/Constant_2")()
        h_0_attn_gather_2 = getattr(self, "h/0/attn/Gather_2")(
            h_0_attn_shape_2, h_0_attn_constant_2
        )
        h_0_attn_layer_norm_reduce_mean = getattr(self, "h/0/attn/layer_norm/ReduceMean")(
            add
        )
        h_0_attn_layer_norm_sub = getattr(self, "h/0/attn/layer_norm/Sub")(
            add, h_0_attn_layer_norm_reduce_mean
        )
        h_0_attn_layer_norm_constant = getattr(self, "h/0/attn/layer_norm/Constant")()
        h_0_attn_layer_norm_pow = getattr(self, "h/0/attn/layer_norm/Pow")(
            h_0_attn_layer_norm_sub, h_0_attn_layer_norm_constant
        )
        h_0_attn_layer_norm_reduce_mean_1 = getattr(
            self, "h/0/attn/layer_norm/ReduceMean_1"
        )(h_0_attn_layer_norm_pow)
        h_0_attn_layer_norm_constant_1 = getattr(self, "h/0/attn/layer_norm/Constant_1")()
        h_0_attn_layer_norm_add = getattr(self, "h/0/attn/layer_norm/Add")(
            h_0_attn_layer_norm_reduce_mean_1, h_0_attn_layer_norm_constant_1
        )
        h_0_attn_layer_norm_sqrt = getattr(self, "h/0/attn/layer_norm/Sqrt")(
            h_0_attn_layer_norm_add
        )
        h_0_attn_layer_norm_div = getattr(self, "h/0/attn/layer_norm/Div")(
            h_0_attn_layer_norm_sub, h_0_attn_layer_norm_sqrt
        )
        initializers_onnx_initializer_7 = self.initializers.onnx_initializer_7
        h_0_attn_layer_norm_mul = getattr(self, "h/0/attn/layer_norm/Mul")(
            h_0_attn_layer_norm_div, initializers_onnx_initializer_7
        )
        initializers_onnx_initializer_8 = self.initializers.onnx_initializer_8
        h_0_attn_layer_norm_add_1 = getattr(self, "h/0/attn/layer_norm/Add_1")(
            h_0_attn_layer_norm_mul, initializers_onnx_initializer_8
        )
        initializers_onnx_initializer_9 = self.initializers.onnx_initializer_9
        h_0_attn_c_attn_mat_mul = getattr(self, "h/0/attn/c_attn/MatMul")(
            h_0_attn_layer_norm_add_1, initializers_onnx_initializer_9
        )
        h_0_attn_constant_3 = getattr(self, "h/0/attn/Constant_3")()
        h_0_attn_split = getattr(self, "h/0/attn/Split")(
            h_0_attn_c_attn_mat_mul, h_0_attn_constant_3
        )
        h_0_attn_constant_4 = getattr(self, "h/0/attn/Constant_4")()
        h_0_attn_div = getattr(self, "h/0/attn/Div")(h_0_attn_gather_2, h_0_attn_constant_4)
        h_0_attn_cast = getattr(self, "h/0/attn/Cast")(h_0_attn_div)
        h_0_attn_cast_1 = getattr(self, "h/0/attn/Cast_1")(h_0_attn_cast)
        h_0_attn_unsqueeze = getattr(self, "h/0/attn/Unsqueeze")(h_0_attn_gather)
        h_0_attn_unsqueeze_1 = getattr(self, "h/0/attn/Unsqueeze_1")(h_0_attn_gather_1)
        h_0_attn_constant_5 = getattr(self, "h/0/attn/Constant_5")()
        h_0_attn_unsqueeze_2 = getattr(self, "h/0/attn/Unsqueeze_2")(h_0_attn_cast_1)
        h_0_attn_concat = getattr(self, "h/0/attn/Concat")(
            h_0_attn_unsqueeze,
            h_0_attn_unsqueeze_1,
            h_0_attn_constant_5,
            h_0_attn_unsqueeze_2,
        )
        h_0_attn_unsqueeze_3 = getattr(self, "h/0/attn/Unsqueeze_3")(h_0_attn_gather)
        h_0_attn_unsqueeze_4 = getattr(self, "h/0/attn/Unsqueeze_4")(h_0_attn_gather_1)
        h_0_attn_constant_6 = getattr(self, "h/0/attn/Constant_6")()
        h_0_attn_unsqueeze_5 = getattr(self, "h/0/attn/Unsqueeze_5")(h_0_attn_cast_1)
        h_0_attn_concat_1 = getattr(self, "h/0/attn/Concat_1")(
            h_0_attn_unsqueeze_3,
            h_0_attn_unsqueeze_4,
            h_0_attn_constant_6,
            h_0_attn_unsqueeze_5,
        )
        h_0_attn_unsqueeze_6 = getattr(self, "h/0/attn/Unsqueeze_6")(h_0_attn_gather)
        h_0_attn_unsqueeze_7 = getattr(self, "h/0/attn/Unsqueeze_7")(h_0_attn_gather_1)
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
        h_0_attn_reshape_1 = getattr(self, "h/0/attn/Reshape_1")(
            getitem_1, h_0_attn_concat_1
        )
        h_0_attn_transpose = getattr(self, "h/0/attn/Transpose")(h_0_attn_reshape_1)
        getitem_2 = h_0_attn_split[2]
        h_0_attn_reshape_2 = getattr(self, "h/0/attn/Reshape_2")(
            getitem_2, h_0_attn_concat_2
        )
        h_0_attn_transpose_1 = getattr(self, "h/0/attn/Transpose_1")(h_0_attn_reshape_2)
        initializers_onnx_initializer_10 = self.initializers.onnx_initializer_10
        h_0_attn_gather_3 = getattr(self, "h/0/attn/Gather_3")(
            initializers_onnx_initializer_10, range_1
        )
        h_0_attn_transpose_2 = getattr(self, "h/0/attn/Transpose_2")(h_0_attn_reshape)
        h_0_attn_mat_mul = getattr(self, "h/0/attn/MatMul")(
            h_0_attn_transpose, h_0_attn_transpose_2
        )
        h_0_attn_constant_8 = getattr(self, "h/0/attn/Constant_8")()
        h_0_attn_mul = getattr(self, "h/0/attn/Mul")(h_0_attn_mat_mul, h_0_attn_constant_8)
        h_0_attn_not = getattr(self, "h/0/attn/Not")(h_0_attn_gather_3)
        h_0_attn_cast_2 = getattr(self, "h/0/attn/Cast_2")(h_0_attn_not)
        h_0_attn_constant_9 = getattr(self, "h/0/attn/Constant_9")()
        h_0_attn_where = getattr(self, "h/0/attn/Where")(
            h_0_attn_cast_2, h_0_attn_constant_9, h_0_attn_mul
        )
        h_0_attn_softmax = getattr(self, "h/0/attn/Softmax")(h_0_attn_where)
        h_0_attn_mat_mul_1 = getattr(self, "h/0/attn/MatMul_1")(
            h_0_attn_softmax, h_0_attn_transpose_1
        )
        h_0_attn_transpose_3 = getattr(self, "h/0/attn/Transpose_3")(h_0_attn_mat_mul_1)
        h_0_attn_unsqueeze_9 = getattr(self, "h/0/attn/Unsqueeze_9")(h_0_attn_gather)
        h_0_attn_unsqueeze_10 = getattr(self, "h/0/attn/Unsqueeze_10")(h_0_attn_gather_1)
        h_0_attn_unsqueeze_11 = getattr(self, "h/0/attn/Unsqueeze_11")(h_0_attn_gather_2)
        h_0_attn_concat_3 = getattr(self, "h/0/attn/Concat_3")(
            h_0_attn_unsqueeze_9, h_0_attn_unsqueeze_10, h_0_attn_unsqueeze_11
        )
        h_0_attn_reshape_3 = getattr(self, "h/0/attn/Reshape_3")(
            h_0_attn_transpose_3, h_0_attn_concat_3
        )
        initializers_onnx_initializer_11 = self.initializers.onnx_initializer_11
        h_0_attn_c_proj_mat_mul = getattr(self, "h/0/attn/c_proj/MatMul")(
            h_0_attn_reshape_3, initializers_onnx_initializer_11
        )
        h_0_add = getattr(self, "h/0/Add")(add, h_0_attn_c_proj_mat_mul)
        h_0_mlp_layer_norm_reduce_mean = getattr(self, "h/0/mlp/layer_norm/ReduceMean")(
            h_0_add
        )
        h_0_mlp_layer_norm_sub = getattr(self, "h/0/mlp/layer_norm/Sub")(
            h_0_add, h_0_mlp_layer_norm_reduce_mean
        )
        h_0_mlp_layer_norm_constant = getattr(self, "h/0/mlp/layer_norm/Constant")()
        h_0_mlp_layer_norm_pow = getattr(self, "h/0/mlp/layer_norm/Pow")(
            h_0_mlp_layer_norm_sub, h_0_mlp_layer_norm_constant
        )
        h_0_mlp_layer_norm_reduce_mean_1 = getattr(self, "h/0/mlp/layer_norm/ReduceMean_1")(
            h_0_mlp_layer_norm_pow
        )
        h_0_mlp_layer_norm_constant_1 = getattr(self, "h/0/mlp/layer_norm/Constant_1")()
        h_0_mlp_layer_norm_add = getattr(self, "h/0/mlp/layer_norm/Add")(
            h_0_mlp_layer_norm_reduce_mean_1, h_0_mlp_layer_norm_constant_1
        )
        h_0_mlp_layer_norm_sqrt = getattr(self, "h/0/mlp/layer_norm/Sqrt")(
            h_0_mlp_layer_norm_add
        )
        h_0_mlp_layer_norm_div = getattr(self, "h/0/mlp/layer_norm/Div")(
            h_0_mlp_layer_norm_sub, h_0_mlp_layer_norm_sqrt
        )
        initializers_onnx_initializer_12 = self.initializers.onnx_initializer_12
        h_0_mlp_layer_norm_mul = getattr(self, "h/0/mlp/layer_norm/Mul")(
            h_0_mlp_layer_norm_div, initializers_onnx_initializer_12
        )
        initializers_onnx_initializer_13 = self.initializers.onnx_initializer_13
        h_0_mlp_layer_norm_add_1 = getattr(self, "h/0/mlp/layer_norm/Add_1")(
            h_0_mlp_layer_norm_mul, initializers_onnx_initializer_13
        )
        initializers_onnx_initializer_14 = self.initializers.onnx_initializer_14
        h_0_mlp_c_fc_mat_mul = getattr(self, "h/0/mlp/c_fc/MatMul")(
            h_0_mlp_layer_norm_add_1, initializers_onnx_initializer_14
        )
        h_0_mlp_act_mul = getattr(self, "h/0/mlp/act/Mul")(
            h_0_mlp_c_fc_mat_mul, h_0_mlp_c_fc_mat_mul
        )
        h_0_mlp_act_mul_1 = getattr(self, "h/0/mlp/act/Mul_1")(
            h_0_mlp_c_fc_mat_mul, h_0_mlp_act_mul
        )
        h_0_mlp_act_constant = getattr(self, "h/0/mlp/act/Constant")()
        h_0_mlp_act_mul_2 = getattr(self, "h/0/mlp/act/Mul_2")(
            h_0_mlp_act_constant, h_0_mlp_act_mul_1
        )
        h_0_mlp_act_add = getattr(self, "h/0/mlp/act/Add")(
            h_0_mlp_c_fc_mat_mul, h_0_mlp_act_mul_2
        )
        h_0_mlp_act_constant_1 = getattr(self, "h/0/mlp/act/Constant_1")()
        h_0_mlp_act_mul_3 = getattr(self, "h/0/mlp/act/Mul_3")(
            h_0_mlp_act_constant_1, h_0_mlp_act_add
        )
        h_0_mlp_act_tanh = getattr(self, "h/0/mlp/act/Tanh")(h_0_mlp_act_mul_3)
        h_0_mlp_act_constant_2 = getattr(self, "h/0/mlp/act/Constant_2")()
        h_0_mlp_act_add_1 = getattr(self, "h/0/mlp/act/Add_1")(
            h_0_mlp_act_constant_2, h_0_mlp_act_tanh
        )
        h_0_mlp_act_mul_4 = getattr(self, "h/0/mlp/act/Mul_4")(
            h_0_mlp_c_fc_mat_mul, h_0_mlp_act_add_1
        )
        h_0_mlp_act_constant_3 = getattr(self, "h/0/mlp/act/Constant_3")()
        h_0_mlp_act_mul_5 = getattr(self, "h/0/mlp/act/Mul_5")(
            h_0_mlp_act_constant_3, h_0_mlp_act_mul_4
        )
        initializers_onnx_initializer_15 = self.initializers.onnx_initializer_15
        h_0_mlp_c_proj_mat_mul = getattr(self, "h/0/mlp/c_proj/MatMul")(
            h_0_mlp_act_mul_5, initializers_onnx_initializer_15
        )
        h_0_add_1 = getattr(self, "h/0/Add_1")(h_0_add, h_0_mlp_c_proj_mat_mul)
        h_1_attn_shape = getattr(self, "h/1/attn/Shape")(h_0_add_1)
        h_1_attn_constant = getattr(self, "h/1/attn/Constant")()
        h_1_attn_gather = getattr(self, "h/1/attn/Gather")(
            h_1_attn_shape, h_1_attn_constant
        )
        h_1_attn_shape_1 = getattr(self, "h/1/attn/Shape_1")(h_0_add_1)
        h_1_attn_constant_1 = getattr(self, "h/1/attn/Constant_1")()
        h_1_attn_gather_1 = getattr(self, "h/1/attn/Gather_1")(
            h_1_attn_shape_1, h_1_attn_constant_1
        )
        h_1_attn_shape_2 = getattr(self, "h/1/attn/Shape_2")(h_0_add_1)
        h_1_attn_constant_2 = getattr(self, "h/1/attn/Constant_2")()
        h_1_attn_gather_2 = getattr(self, "h/1/attn/Gather_2")(
            h_1_attn_shape_2, h_1_attn_constant_2
        )
        h_1_attn_layer_norm_reduce_mean = getattr(self, "h/1/attn/layer_norm/ReduceMean")(
            h_0_add_1
        )
        h_1_attn_layer_norm_sub = getattr(self, "h/1/attn/layer_norm/Sub")(
            h_0_add_1, h_1_attn_layer_norm_reduce_mean
        )
        h_1_attn_layer_norm_constant = getattr(self, "h/1/attn/layer_norm/Constant")()
        h_1_attn_layer_norm_pow = getattr(self, "h/1/attn/layer_norm/Pow")(
            h_1_attn_layer_norm_sub, h_1_attn_layer_norm_constant
        )
        h_1_attn_layer_norm_reduce_mean_1 = getattr(
            self, "h/1/attn/layer_norm/ReduceMean_1"
        )(h_1_attn_layer_norm_pow)
        h_1_attn_layer_norm_constant_1 = getattr(self, "h/1/attn/layer_norm/Constant_1")()
        h_1_attn_layer_norm_add = getattr(self, "h/1/attn/layer_norm/Add")(
            h_1_attn_layer_norm_reduce_mean_1, h_1_attn_layer_norm_constant_1
        )
        h_1_attn_layer_norm_sqrt = getattr(self, "h/1/attn/layer_norm/Sqrt")(
            h_1_attn_layer_norm_add
        )
        h_1_attn_layer_norm_div = getattr(self, "h/1/attn/layer_norm/Div")(
            h_1_attn_layer_norm_sub, h_1_attn_layer_norm_sqrt
        )
        initializers_onnx_initializer_16 = self.initializers.onnx_initializer_16
        h_1_attn_layer_norm_mul = getattr(self, "h/1/attn/layer_norm/Mul")(
            h_1_attn_layer_norm_div, initializers_onnx_initializer_16
        )
        initializers_onnx_initializer_17 = self.initializers.onnx_initializer_17
        h_1_attn_layer_norm_add_1 = getattr(self, "h/1/attn/layer_norm/Add_1")(
            h_1_attn_layer_norm_mul, initializers_onnx_initializer_17
        )
        initializers_onnx_initializer_18 = self.initializers.onnx_initializer_18
        h_1_attn_c_attn_mat_mul = getattr(self, "h/1/attn/c_attn/MatMul")(
            h_1_attn_layer_norm_add_1, initializers_onnx_initializer_18
        )
        h_1_attn_constant_3 = getattr(self, "h/1/attn/Constant_3")()
        h_1_attn_split = getattr(self, "h/1/attn/Split")(
            h_1_attn_c_attn_mat_mul, h_1_attn_constant_3
        )
        h_1_attn_constant_4 = getattr(self, "h/1/attn/Constant_4")()
        h_1_attn_div = getattr(self, "h/1/attn/Div")(h_1_attn_gather_2, h_1_attn_constant_4)
        h_1_attn_cast = getattr(self, "h/1/attn/Cast")(h_1_attn_div)
        h_1_attn_cast_1 = getattr(self, "h/1/attn/Cast_1")(h_1_attn_cast)
        h_1_attn_unsqueeze = getattr(self, "h/1/attn/Unsqueeze")(h_1_attn_gather)
        h_1_attn_unsqueeze_1 = getattr(self, "h/1/attn/Unsqueeze_1")(h_1_attn_gather_1)
        h_1_attn_constant_5 = getattr(self, "h/1/attn/Constant_5")()
        h_1_attn_unsqueeze_2 = getattr(self, "h/1/attn/Unsqueeze_2")(h_1_attn_cast_1)
        h_1_attn_concat = getattr(self, "h/1/attn/Concat")(
            h_1_attn_unsqueeze,
            h_1_attn_unsqueeze_1,
            h_1_attn_constant_5,
            h_1_attn_unsqueeze_2,
        )
        h_1_attn_unsqueeze_3 = getattr(self, "h/1/attn/Unsqueeze_3")(h_1_attn_gather)
        h_1_attn_unsqueeze_4 = getattr(self, "h/1/attn/Unsqueeze_4")(h_1_attn_gather_1)
        h_1_attn_constant_6 = getattr(self, "h/1/attn/Constant_6")()
        h_1_attn_unsqueeze_5 = getattr(self, "h/1/attn/Unsqueeze_5")(h_1_attn_cast_1)
        h_1_attn_concat_1 = getattr(self, "h/1/attn/Concat_1")(
            h_1_attn_unsqueeze_3,
            h_1_attn_unsqueeze_4,
            h_1_attn_constant_6,
            h_1_attn_unsqueeze_5,
        )
        h_1_attn_unsqueeze_6 = getattr(self, "h/1/attn/Unsqueeze_6")(h_1_attn_gather)
        h_1_attn_unsqueeze_7 = getattr(self, "h/1/attn/Unsqueeze_7")(h_1_attn_gather_1)
        h_1_attn_constant_7 = getattr(self, "h/1/attn/Constant_7")()
        h_1_attn_unsqueeze_8 = getattr(self, "h/1/attn/Unsqueeze_8")(h_1_attn_cast_1)
        h_1_attn_concat_2 = getattr(self, "h/1/attn/Concat_2")(
            h_1_attn_unsqueeze_6,
            h_1_attn_unsqueeze_7,
            h_1_attn_constant_7,
            h_1_attn_unsqueeze_8,
        )
        getitem_3 = h_1_attn_split[1]
        h_1_attn_reshape = getattr(self, "h/1/attn/Reshape")(getitem_3, h_1_attn_concat)
        getitem_4 = h_1_attn_split[0]
        h_1_attn_reshape_1 = getattr(self, "h/1/attn/Reshape_1")(
            getitem_4, h_1_attn_concat_1
        )
        h_1_attn_transpose = getattr(self, "h/1/attn/Transpose")(h_1_attn_reshape_1)
        getitem_5 = h_1_attn_split[2]
        h_1_attn_reshape_2 = getattr(self, "h/1/attn/Reshape_2")(
            getitem_5, h_1_attn_concat_2
        )
        h_1_attn_transpose_1 = getattr(self, "h/1/attn/Transpose_1")(h_1_attn_reshape_2)
        h_1_attn_gather_3 = getattr(self, "h/1/attn/Gather_3")(identity_130, range_1)
        h_1_attn_transpose_2 = getattr(self, "h/1/attn/Transpose_2")(h_1_attn_reshape)
        h_1_attn_mat_mul = getattr(self, "h/1/attn/MatMul")(
            h_1_attn_transpose, h_1_attn_transpose_2
        )
        h_1_attn_constant_8 = getattr(self, "h/1/attn/Constant_8")()
        h_1_attn_mul = getattr(self, "h/1/attn/Mul")(h_1_attn_mat_mul, h_1_attn_constant_8)
        h_1_attn_not = getattr(self, "h/1/attn/Not")(h_1_attn_gather_3)
        h_1_attn_cast_2 = getattr(self, "h/1/attn/Cast_2")(h_1_attn_not)
        h_1_attn_constant_9 = getattr(self, "h/1/attn/Constant_9")()
        h_1_attn_where = getattr(self, "h/1/attn/Where")(
            h_1_attn_cast_2, h_1_attn_constant_9, h_1_attn_mul
        )
        h_1_attn_softmax = getattr(self, "h/1/attn/Softmax")(h_1_attn_where)
        h_1_attn_mat_mul_1 = getattr(self, "h/1/attn/MatMul_1")(
            h_1_attn_softmax, h_1_attn_transpose_1
        )
        h_1_attn_transpose_3 = getattr(self, "h/1/attn/Transpose_3")(h_1_attn_mat_mul_1)
        h_1_attn_unsqueeze_9 = getattr(self, "h/1/attn/Unsqueeze_9")(h_1_attn_gather)
        h_1_attn_unsqueeze_10 = getattr(self, "h/1/attn/Unsqueeze_10")(h_1_attn_gather_1)
        h_1_attn_unsqueeze_11 = getattr(self, "h/1/attn/Unsqueeze_11")(h_1_attn_gather_2)
        h_1_attn_concat_3 = getattr(self, "h/1/attn/Concat_3")(
            h_1_attn_unsqueeze_9, h_1_attn_unsqueeze_10, h_1_attn_unsqueeze_11
        )
        h_1_attn_reshape_3 = getattr(self, "h/1/attn/Reshape_3")(
            h_1_attn_transpose_3, h_1_attn_concat_3
        )
        initializers_onnx_initializer_19 = self.initializers.onnx_initializer_19
        h_1_attn_c_proj_mat_mul = getattr(self, "h/1/attn/c_proj/MatMul")(
            h_1_attn_reshape_3, initializers_onnx_initializer_19
        )
        h_1_add = getattr(self, "h/1/Add")(h_0_add_1, h_1_attn_c_proj_mat_mul)
        h_1_mlp_layer_norm_reduce_mean = getattr(self, "h/1/mlp/layer_norm/ReduceMean")(
            h_1_add
        )
        h_1_mlp_layer_norm_sub = getattr(self, "h/1/mlp/layer_norm/Sub")(
            h_1_add, h_1_mlp_layer_norm_reduce_mean
        )
        h_1_mlp_layer_norm_constant = getattr(self, "h/1/mlp/layer_norm/Constant")()
        h_1_mlp_layer_norm_pow = getattr(self, "h/1/mlp/layer_norm/Pow")(
            h_1_mlp_layer_norm_sub, h_1_mlp_layer_norm_constant
        )
        h_1_mlp_layer_norm_reduce_mean_1 = getattr(self, "h/1/mlp/layer_norm/ReduceMean_1")(
            h_1_mlp_layer_norm_pow
        )
        h_1_mlp_layer_norm_constant_1 = getattr(self, "h/1/mlp/layer_norm/Constant_1")()
        h_1_mlp_layer_norm_add = getattr(self, "h/1/mlp/layer_norm/Add")(
            h_1_mlp_layer_norm_reduce_mean_1, h_1_mlp_layer_norm_constant_1
        )
        h_1_mlp_layer_norm_sqrt = getattr(self, "h/1/mlp/layer_norm/Sqrt")(
            h_1_mlp_layer_norm_add
        )
        h_1_mlp_layer_norm_div = getattr(self, "h/1/mlp/layer_norm/Div")(
            h_1_mlp_layer_norm_sub, h_1_mlp_layer_norm_sqrt
        )
        initializers_onnx_initializer_20 = self.initializers.onnx_initializer_20
        h_1_mlp_layer_norm_mul = getattr(self, "h/1/mlp/layer_norm/Mul")(
            h_1_mlp_layer_norm_div, initializers_onnx_initializer_20
        )
        initializers_onnx_initializer_21 = self.initializers.onnx_initializer_21
        h_1_mlp_layer_norm_add_1 = getattr(self, "h/1/mlp/layer_norm/Add_1")(
            h_1_mlp_layer_norm_mul, initializers_onnx_initializer_21
        )
        initializers_onnx_initializer_22 = self.initializers.onnx_initializer_22
        h_1_mlp_c_fc_mat_mul = getattr(self, "h/1/mlp/c_fc/MatMul")(
            h_1_mlp_layer_norm_add_1, initializers_onnx_initializer_22
        )
        h_1_mlp_act_mul = getattr(self, "h/1/mlp/act/Mul")(
            h_1_mlp_c_fc_mat_mul, h_1_mlp_c_fc_mat_mul
        )
        h_1_mlp_act_mul_1 = getattr(self, "h/1/mlp/act/Mul_1")(
            h_1_mlp_c_fc_mat_mul, h_1_mlp_act_mul
        )
        h_1_mlp_act_constant = getattr(self, "h/1/mlp/act/Constant")()
        h_1_mlp_act_mul_2 = getattr(self, "h/1/mlp/act/Mul_2")(
            h_1_mlp_act_constant, h_1_mlp_act_mul_1
        )
        h_1_mlp_act_add = getattr(self, "h/1/mlp/act/Add")(
            h_1_mlp_c_fc_mat_mul, h_1_mlp_act_mul_2
        )
        h_1_mlp_act_constant_1 = getattr(self, "h/1/mlp/act/Constant_1")()
        h_1_mlp_act_mul_3 = getattr(self, "h/1/mlp/act/Mul_3")(
            h_1_mlp_act_constant_1, h_1_mlp_act_add
        )
        h_1_mlp_act_tanh = getattr(self, "h/1/mlp/act/Tanh")(h_1_mlp_act_mul_3)
        h_1_mlp_act_constant_2 = getattr(self, "h/1/mlp/act/Constant_2")()
        h_1_mlp_act_add_1 = getattr(self, "h/1/mlp/act/Add_1")(
            h_1_mlp_act_constant_2, h_1_mlp_act_tanh
        )
        h_1_mlp_act_mul_4 = getattr(self, "h/1/mlp/act/Mul_4")(
            h_1_mlp_c_fc_mat_mul, h_1_mlp_act_add_1
        )
        h_1_mlp_act_constant_3 = getattr(self, "h/1/mlp/act/Constant_3")()
        h_1_mlp_act_mul_5 = getattr(self, "h/1/mlp/act/Mul_5")(
            h_1_mlp_act_constant_3, h_1_mlp_act_mul_4
        )
        initializers_onnx_initializer_23 = self.initializers.onnx_initializer_23
        h_1_mlp_c_proj_mat_mul = getattr(self, "h/1/mlp/c_proj/MatMul")(
            h_1_mlp_act_mul_5, initializers_onnx_initializer_23
        )
        h_1_add_1 = getattr(self, "h/1/Add_1")(h_1_add, h_1_mlp_c_proj_mat_mul)
        h_2_attn_shape = getattr(self, "h/2/attn/Shape")(h_1_add_1)
        h_2_attn_constant = getattr(self, "h/2/attn/Constant")()
        h_2_attn_gather = getattr(self, "h/2/attn/Gather")(
            h_2_attn_shape, h_2_attn_constant
        )
        h_2_attn_shape_1 = getattr(self, "h/2/attn/Shape_1")(h_1_add_1)
        h_2_attn_constant_1 = getattr(self, "h/2/attn/Constant_1")()
        h_2_attn_gather_1 = getattr(self, "h/2/attn/Gather_1")(
            h_2_attn_shape_1, h_2_attn_constant_1
        )
        h_2_attn_shape_2 = getattr(self, "h/2/attn/Shape_2")(h_1_add_1)
        h_2_attn_constant_2 = getattr(self, "h/2/attn/Constant_2")()
        h_2_attn_gather_2 = getattr(self, "h/2/attn/Gather_2")(
            h_2_attn_shape_2, h_2_attn_constant_2
        )
        h_2_attn_layer_norm_reduce_mean = getattr(self, "h/2/attn/layer_norm/ReduceMean")(
            h_1_add_1
        )
        h_2_attn_layer_norm_sub = getattr(self, "h/2/attn/layer_norm/Sub")(
            h_1_add_1, h_2_attn_layer_norm_reduce_mean
        )
        h_2_attn_layer_norm_constant = getattr(self, "h/2/attn/layer_norm/Constant")()
        h_2_attn_layer_norm_pow = getattr(self, "h/2/attn/layer_norm/Pow")(
            h_2_attn_layer_norm_sub, h_2_attn_layer_norm_constant
        )
        h_2_attn_layer_norm_reduce_mean_1 = getattr(
            self, "h/2/attn/layer_norm/ReduceMean_1"
        )(h_2_attn_layer_norm_pow)
        h_2_attn_layer_norm_constant_1 = getattr(self, "h/2/attn/layer_norm/Constant_1")()
        h_2_attn_layer_norm_add = getattr(self, "h/2/attn/layer_norm/Add")(
            h_2_attn_layer_norm_reduce_mean_1, h_2_attn_layer_norm_constant_1
        )
        h_2_attn_layer_norm_sqrt = getattr(self, "h/2/attn/layer_norm/Sqrt")(
            h_2_attn_layer_norm_add
        )
        h_2_attn_layer_norm_div = getattr(self, "h/2/attn/layer_norm/Div")(
            h_2_attn_layer_norm_sub, h_2_attn_layer_norm_sqrt
        )
        initializers_onnx_initializer_24 = self.initializers.onnx_initializer_24
        h_2_attn_layer_norm_mul = getattr(self, "h/2/attn/layer_norm/Mul")(
            h_2_attn_layer_norm_div, initializers_onnx_initializer_24
        )
        initializers_onnx_initializer_25 = self.initializers.onnx_initializer_25
        h_2_attn_layer_norm_add_1 = getattr(self, "h/2/attn/layer_norm/Add_1")(
            h_2_attn_layer_norm_mul, initializers_onnx_initializer_25
        )
        initializers_onnx_initializer_26 = self.initializers.onnx_initializer_26
        h_2_attn_c_attn_mat_mul = getattr(self, "h/2/attn/c_attn/MatMul")(
            h_2_attn_layer_norm_add_1, initializers_onnx_initializer_26
        )
        h_2_attn_constant_3 = getattr(self, "h/2/attn/Constant_3")()
        h_2_attn_split = getattr(self, "h/2/attn/Split")(
            h_2_attn_c_attn_mat_mul, h_2_attn_constant_3
        )
        h_2_attn_constant_4 = getattr(self, "h/2/attn/Constant_4")()
        h_2_attn_div = getattr(self, "h/2/attn/Div")(h_2_attn_gather_2, h_2_attn_constant_4)
        h_2_attn_cast = getattr(self, "h/2/attn/Cast")(h_2_attn_div)
        h_2_attn_cast_1 = getattr(self, "h/2/attn/Cast_1")(h_2_attn_cast)
        h_2_attn_unsqueeze = getattr(self, "h/2/attn/Unsqueeze")(h_2_attn_gather)
        h_2_attn_unsqueeze_1 = getattr(self, "h/2/attn/Unsqueeze_1")(h_2_attn_gather_1)
        h_2_attn_constant_5 = getattr(self, "h/2/attn/Constant_5")()
        h_2_attn_unsqueeze_2 = getattr(self, "h/2/attn/Unsqueeze_2")(h_2_attn_cast_1)
        h_2_attn_concat = getattr(self, "h/2/attn/Concat")(
            h_2_attn_unsqueeze,
            h_2_attn_unsqueeze_1,
            h_2_attn_constant_5,
            h_2_attn_unsqueeze_2,
        )
        h_2_attn_unsqueeze_3 = getattr(self, "h/2/attn/Unsqueeze_3")(h_2_attn_gather)
        h_2_attn_unsqueeze_4 = getattr(self, "h/2/attn/Unsqueeze_4")(h_2_attn_gather_1)
        h_2_attn_constant_6 = getattr(self, "h/2/attn/Constant_6")()
        h_2_attn_unsqueeze_5 = getattr(self, "h/2/attn/Unsqueeze_5")(h_2_attn_cast_1)
        h_2_attn_concat_1 = getattr(self, "h/2/attn/Concat_1")(
            h_2_attn_unsqueeze_3,
            h_2_attn_unsqueeze_4,
            h_2_attn_constant_6,
            h_2_attn_unsqueeze_5,
        )
        h_2_attn_unsqueeze_6 = getattr(self, "h/2/attn/Unsqueeze_6")(h_2_attn_gather)
        h_2_attn_unsqueeze_7 = getattr(self, "h/2/attn/Unsqueeze_7")(h_2_attn_gather_1)
        h_2_attn_constant_7 = getattr(self, "h/2/attn/Constant_7")()
        h_2_attn_unsqueeze_8 = getattr(self, "h/2/attn/Unsqueeze_8")(h_2_attn_cast_1)
        h_2_attn_concat_2 = getattr(self, "h/2/attn/Concat_2")(
            h_2_attn_unsqueeze_6,
            h_2_attn_unsqueeze_7,
            h_2_attn_constant_7,
            h_2_attn_unsqueeze_8,
        )
        getitem_6 = h_2_attn_split[1]
        h_2_attn_reshape = getattr(self, "h/2/attn/Reshape")(getitem_6, h_2_attn_concat)
        getitem_7 = h_2_attn_split[0]
        h_2_attn_reshape_1 = getattr(self, "h/2/attn/Reshape_1")(
            getitem_7, h_2_attn_concat_1
        )
        h_2_attn_transpose = getattr(self, "h/2/attn/Transpose")(h_2_attn_reshape_1)
        getitem_8 = h_2_attn_split[2]
        h_2_attn_reshape_2 = getattr(self, "h/2/attn/Reshape_2")(
            getitem_8, h_2_attn_concat_2
        )
        h_2_attn_transpose_1 = getattr(self, "h/2/attn/Transpose_1")(h_2_attn_reshape_2)
        h_2_attn_gather_3 = getattr(self, "h/2/attn/Gather_3")(identity_129, range_1)
        h_2_attn_transpose_2 = getattr(self, "h/2/attn/Transpose_2")(h_2_attn_reshape)
        h_2_attn_mat_mul = getattr(self, "h/2/attn/MatMul")(
            h_2_attn_transpose, h_2_attn_transpose_2
        )
        h_2_attn_constant_8 = getattr(self, "h/2/attn/Constant_8")()
        h_2_attn_mul = getattr(self, "h/2/attn/Mul")(h_2_attn_mat_mul, h_2_attn_constant_8)
        h_2_attn_not = getattr(self, "h/2/attn/Not")(h_2_attn_gather_3)
        h_2_attn_cast_2 = getattr(self, "h/2/attn/Cast_2")(h_2_attn_not)
        h_2_attn_constant_9 = getattr(self, "h/2/attn/Constant_9")()
        h_2_attn_where = getattr(self, "h/2/attn/Where")(
            h_2_attn_cast_2, h_2_attn_constant_9, h_2_attn_mul
        )
        h_2_attn_softmax = getattr(self, "h/2/attn/Softmax")(h_2_attn_where)
        h_2_attn_mat_mul_1 = getattr(self, "h/2/attn/MatMul_1")(
            h_2_attn_softmax, h_2_attn_transpose_1
        )
        h_2_attn_transpose_3 = getattr(self, "h/2/attn/Transpose_3")(h_2_attn_mat_mul_1)
        h_2_attn_unsqueeze_9 = getattr(self, "h/2/attn/Unsqueeze_9")(h_2_attn_gather)
        h_2_attn_unsqueeze_10 = getattr(self, "h/2/attn/Unsqueeze_10")(h_2_attn_gather_1)
        h_2_attn_unsqueeze_11 = getattr(self, "h/2/attn/Unsqueeze_11")(h_2_attn_gather_2)
        h_2_attn_concat_3 = getattr(self, "h/2/attn/Concat_3")(
            h_2_attn_unsqueeze_9, h_2_attn_unsqueeze_10, h_2_attn_unsqueeze_11
        )
        h_2_attn_reshape_3 = getattr(self, "h/2/attn/Reshape_3")(
            h_2_attn_transpose_3, h_2_attn_concat_3
        )
        initializers_onnx_initializer_27 = self.initializers.onnx_initializer_27
        h_2_attn_c_proj_mat_mul = getattr(self, "h/2/attn/c_proj/MatMul")(
            h_2_attn_reshape_3, initializers_onnx_initializer_27
        )
        h_2_add = getattr(self, "h/2/Add")(h_1_add_1, h_2_attn_c_proj_mat_mul)
        h_2_mlp_layer_norm_reduce_mean = getattr(self, "h/2/mlp/layer_norm/ReduceMean")(
            h_2_add
        )
        h_2_mlp_layer_norm_sub = getattr(self, "h/2/mlp/layer_norm/Sub")(
            h_2_add, h_2_mlp_layer_norm_reduce_mean
        )
        h_2_mlp_layer_norm_constant = getattr(self, "h/2/mlp/layer_norm/Constant")()
        h_2_mlp_layer_norm_pow = getattr(self, "h/2/mlp/layer_norm/Pow")(
            h_2_mlp_layer_norm_sub, h_2_mlp_layer_norm_constant
        )
        h_2_mlp_layer_norm_reduce_mean_1 = getattr(self, "h/2/mlp/layer_norm/ReduceMean_1")(
            h_2_mlp_layer_norm_pow
        )
        h_2_mlp_layer_norm_constant_1 = getattr(self, "h/2/mlp/layer_norm/Constant_1")()
        h_2_mlp_layer_norm_add = getattr(self, "h/2/mlp/layer_norm/Add")(
            h_2_mlp_layer_norm_reduce_mean_1, h_2_mlp_layer_norm_constant_1
        )
        h_2_mlp_layer_norm_sqrt = getattr(self, "h/2/mlp/layer_norm/Sqrt")(
            h_2_mlp_layer_norm_add
        )
        h_2_mlp_layer_norm_div = getattr(self, "h/2/mlp/layer_norm/Div")(
            h_2_mlp_layer_norm_sub, h_2_mlp_layer_norm_sqrt
        )
        initializers_onnx_initializer_28 = self.initializers.onnx_initializer_28
        h_2_mlp_layer_norm_mul = getattr(self, "h/2/mlp/layer_norm/Mul")(
            h_2_mlp_layer_norm_div, initializers_onnx_initializer_28
        )
        initializers_onnx_initializer_29 = self.initializers.onnx_initializer_29
        h_2_mlp_layer_norm_add_1 = getattr(self, "h/2/mlp/layer_norm/Add_1")(
            h_2_mlp_layer_norm_mul, initializers_onnx_initializer_29
        )
        initializers_onnx_initializer_30 = self.initializers.onnx_initializer_30
        h_2_mlp_c_fc_mat_mul = getattr(self, "h/2/mlp/c_fc/MatMul")(
            h_2_mlp_layer_norm_add_1, initializers_onnx_initializer_30
        )
        h_2_mlp_act_mul = getattr(self, "h/2/mlp/act/Mul")(
            h_2_mlp_c_fc_mat_mul, h_2_mlp_c_fc_mat_mul
        )
        h_2_mlp_act_mul_1 = getattr(self, "h/2/mlp/act/Mul_1")(
            h_2_mlp_c_fc_mat_mul, h_2_mlp_act_mul
        )
        h_2_mlp_act_constant = getattr(self, "h/2/mlp/act/Constant")()
        h_2_mlp_act_mul_2 = getattr(self, "h/2/mlp/act/Mul_2")(
            h_2_mlp_act_constant, h_2_mlp_act_mul_1
        )
        h_2_mlp_act_add = getattr(self, "h/2/mlp/act/Add")(
            h_2_mlp_c_fc_mat_mul, h_2_mlp_act_mul_2
        )
        h_2_mlp_act_constant_1 = getattr(self, "h/2/mlp/act/Constant_1")()
        h_2_mlp_act_mul_3 = getattr(self, "h/2/mlp/act/Mul_3")(
            h_2_mlp_act_constant_1, h_2_mlp_act_add
        )
        h_2_mlp_act_tanh = getattr(self, "h/2/mlp/act/Tanh")(h_2_mlp_act_mul_3)
        h_2_mlp_act_constant_2 = getattr(self, "h/2/mlp/act/Constant_2")()
        h_2_mlp_act_add_1 = getattr(self, "h/2/mlp/act/Add_1")(
            h_2_mlp_act_constant_2, h_2_mlp_act_tanh
        )
        h_2_mlp_act_mul_4 = getattr(self, "h/2/mlp/act/Mul_4")(
            h_2_mlp_c_fc_mat_mul, h_2_mlp_act_add_1
        )
        h_2_mlp_act_constant_3 = getattr(self, "h/2/mlp/act/Constant_3")()
        h_2_mlp_act_mul_5 = getattr(self, "h/2/mlp/act/Mul_5")(
            h_2_mlp_act_constant_3, h_2_mlp_act_mul_4
        )
        initializers_onnx_initializer_31 = self.initializers.onnx_initializer_31
        h_2_mlp_c_proj_mat_mul = getattr(self, "h/2/mlp/c_proj/MatMul")(
            h_2_mlp_act_mul_5, initializers_onnx_initializer_31
        )
        h_2_add_1 = getattr(self, "h/2/Add_1")(h_2_add, h_2_mlp_c_proj_mat_mul)
        h_3_attn_shape = getattr(self, "h/3/attn/Shape")(h_2_add_1)
        h_3_attn_constant = getattr(self, "h/3/attn/Constant")()
        h_3_attn_gather = getattr(self, "h/3/attn/Gather")(
            h_3_attn_shape, h_3_attn_constant
        )
        h_3_attn_shape_1 = getattr(self, "h/3/attn/Shape_1")(h_2_add_1)
        h_3_attn_constant_1 = getattr(self, "h/3/attn/Constant_1")()
        h_3_attn_gather_1 = getattr(self, "h/3/attn/Gather_1")(
            h_3_attn_shape_1, h_3_attn_constant_1
        )
        h_3_attn_shape_2 = getattr(self, "h/3/attn/Shape_2")(h_2_add_1)
        h_3_attn_constant_2 = getattr(self, "h/3/attn/Constant_2")()
        h_3_attn_gather_2 = getattr(self, "h/3/attn/Gather_2")(
            h_3_attn_shape_2, h_3_attn_constant_2
        )
        h_3_attn_layer_norm_reduce_mean = getattr(self, "h/3/attn/layer_norm/ReduceMean")(
            h_2_add_1
        )
        h_3_attn_layer_norm_sub = getattr(self, "h/3/attn/layer_norm/Sub")(
            h_2_add_1, h_3_attn_layer_norm_reduce_mean
        )
        h_3_attn_layer_norm_constant = getattr(self, "h/3/attn/layer_norm/Constant")()
        h_3_attn_layer_norm_pow = getattr(self, "h/3/attn/layer_norm/Pow")(
            h_3_attn_layer_norm_sub, h_3_attn_layer_norm_constant
        )
        h_3_attn_layer_norm_reduce_mean_1 = getattr(
            self, "h/3/attn/layer_norm/ReduceMean_1"
        )(h_3_attn_layer_norm_pow)
        h_3_attn_layer_norm_constant_1 = getattr(self, "h/3/attn/layer_norm/Constant_1")()
        h_3_attn_layer_norm_add = getattr(self, "h/3/attn/layer_norm/Add")(
            h_3_attn_layer_norm_reduce_mean_1, h_3_attn_layer_norm_constant_1
        )
        h_3_attn_layer_norm_sqrt = getattr(self, "h/3/attn/layer_norm/Sqrt")(
            h_3_attn_layer_norm_add
        )
        h_3_attn_layer_norm_div = getattr(self, "h/3/attn/layer_norm/Div")(
            h_3_attn_layer_norm_sub, h_3_attn_layer_norm_sqrt
        )
        initializers_onnx_initializer_32 = self.initializers.onnx_initializer_32
        h_3_attn_layer_norm_mul = getattr(self, "h/3/attn/layer_norm/Mul")(
            h_3_attn_layer_norm_div, initializers_onnx_initializer_32
        )
        initializers_onnx_initializer_33 = self.initializers.onnx_initializer_33
        h_3_attn_layer_norm_add_1 = getattr(self, "h/3/attn/layer_norm/Add_1")(
            h_3_attn_layer_norm_mul, initializers_onnx_initializer_33
        )
        initializers_onnx_initializer_34 = self.initializers.onnx_initializer_34
        h_3_attn_c_attn_mat_mul = getattr(self, "h/3/attn/c_attn/MatMul")(
            h_3_attn_layer_norm_add_1, initializers_onnx_initializer_34
        )
        h_3_attn_constant_3 = getattr(self, "h/3/attn/Constant_3")()
        h_3_attn_split = getattr(self, "h/3/attn/Split")(
            h_3_attn_c_attn_mat_mul, h_3_attn_constant_3
        )
        h_3_attn_constant_4 = getattr(self, "h/3/attn/Constant_4")()
        h_3_attn_div = getattr(self, "h/3/attn/Div")(h_3_attn_gather_2, h_3_attn_constant_4)
        h_3_attn_cast = getattr(self, "h/3/attn/Cast")(h_3_attn_div)
        h_3_attn_cast_1 = getattr(self, "h/3/attn/Cast_1")(h_3_attn_cast)
        h_3_attn_unsqueeze = getattr(self, "h/3/attn/Unsqueeze")(h_3_attn_gather)
        h_3_attn_unsqueeze_1 = getattr(self, "h/3/attn/Unsqueeze_1")(h_3_attn_gather_1)
        h_3_attn_constant_5 = getattr(self, "h/3/attn/Constant_5")()
        h_3_attn_unsqueeze_2 = getattr(self, "h/3/attn/Unsqueeze_2")(h_3_attn_cast_1)
        h_3_attn_concat = getattr(self, "h/3/attn/Concat")(
            h_3_attn_unsqueeze,
            h_3_attn_unsqueeze_1,
            h_3_attn_constant_5,
            h_3_attn_unsqueeze_2,
        )
        h_3_attn_unsqueeze_3 = getattr(self, "h/3/attn/Unsqueeze_3")(h_3_attn_gather)
        h_3_attn_unsqueeze_4 = getattr(self, "h/3/attn/Unsqueeze_4")(h_3_attn_gather_1)
        h_3_attn_constant_6 = getattr(self, "h/3/attn/Constant_6")()
        h_3_attn_unsqueeze_5 = getattr(self, "h/3/attn/Unsqueeze_5")(h_3_attn_cast_1)
        h_3_attn_concat_1 = getattr(self, "h/3/attn/Concat_1")(
            h_3_attn_unsqueeze_3,
            h_3_attn_unsqueeze_4,
            h_3_attn_constant_6,
            h_3_attn_unsqueeze_5,
        )
        h_3_attn_unsqueeze_6 = getattr(self, "h/3/attn/Unsqueeze_6")(h_3_attn_gather)
        h_3_attn_unsqueeze_7 = getattr(self, "h/3/attn/Unsqueeze_7")(h_3_attn_gather_1)
        h_3_attn_constant_7 = getattr(self, "h/3/attn/Constant_7")()
        h_3_attn_unsqueeze_8 = getattr(self, "h/3/attn/Unsqueeze_8")(h_3_attn_cast_1)
        h_3_attn_concat_2 = getattr(self, "h/3/attn/Concat_2")(
            h_3_attn_unsqueeze_6,
            h_3_attn_unsqueeze_7,
            h_3_attn_constant_7,
            h_3_attn_unsqueeze_8,
        )
        getitem_9 = h_3_attn_split[1]
        h_3_attn_reshape = getattr(self, "h/3/attn/Reshape")(getitem_9, h_3_attn_concat)
        getitem_10 = h_3_attn_split[0]
        h_3_attn_reshape_1 = getattr(self, "h/3/attn/Reshape_1")(
            getitem_10, h_3_attn_concat_1
        )
        h_3_attn_transpose = getattr(self, "h/3/attn/Transpose")(h_3_attn_reshape_1)
        getitem_11 = h_3_attn_split[2]
        h_3_attn_reshape_2 = getattr(self, "h/3/attn/Reshape_2")(
            getitem_11, h_3_attn_concat_2
        )
        h_3_attn_transpose_1 = getattr(self, "h/3/attn/Transpose_1")(h_3_attn_reshape_2)
        h_3_attn_gather_3 = getattr(self, "h/3/attn/Gather_3")(identity_128, range_1)
        h_3_attn_transpose_2 = getattr(self, "h/3/attn/Transpose_2")(h_3_attn_reshape)
        h_3_attn_mat_mul = getattr(self, "h/3/attn/MatMul")(
            h_3_attn_transpose, h_3_attn_transpose_2
        )
        h_3_attn_constant_8 = getattr(self, "h/3/attn/Constant_8")()
        h_3_attn_mul = getattr(self, "h/3/attn/Mul")(h_3_attn_mat_mul, h_3_attn_constant_8)
        h_3_attn_not = getattr(self, "h/3/attn/Not")(h_3_attn_gather_3)
        h_3_attn_cast_2 = getattr(self, "h/3/attn/Cast_2")(h_3_attn_not)
        h_3_attn_constant_9 = getattr(self, "h/3/attn/Constant_9")()
        h_3_attn_where = getattr(self, "h/3/attn/Where")(
            h_3_attn_cast_2, h_3_attn_constant_9, h_3_attn_mul
        )
        h_3_attn_softmax = getattr(self, "h/3/attn/Softmax")(h_3_attn_where)
        h_3_attn_mat_mul_1 = getattr(self, "h/3/attn/MatMul_1")(
            h_3_attn_softmax, h_3_attn_transpose_1
        )
        h_3_attn_transpose_3 = getattr(self, "h/3/attn/Transpose_3")(h_3_attn_mat_mul_1)
        h_3_attn_unsqueeze_9 = getattr(self, "h/3/attn/Unsqueeze_9")(h_3_attn_gather)
        h_3_attn_unsqueeze_10 = getattr(self, "h/3/attn/Unsqueeze_10")(h_3_attn_gather_1)
        h_3_attn_unsqueeze_11 = getattr(self, "h/3/attn/Unsqueeze_11")(h_3_attn_gather_2)
        h_3_attn_concat_3 = getattr(self, "h/3/attn/Concat_3")(
            h_3_attn_unsqueeze_9, h_3_attn_unsqueeze_10, h_3_attn_unsqueeze_11
        )
        h_3_attn_reshape_3 = getattr(self, "h/3/attn/Reshape_3")(
            h_3_attn_transpose_3, h_3_attn_concat_3
        )
        initializers_onnx_initializer_35 = self.initializers.onnx_initializer_35
        h_3_attn_c_proj_mat_mul = getattr(self, "h/3/attn/c_proj/MatMul")(
            h_3_attn_reshape_3, initializers_onnx_initializer_35
        )
        h_3_add = getattr(self, "h/3/Add")(h_2_add_1, h_3_attn_c_proj_mat_mul)
        h_3_mlp_layer_norm_reduce_mean = getattr(self, "h/3/mlp/layer_norm/ReduceMean")(
            h_3_add
        )
        h_3_mlp_layer_norm_sub = getattr(self, "h/3/mlp/layer_norm/Sub")(
            h_3_add, h_3_mlp_layer_norm_reduce_mean
        )
        h_3_mlp_layer_norm_constant = getattr(self, "h/3/mlp/layer_norm/Constant")()
        h_3_mlp_layer_norm_pow = getattr(self, "h/3/mlp/layer_norm/Pow")(
            h_3_mlp_layer_norm_sub, h_3_mlp_layer_norm_constant
        )
        h_3_mlp_layer_norm_reduce_mean_1 = getattr(self, "h/3/mlp/layer_norm/ReduceMean_1")(
            h_3_mlp_layer_norm_pow
        )
        h_3_mlp_layer_norm_constant_1 = getattr(self, "h/3/mlp/layer_norm/Constant_1")()
        h_3_mlp_layer_norm_add = getattr(self, "h/3/mlp/layer_norm/Add")(
            h_3_mlp_layer_norm_reduce_mean_1, h_3_mlp_layer_norm_constant_1
        )
        h_3_mlp_layer_norm_sqrt = getattr(self, "h/3/mlp/layer_norm/Sqrt")(
            h_3_mlp_layer_norm_add
        )
        h_3_mlp_layer_norm_div = getattr(self, "h/3/mlp/layer_norm/Div")(
            h_3_mlp_layer_norm_sub, h_3_mlp_layer_norm_sqrt
        )
        initializers_onnx_initializer_36 = self.initializers.onnx_initializer_36
        h_3_mlp_layer_norm_mul = getattr(self, "h/3/mlp/layer_norm/Mul")(
            h_3_mlp_layer_norm_div, initializers_onnx_initializer_36
        )
        initializers_onnx_initializer_37 = self.initializers.onnx_initializer_37
        h_3_mlp_layer_norm_add_1 = getattr(self, "h/3/mlp/layer_norm/Add_1")(
            h_3_mlp_layer_norm_mul, initializers_onnx_initializer_37
        )
        initializers_onnx_initializer_38 = self.initializers.onnx_initializer_38
        h_3_mlp_c_fc_mat_mul = getattr(self, "h/3/mlp/c_fc/MatMul")(
            h_3_mlp_layer_norm_add_1, initializers_onnx_initializer_38
        )
        h_3_mlp_act_mul = getattr(self, "h/3/mlp/act/Mul")(
            h_3_mlp_c_fc_mat_mul, h_3_mlp_c_fc_mat_mul
        )
        h_3_mlp_act_mul_1 = getattr(self, "h/3/mlp/act/Mul_1")(
            h_3_mlp_c_fc_mat_mul, h_3_mlp_act_mul
        )
        h_3_mlp_act_constant = getattr(self, "h/3/mlp/act/Constant")()
        h_3_mlp_act_mul_2 = getattr(self, "h/3/mlp/act/Mul_2")(
            h_3_mlp_act_constant, h_3_mlp_act_mul_1
        )
        h_3_mlp_act_add = getattr(self, "h/3/mlp/act/Add")(
            h_3_mlp_c_fc_mat_mul, h_3_mlp_act_mul_2
        )
        h_3_mlp_act_constant_1 = getattr(self, "h/3/mlp/act/Constant_1")()
        h_3_mlp_act_mul_3 = getattr(self, "h/3/mlp/act/Mul_3")(
            h_3_mlp_act_constant_1, h_3_mlp_act_add
        )
        h_3_mlp_act_tanh = getattr(self, "h/3/mlp/act/Tanh")(h_3_mlp_act_mul_3)
        h_3_mlp_act_constant_2 = getattr(self, "h/3/mlp/act/Constant_2")()
        h_3_mlp_act_add_1 = getattr(self, "h/3/mlp/act/Add_1")(
            h_3_mlp_act_constant_2, h_3_mlp_act_tanh
        )
        h_3_mlp_act_mul_4 = getattr(self, "h/3/mlp/act/Mul_4")(
            h_3_mlp_c_fc_mat_mul, h_3_mlp_act_add_1
        )
        h_3_mlp_act_constant_3 = getattr(self, "h/3/mlp/act/Constant_3")()
        h_3_mlp_act_mul_5 = getattr(self, "h/3/mlp/act/Mul_5")(
            h_3_mlp_act_constant_3, h_3_mlp_act_mul_4
        )
        initializers_onnx_initializer_39 = self.initializers.onnx_initializer_39
        h_3_mlp_c_proj_mat_mul = getattr(self, "h/3/mlp/c_proj/MatMul")(
            h_3_mlp_act_mul_5, initializers_onnx_initializer_39
        )
        h_3_add_1 = getattr(self, "h/3/Add_1")(h_3_add, h_3_mlp_c_proj_mat_mul)
        layer_norm_f_reduce_mean = getattr(self, "layer_norm_f/ReduceMean")(h_3_add_1)
        layer_norm_f_sub = getattr(self, "layer_norm_f/Sub")(
            h_3_add_1, layer_norm_f_reduce_mean
        )
        layer_norm_f_constant = getattr(self, "layer_norm_f/Constant")()
        layer_norm_f_pow = getattr(self, "layer_norm_f/Pow")(
            layer_norm_f_sub, layer_norm_f_constant
        )
        layer_norm_f_reduce_mean_1 = getattr(self, "layer_norm_f/ReduceMean_1")(
            layer_norm_f_pow
        )
        layer_norm_f_constant_1 = getattr(self, "layer_norm_f/Constant_1")()
        layer_norm_f_add = getattr(self, "layer_norm_f/Add")(
            layer_norm_f_reduce_mean_1, layer_norm_f_constant_1
        )
        layer_norm_f_sqrt = getattr(self, "layer_norm_f/Sqrt")(layer_norm_f_add)
        layer_norm_f_div = getattr(self, "layer_norm_f/Div")(
            layer_norm_f_sub, layer_norm_f_sqrt
        )
        initializers_onnx_initializer_40 = self.initializers.onnx_initializer_40
        layer_norm_f_mul = getattr(self, "layer_norm_f/Mul")(
            layer_norm_f_div, initializers_onnx_initializer_40
        )
        initializers_onnx_initializer_41 = self.initializers.onnx_initializer_41
        layer_norm_f_add_1 = getattr(self, "layer_norm_f/Add_1")(
            layer_norm_f_mul, initializers_onnx_initializer_41
        )
        initializers_onnx_initializer_42 = self.initializers.onnx_initializer_42
        lm_head_mat_mul = getattr(self, "lm_head/MatMul")(
            layer_norm_f_add_1, initializers_onnx_initializer_42
        )
        return lm_head_mat_mul
