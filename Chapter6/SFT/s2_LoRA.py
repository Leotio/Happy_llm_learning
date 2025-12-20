import torch
import torch.nn as nn
import torch.functional as F
import re
target_modules = ["q_proj","v_proj"]
# 找到模型的各个组件中，名字⾥带"q_proj"，"v_proj"的
target_module_found = re.fullmatch(self.peft_config.target_modules, key)

# 这⾥的 key，是模型的组件名
class LoraLayer:
    def __init__(
        self,
        r: int, # LoRA 的秩
        lora_alpha: int, # 归⼀化参数
        lora_dropout: float, # LoRA 层的 dropout ⽐例
        merge_weights: bool, # eval 模式中，是否将 LoRA 矩阵的值加到原权重矩阵上
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # LoRA 层
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs,
    ):
        # 继承两个基类的构造函数
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, 
            merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # 参数矩阵 A
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # 参数矩阵 B
            self.lora_B = nn.Linear(r, out_features, bias=False)
            # 归⼀化系数
            self.scaling = self.lora_alpha / self.r
            # 冻结原参数，仅更新 A 和 B
            self.weight.requires_grad = False
            # 初始化 A 和 B
            self.reset_parameters()
            if fan_in_fan_out:
                self.weight.data = self.weight.data.T

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (torch.transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling)
                self.merged = False
            return F.linear(x, torch.transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        elif self.r > 0 and not self.merged:
            result = F.linear(x, torch.transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, torch.transpose(self.weight, self.fan_in_fan_out), bias=self.bias)