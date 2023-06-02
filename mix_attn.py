import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, Block, Attention


class Attention_mixed(Attention):
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



def replace_attn(model):
    if not isinstance(model, VisionTransformer):
        raise NotImplementedError('replace_attn only supports timm VisionTransformer')

    for name, module in model.named_modules():
        if isinstance(module, Block):
            dim = module.attn.qkv.in_features
            if module.attn.qkv.bias is None:
                qkv_bias = False
            else:
                qkv_bias = True
            module.attn = Attention_mixed(dim, module.attn.num_heads, qkv_bias, module.attn.attn_drop.p, module.attn.proj_drop.p)

    return model


if __name__ == '__main__':

    model = timm.create_model('vit_tiny_patch16_224')
    print(model)

    model = replace_attn(model)
    print(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(output.shape)