import torch
import torch.nn as nn

# 实现Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = 224 * 224 / 4 * 4
        self.patches_resolution = [224/4, 224/4]
        
    def forward(self, x):
        # x:[B, 3, H, W]
        x = self.patch_embed(x) # x:[B, embed_dim, h', w']
        x = x.flatten(2) # x:[B, embed_dim, h'*w']
        x = x.transpose([0,2,1]) # x:[B, h'*w', embed_dim]
        x = self.norm(x)
        return x        
        
        
    
class SwinTransformer(nn.Module):
    pass
    
    
    
    
    
swin = SwinTransformer()
x = paddle.randn([4, 3, 224, 224])
y = swin(x)
print(y.shape)
paddle.summary(swin, (4, 3, 224, 224))    

