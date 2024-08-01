import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, embed_size=768, patch_size=16, channels=3, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # Version 1.0
        # self.patch_projection = nn.Sequential(
        #     Rearrange("b c (h h1) (w w1) -> b (h w) (h1 w1 c)", h1=patch_size, w1=patch_size),
        #     nn.Linear(patch_size * patch_size * channels, embed_size)
        # )

        # Version 2.0
        self.patch_projection = nn.Sequential(
            nn.Conv2d(channels, embed_size, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embed_size))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_projection(x)
        # prepend the cls token to the input
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, n_heads=8, dropout=0.):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 768 dim
        self.n_heads = n_heads  # 8
        self.head_dim = self.embed_dim // self.n_heads  # 768/8 = 96. each key,query,value will be of 96d
        self.scale = self.head_dim ** -0.5

        self.attn_drop = nn.Dropout(dropout)
        # key,query and value matrixes
        self.to_qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
           x : a unified vector of key query value
        Returns:
           output vector from multihead attention
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim=768, depth=12, n_heads=8, mlp_expansions=4, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, MultiHeadAttention(dim, n_heads, dropout))),
                Residual(FeedForward(dim, dim * mlp_expansions, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, dim=768,
                 patch_size=16,
                 channels=3,
                 img_size=224,
                 depth=12,
                 n_heads=8,
                 mlp_expansions=4,
                 dropout=0.,
                 num_classes=0,
                 global_pool='avg'):
        super(VisionTransformer, self).__init__()
        assert global_pool in ('avg', 'token')
        self.global_pool = global_pool
        self.patch_embedding = PatchEmbedding(dim, patch_size, channels, img_size)
        self.transformer = Transformer(dim, depth, n_heads, mlp_expansions, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) if num_classes > 0 else nn.Identity()

    def forward(self, img):
        x = self.patch_embedding(img)
        x = self.transformer(x)
        x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.mlp_head(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.randn((16, 3, 224, 224)).to(device)
    vit = VisionTransformer(num_classes=4, global_pool="token").to(device)
    output = vit(images)
    print(output)
    torch.save(vit.state_dict(), "model.pth")
