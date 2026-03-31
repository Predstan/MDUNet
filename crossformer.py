"""

   Adapted from https://github.com/lucidrains/vit-pytorch in reference to

   @misc{wang2021crossformer,
               title   = {CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention}, 
               author  = {Wenxiao Wang and Lu Yao and Long Chen and Binbin Lin and Deng Cai and Xiaofei He and Wei Liu},
               year    = {2021},
               eprint  = {2108.00154},
               archivePrefix = {arXiv},
               primaryClass = {cs.CV}
}
   
"""
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.nn import Module


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

attn_map = []
# helperss
def cast_tuple(val, length = 1):
   return val if isinstance(val, tuple) else ((val,) * length)

# cross embed layer
class CrossEmbedLayer(Module):
   def __init__(
       self,
       dim_in,
       dim_out,
       kernel_sizes,
       stride = 2
   ):
       super().__init__()
       kernel_sizes = sorted(kernel_sizes)
       num_scales = len(kernel_sizes)

       # calculate the dimension at each scale
       dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
       dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

       self.convs = nn.ModuleList([])
       for kernel, dim_scale in zip(kernel_sizes, dim_scales):
           self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

   def forward(self, x):
       fmaps = tuple(map(lambda conv: conv(x), self.convs))
       return torch.cat(fmaps, dim = 1)
   
class Upsample(nn.Module):
   """
   An upsampling layer with an optional convolution.

   :param channels: channels in the inputs and outputs.
   :param use_conv: a bool determining if a convolution is applied.
   :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                upsampling occurs in the inner-two dimensions.
   """
   def __init__(self, dim_in, dim_out, kernel, stride, padding):
       super().__init__()
       self.channels = dim_in
       self.out_channels = dim_out
       self.conv = nn.Conv2d(self.channels, self.out_channels, kernel+1, stride=1, padding=(kernel) // 2)
       
   def forward(self, x):
       assert x.shape[1] == self.channels
       x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
       x = self.conv(x)
       return x
   
   
class TransCrossEmbedLayer(nn.Module):
   def __init__(
       self,
       dim_in,
       dim_out,
       kernel_sizes,
       stride = 2
   ):
       super().__init__()
       kernel_sizes = sorted(kernel_sizes)
       num_scales = len(kernel_sizes)

       # calculate the dimension at each scale
       dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
       dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

       self.convs = nn.ModuleList([])
       for kernel, dim_scale in zip(kernel_sizes, dim_scales):
           self.convs.append(Upsample(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2,))

   def forward(self, x):
       fmaps = tuple(map(lambda conv: conv(x), self.convs))
       return torch.cat(fmaps, dim = 1)

# dynamic positional bias
def DynamicPositionBias(dim):
   return nn.Sequential(
       nn.Linear(2, dim),
       nn.LayerNorm(dim),
       nn.ReLU(),
       nn.Linear(dim, dim),
       nn.LayerNorm(dim),
       nn.ReLU(),
       nn.Linear(dim, dim),
       nn.LayerNorm(dim),
       nn.ReLU(),
       nn.Linear(dim, 1),
       Rearrange('... () -> ...')
   )

# transformer classes
class LayerNorm(Module):
   def __init__(self, dim, eps = 1e-5):
       super().__init__()
       self.eps = eps
       self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
       self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

   def forward(self, x):
       var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
       mean = torch.mean(x, dim = 1, keepdim = True)
       return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
   return nn.Sequential(
       LayerNorm(dim),
       nn.Conv2d(dim, dim * mult, 1),
       nn.GELU(),
       nn.Dropout(dropout),
       nn.Conv2d(dim * mult, dim, 1)
   )

class Attention(Module):
   def __init__(
       self,
       dim,
       attn_type,
       window_size,
       dim_head = 32,
       dropout = 0.,
       return_map=False
   ):
       super().__init__()
       assert attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
       heads = dim // dim_head
       self.heads = heads
       self.scale = dim_head ** -0.5
       inner_dim = dim_head * heads

       self.return_map = return_map

       self.attn_type = attn_type
       self.window_size = window_size

       self.norm = LayerNorm(dim)

       self.dropout = nn.Dropout(dropout)

       self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
       self.to_out = nn.Conv2d(inner_dim, dim, 1)

       # positions

       self.dpb = DynamicPositionBias(dim // 4)
     

       # calculate and store indices for retrieving bias

       pos = torch.arange(window_size)
       grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
       grid = rearrange(grid, 'c i j -> (i j) c').contiguous()
       rel_pos = grid[:, None] - grid[None, :]
       rel_pos += window_size - 1
       rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

       self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

   def forward(self, x):
       *_, height, width, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device

       # prenorm

       x = self.norm(x)

       # rearrange for short or long distance attention

       if self.attn_type == 'short':
           x = rearrange(x, 'b d (h s1) (w s2) -> (b h w) d s1 s2', s1 = wsz, s2 = wsz).contiguous()
       elif self.attn_type == 'long':
           x = rearrange(x, 'b d (l1 h) (l2 w) -> (b h w) d l1 l2', l1 = wsz, l2 = wsz).contiguous()

       # queries / keys / values

       q, k, v = self.to_qkv(x).chunk(3, dim = 1)

       # split heads
       q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads).contiguous(), (q, k, v))
       q = q * self.scale

       sim = einsum('b h i d, b h j d -> b h i j', q, k)

       # add dynamic positional bias

       pos = torch.arange(-wsz, wsz + 1, device = device)
       rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
       rel_pos = rearrange(rel_pos, 'c i j -> (i j) c').contiguous()
       biases = self.dpb(rel_pos.float())
       rel_pos_bias = biases[self.rel_pos_indices]

       sim = sim + rel_pos_bias

       # attend

       attn = sim.softmax(dim = -1)
       attn = self.dropout(attn)

       if self.return_map:
           if self.attn_type == 'short':
               attn_map.append(rearrange(attn, '(b h w) d s1 s2 -> b d (h s1) (w s2)', h = height // wsz, w = width // wsz))
           elif self.attn_type == 'long':
               attn_map.append(rearrange(attn, '(b h w) d l1 l2 -> b d (l1 h) (l2 w)', h = height // wsz, w = width // wsz))
       

       # merge heads

       out = einsum('b h i j, b h j d -> b h i d', attn, v)
       out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = wsz, y = wsz)
       out = self.to_out(out)

       # rearrange back for long or short distance attention

       if self.attn_type == 'short':
           out = rearrange(out, '(b h w) d s1 s2 -> b d (h s1) (w s2)', h = height // wsz, w = width // wsz).contiguous()
       elif self.attn_type == 'long':
           out = rearrange(out, '(b h w) d l1 l2 -> b d (l1 h) (l2 w)', h = height // wsz, w = width // wsz).contiguous()

       return out

class Transformer(Module):
   def __init__(
       self,
       dim,
       *,
       local_window_size,
       global_window_size,
       depth = 4,
       dim_head = 32,
       attn_dropout = 0.,
       ff_dropout = 0.,
       return_map=False,
   ):
       super().__init__()
       self.layers = nn.ModuleList([])

       for _ in range(depth):
           self.layers.append(nn.ModuleList([
               Attention(dim, attn_type = 'short', window_size = local_window_size, dim_head = dim_head, dropout = attn_dropout, return_map=return_map),
               FeedForward(dim, dropout = ff_dropout),
               Attention(dim, attn_type = 'long', window_size = global_window_size, dim_head = dim_head, dropout = attn_dropout,return_map=return_map),
               FeedForward(dim, dropout = ff_dropout)
           ]))

   def forward(self, x):
       for short_attn, short_ff, long_attn, long_ff in self.layers:
           x = short_attn(x) + x
           x = short_ff(x) + x
           x = long_attn(x) + x
           x = long_ff(x) + x

       return x
   
   
class Encoder(nn.Module):

   def __init__(self,
               dims = (64, 128, 64),
               out_dim = 16,
               depth = (4, 4, 4),
               global_window_size = (8, 4, 2),
               local_window_size = 2,
               cross_embed_kernel_sizes = ((4, 8, 16, 32), (2, 4, 8,), (2, 4, 8)),
               cross_embed_strides = (2, 2, 2),
               attn_dropout = 0.2,
               ff_dropout = 0.2,
               channels = 3):
       
       super().__init__()
       
       num = len(dims)
       dim = cast_tuple(dims, num)
       depth = cast_tuple(depth, num)
       global_window_size = cast_tuple(global_window_size, num)
       local_window_size = cast_tuple(local_window_size, num)
       cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, num)
       cross_embed_strides = cast_tuple(cross_embed_strides, num)
       
       
       last_dim = dim[-1]
       dims = [channels, *dim]
       dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))
       self.layers = nn.ModuleList([])
       
       for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):
               self.layers.append(nn.ModuleList([
                   CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride = cel_stride),
                   Transformer(dim_out, local_window_size = local_wsz, global_window_size = global_wsz, depth = layers, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
               ]))
               
       self.to_out = nn.Sequential(
           nn.Conv2d(dim_out, out_dim, kernel_size=3, stride=1, padding=1),
       ) 
               
               
   def forward(self, x):
       
       for cel, transformer in self.layers:
           x = cel(x)
           x = transformer(x) 
       return self.to_out(x)       
   
class Upsampler(nn.Module):
   def __init__(self,
               dims = (128,),
               out_dim = 16,
               depth = (4, 4, 4),
               global_window_size = (8, 4, 2),
               local_window_size = 2,
               cross_embed_kernel_sizes = ((4, 8, 16, 32), (2, 4, 8,)),
               cross_embed_strides = (2, 2, 2),
               attn_dropout = 0.2,
               ff_dropout = 0.2,
               channels = 3):
       
       super().__init__()
       
       num = len(dims)
       dim = cast_tuple(dims, num)
       depth = cast_tuple(depth, num)
       global_window_size = cast_tuple(global_window_size, num)
       local_window_size = cast_tuple(local_window_size, num)
       cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, num)
       cross_embed_strides = cast_tuple(cross_embed_strides, num)
       
       dims = [channels, *dim]
       dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))
       self.layers = nn.ModuleList([])
       
       for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):
               self.layers.append(nn.ModuleList([
                   TransCrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride = cel_stride),
                   Transformer(dim_out, local_window_size = local_wsz, global_window_size = global_wsz, depth = layers, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
               ]))
               
       self.to_out = nn.Sequential(
           nn.Conv2d(dim_out, out_dim, kernel_size=3, stride=1, padding=1),
       ) 
               
               
   def forward(self, x):
       
       for cel, transformer in self.layers:
           x = cel(x)
           x = transformer(x) 
       return self.to_out(x)    
   
   

# classes
class CrossFormer(nn.Module):
   def __init__(
       self,
       *,
       dim = (128, 128, 512, 512),
       depth = (4, 4, 8, 8),
       global_window_size = (8, 4, 2, 1),
       local_window_size = 4,
       cross_embed_kernel_sizes = ((4, 8, 16, 32), (2, 4, 8,), (2, 4, 8), (2, 4)),
       cross_embed_strides = (4, 2, 2, 2),
       attn_dropout = 0.2,
       ff_dropout = 0.2,
       channels = 3+6,
   ):
       super().__init__()

       num_layers = len(dim)
       dim = cast_tuple(dim, 4)
       depth = cast_tuple(depth, 4)
       global_window_size = cast_tuple(global_window_size, 4)
       local_window_size = cast_tuple(local_window_size, 4)
       cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
       cross_embed_strides = cast_tuple(cross_embed_strides, 4)

       assert len(dim) == 4
       assert len(depth) == 4
       assert len(global_window_size) == 4
       assert len(local_window_size) == 4
       assert len(cross_embed_kernel_sizes) == 4
       assert len(cross_embed_strides) == 4

       # dimensions

       last_dim = dim[-1]
       dims = [channels, *dim]
       dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

       # layers
       self.layers = nn.ModuleList([])
       self.down_layers = nn.ModuleList([])
       for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):
               self.layers.append(nn.ModuleList([
                   CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride = cel_stride),
                   Transformer(dim_out, local_window_size = local_wsz, global_window_size = global_wsz, depth = layers, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
               ]))
                   
       i=0
       for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(
                                           reversed(dim_in_and_out), reversed(depth), reversed(global_window_size), 
                                           reversed(local_window_size), reversed(cross_embed_kernel_sizes), reversed(cross_embed_strides)
                                       ):
           
           if dim_in == 3:dim_in=64 

           if i < 2:
               self.down_layers.append(nn.ModuleList([
                       TransCrossEmbedLayer(dim_out*2, dim_in, cel_kernel_sizes, stride = cel_stride),
                       Transformer(dim_in, local_window_size = local_wsz, global_window_size = global_wsz, depth = layers, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
                   ]))
               
           if i==1:
               break
           
           i+=1
           
       # final logits
       self.to_image = nn.Sequential(
           TransCrossEmbedLayer(dim_in, dim_in, cel_kernel_sizes, stride = cel_stride),
           nn.LeakyReLU(),
           TransCrossEmbedLayer(dim_in, 64, cel_kernel_sizes, stride = cel_stride),
           nn.LeakyReLU(),
           zero_module(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))
       ) 
       
       self.to_sdf = nn.Sequential(
           nn.Conv2d(dim_in, 128, kernel_size=3, stride=1, padding=1),
           nn.LeakyReLU(),
           zero_module(nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1))
       )

   def forward(self, x):
       h =[]
       for cel, transformer in self.layers:
           x = cel(x)
           h.append(x)
           x = transformer(x) 

       for cel, transformer in self.down_layers:
           x = cel(torch.cat([x, h.pop()], 1))
           x = transformer(x)

       return self.to_image(x), self.to_sdf(x)
   
# device = "cuda"
# model = CrossFormer().to(device)
