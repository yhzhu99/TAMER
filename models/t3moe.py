import math

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from typing import List

from einops import rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

class Task(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(Task, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Theta
        self.t_k = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.t_v = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.t_q = nn.Parameter(torch.randn(hidden_dim, input_dim))

    def loss(self, f, x: Tensor) -> Tensor:
        train_view = self.t_k @ x.t()
        label_view = self.t_v @ x.t()
        return nn.functional.mse_loss(f(train_view.t()), label_view.t())

class Recon(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Recon, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.GELU(),
            nn.Linear(hidden_dim//4, hidden_dim)
        )
    
    def forward(self, x):
        return x + self.mlp(x)  # Residual connection

class Learner:
    def __init__(self, task: Task, hidden_dim: int, lr: float = 0.001):
        self.task = task
        self.model = Recon(hidden_dim)
        self.lr = lr
        
        # Ensure all parameters have requires_grad=True
        for param in self.model.parameters():
            param.requires_grad = True

    def train(self, x: Tensor):
        self.model.to(x.device)
        # Temporarily enable gradients, even if we're in a no_grad context
        with torch.enable_grad():
            loss = self.task.loss(self.model, x)
            grad_fn = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )
        
        # Update parameters manually
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), grad_fn):
                param.sub_(self.lr * grad)

    def predict(self, x: Tensor) -> Tensor:
        view = self.task.t_q @ x.t()
        return self.model(view.t())

class TTLinear(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TTLinear, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.task = Task(input_dim, hidden_dim)
        self.learner = Learner(self.task, hidden_dim)

    def forward(self, in_seq: List[Tensor]) -> Tensor:
        out_seq = []
        for tok in in_seq:
            self.learner.train(tok)
            hidden = self.learner.predict(tok)
            out_seq.append(hidden)
        return torch.stack(out_seq)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = - 1)

def pad_to_multiple(
    tensor,
    multiple,
    dim = -1,
    value = 0
):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    if m.is_integer():
        return False, tensor

    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma

# expert

def FeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def GLUFeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

# main class

class T3MOELayer(Module):
    def __init__(
        self,
        dim,
        *,
        num_experts = 4,
        expert_mult = 4,
        dropout = 0.,
        geglu = False
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.num_experts = num_experts

        self.to_slot_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_experts, bias = False),
            Rearrange('b n (e d) -> b e n d', e = num_experts),
            RMSNorm(dim)
        )

        expert_klass = GLUFeedForward if geglu else FeedForward

        self.experts = nn.ModuleList([
            expert_klass(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)
        ])

        self.ttt_layers = nn.ModuleList([
            TTLinear(input_dim=dim, hidden_dim=dim) for _ in range(num_experts)
        ])

    def forward(self, x, mask = None):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        seq_len, is_image, num_experts = x.shape[-2], x.ndim == 4, self.num_experts

        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')

        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)

        x = self.norm(x)

        # dynamic slot embeds
        # first average consecutive tokens, by number of experts
        # then, for each position, project out to that number of expert slot tokens
        # there should be # slots ~= sequence length, like in a usual MoE with 1 expert

        is_padded, x = pad_to_multiple(x, num_experts, dim = -2)

        if is_padded:
            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool)

            _, mask = pad_to_multiple(mask, num_experts, dim = -1, value = False)

        x_segmented = rearrange(x, 'b (n e) d -> b n e d', e = num_experts)

        if exists(mask):
            segmented_mask = rearrange(mask, 'b (n e) -> b n e', e = num_experts)
            x_segmented = x_segmented.masked_fill(~rearrange(segmented_mask, '... -> ... 1'), 0.)

        # perform a masked mean

        if exists(mask):
            num = reduce(x_segmented, 'b n e d -> b n d', 'sum')
            den = reduce(segmented_mask.float(), 'b n e -> b n 1', 'sum').clamp(min = 1e-5)
            x_consecutive_mean = num / den
            slots_mask = segmented_mask.any(dim = -1)
        else:
            x_consecutive_mean = reduce(x_segmented, 'b n e d -> b n d', 'mean')

        # project to get dynamic slots embeddings
        # could potentially inject sinusoidal positions here too before projection

        slot_embeds = self.to_slot_embeds(x_consecutive_mean)

        logits = einsum('b n d, b e s d -> b n e s', x, slot_embeds)

        # account for key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            slots_mask = rearrange(slots_mask, 'b s -> b 1 1 s')

            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
            logits = logits.masked_fill(~slots_mask, -torch.finfo(logits.dtype).max)

        # get dispatch and combine weights (softmax across right dimensions)

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # derive slots by weighted average of input tokens using the dispatch weights from above

        slots = einsum('b n d, b n e s -> e b s d', x, dispatch_weights)

        # route the slots per expert to each expert

        out = []
        for slots_per_expert, expert, ttt_layer in zip(slots, self.experts, self.ttt_layers):
            tmp = expert(slots_per_expert)  # Apply feedforward expert
            tmp = ttt_layer(tmp)  # Apply TTLinear layer
            out.append(tmp)

        out = torch.stack(out)

        # combine back out

        out = rearrange(out, 'e b s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')

        return out[:, :seq_len]


class MCGRU(nn.Module):
    def __init__(self, lab_dim, demo_dim, hidden_dim: int=32, feat_dim: int=4):
        super().__init__()
        self.lab_dim = lab_dim
        self.demo_dim = demo_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.demo_proj = nn.Linear(demo_dim, hidden_dim)
        self.lab_proj = nn.Linear(lab_dim, lab_dim)
        self.grus = nn.ModuleList(
            [
                nn.GRU(1, feat_dim, num_layers=1, batch_first=True)
                for _ in range(lab_dim)
            ]
        )
        self.out_proj = nn.Linear(lab_dim*feat_dim+hidden_dim, hidden_dim)
    def forward(self, x, static):
        # x: [bs, time_steps, lab_dim]
        # static: [bs, demo_dim]
        bs, time_steps, lab_dim = x.shape
        demo = self.demo_proj(static) # [bs, hidden_dim]
        x = self.lab_proj(x)
        out = torch.zeros(bs, time_steps, self.lab_dim, self.feat_dim).to(x.device)
        for i, gru in enumerate(self.grus):
            cur_feat = x[:, :, i].unsqueeze(-1)
            cur_feat = gru(cur_feat)[0]
            out[:, :, i] = cur_feat
        out = out.flatten(2) # b t l f -> b t (l f)
        out = out[:,-1,:] # b t (l f) -> b (l f)
        # concat demo and out
        out = torch.cat([demo, out], dim=-1)
        out = self.out_proj(out)
        return out

def concatenate_inputs(x, static):
    B, time_steps, lab_dim = x.shape
    _, demo_dim = static.shape
    # Expand static to match the time dimension of x
    static_expanded = static.unsqueeze(1).expand(-1, time_steps, -1)
    # Concatenate along the last dimension
    concatenated = torch.cat([static_expanded, x], dim=-1)
    return concatenated

class T3MOE(nn.Module):
    def __init__(self, lab_dim, demo_dim, hidden_dim: int=32, act_layer=nn.GELU, drop=0.1, **kwargs):
        super().__init__()
        # self.hidden_dim = hidden_dim
        self.mcgru_layer = MCGRU(lab_dim=lab_dim, demo_dim=demo_dim, hidden_dim=hidden_dim)
        self.gru = nn.GRU(input_size=lab_dim+demo_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.t3moe_layer = T3MOELayer(dim=hidden_dim, num_experts=16)
        self.act = act_layer()
        self.dropout = nn.Dropout(drop)
    def forward(self, x, static, mask, **kwargs):
        # x = concatenate_inputs(x, static)
        # x = self.gru(x)[1]
        x = self.mcgru_layer(x, static).unsqueeze(dim=0)
        out = self.t3moe_layer(x) + x
        out = self.act(out)
        out = self.dropout(out)
        return out.squeeze(dim=0)