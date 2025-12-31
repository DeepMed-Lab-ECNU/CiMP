import torch
import numpy as np
from torch import nn
import torch.nn.functional as F 
from nystrom_attention import NystromAttention
from timm.models.vision_transformer import Attention


def build_snn_block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


class AttnNetGated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(AttnNetGated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class GroupmsaOmics(nn.Module):
    """
    Multi-modal survival analysis model based on multi-omics data
    
    Args:
        omic_sizes: List of feature dimensions for each omics data type
        model_size_omic: Model size configuration ('small')
        n_classes: Number of output classes
        dropout: Dropout rate
    """
    def __init__(
        self,
        omic_sizes: list = [82, 305, 493, 415, 1391, 422],
        model_size_omic: str = 'small',
        n_classes: int = 4,
        dropout: float = 0.25
    ):
        super(GroupmsaOmics, self).__init__()
        
        self.dim = 64
        self.n_classes = n_classes
        
        # Model dimension configuration
        self.size_dict_omic = {'small': [256, 64]}  # First element is SNN output dimension
        
        # Get hidden dimensions
        hidden_dims = self.size_dict_omic[model_size_omic]
        
        # Transformer encoder layer for omics features
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            activation='relu'
        )
        self.omic_transformer = nn.TransformerEncoder(
            omic_encoder_layer,
            num_layers=2
        )
        
        # Attention mechanism for omics features
        self.omic_attention_head = AttnNetGated(
            L=hidden_dims[1],
            D=hidden_dims[1],
            dropout=dropout,
            n_classes=1
        )
        
        # Projection layer for final omics representation
        self.omic_rho = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bottleneck tokens for model capacity
        self.bottleneck = nn.Parameter(torch.randn(2, 1, self.dim))
        
        # Build separate SNN networks for each omics data type
        sig_networks = []
        for input_dim in omic_sizes:
            # Build SNN blocks for each omics type
            fc_blocks = [build_snn_block(dim1=input_dim, dim2=hidden_dims[0])]
            
            # Add additional layers for remaining dimensions
            for i in range(len(hidden_dims) - 1):
                fc_blocks.append(
                    build_snn_block(
                        dim1=hidden_dims[i],
                        dim2=hidden_dims[i + 1],
                        dropout=0.25
                    )
                )
            
            sig_networks.append(nn.Sequential(*fc_blocks))
        
        self.sig_networks = nn.ModuleList(sig_networks)
        
        # Intra-modal attention mechanism
        self.intra_attn = Attention(dim=64, num_heads=2)
        
        # Final classifier layer
        self.classifier = nn.Linear(hidden_dims[1], n_classes)
        
        # CLS token for aggregated representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
    
    def forward(self, x_omic):
        """
        Forward pass through the model
        
        Args:
            x_omic: List of omics features from different modalities
            
        Returns:
            cls_token: CLS token representation for downstream tasks
            bottleneck_1: First bottleneck feature representation
            bottleneck_2: Second bottleneck feature representation
        """
        # Ensure all inputs are float tensors
        x_omic = [feat.float() if isinstance(feat, torch.Tensor) 
                 else torch.tensor(feat).float() for feat in x_omic]
        
        # Process each omics type through its respective network
        h_omic = [
            self.sig_networks[idx].forward(sig_feat)
            for idx, sig_feat in enumerate(x_omic)
        ]
        
        # Stack all processed features
        h_omic_bag = torch.stack(h_omic)  # Shape: [6, 1, 64]
        
        # Apply intra-modal attention
        h_omic_bag = self.intra_attn(h_omic_bag)
        
        # Add CLS token for aggregation
        h_omic_bag = torch.cat((self.cls_token, h_omic_bag), dim=0)
        
        # Add bottleneck tokens
        h_omic_bag = torch.cat((self.bottleneck, h_omic_bag), dim=0)
        
        # Apply transformer encoding
        h_omic_trans = self.omic_transformer(h_omic_bag)
        
        # Apply attention pooling
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        
        # Extract specific tokens for output
        cls_token = h_omic[2, :].unsqueeze(0)    # CLS token (1, 64)
        bottleneck_1 = h_omic[1, :].unsqueeze(0)  # First bottleneck feature
        bottleneck_2 = h_omic[0, :].unsqueeze(0)  # Second bottleneck feature
        
        return cls_token, bottleneck_1, bottleneck_2


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,  # number of landmarks
            pinv_iterations = 6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class TransMIL(nn.Module):
    def __init__(self):
        super(TransMIL, self).__init__()
        
        self.dim = 64
        self.n_classes = 2
        self._fc1 = nn.Sequential(nn.Linear(1024, self.dim), nn.ReLU()) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))   
        
        self.layer1 = TransLayer(dim=self.dim)
        self.layer2 = TransLayer(dim=self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self._fc2 = nn.Linear(self.dim, self.n_classes)

    def forward(self, x, bottleneck_1, bottleneck_2):    
        x = x.cuda()
        h = self._fc1(x)  # (B, n, dim)
    
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(x)
        h = torch.cat((cls_tokens, h), dim=1)
        bottleneck_1 = bottleneck_1.unsqueeze(1)
        bottleneck_2 = bottleneck_2.unsqueeze(1)
    
        h = torch.cat((bottleneck_2, h), dim=1)
        h = torch.cat((bottleneck_1, h), dim=1)
        
        #---->Translayer x1
        h = self.layer1(h)  # (B, n, dim)
        
        #---->Translayer x2
        h = self.layer2(h)  # (B, n, dim)
       
        cls_token = h[:, 2]  # (B, 1, dim)
        bottleneck_1 = h[:,0]
        bottleneck_2 = h[:,1]
        
        return  cls_token ,bottleneck_1 ,bottleneck_2
    

class CIMP(nn.Module):
    def __init__(self):
        super(CIMP, self).__init__()
        self.dim = 64
        self.n_classes = 4
        
        # Encoders for genomic and image data
        self.gene_encoder = GroupmsaOmics()
        self.img_encoder = TransMIL()
        
        # Feature processing layers
        self._fc3 = nn.Sequential(nn.Linear(512, self.dim), nn.ReLU())  # Clinical features
        self._fc1 = nn.Sequential(nn.Linear(128, self.dim), nn.ReLU())  # Gene+Image features
        self._fc2 = nn.Linear(self.dim*2, self.n_classes)               # Final classification
        
    def forward(self, gene, img, clinical):
        # Extract features from genomic data
        ft_gene, bottleneck_1, bottleneck_2 = self.gene_encoder(gene)
        
        # Extract features from WSI with cross-modal guidance
        ft_wsi, _, _ = self.img_encoder(img, bottleneck_1, bottleneck_2)
        
        # Fuse genomic and image features
        embeddings = torch.cat((ft_gene, ft_wsi), dim=1)  # (B, 128)
        
        # Process clinical features (average pooling + projection)
        clinical = clinical.squeeze(dim=0)
        cli_emb = torch.mean(clinical, dim=0, keepdim=True)
        cli_emb = self._fc3(cli_emb)
        
        # Project gene+image features and fuse with clinical features
        embeddings = self._fc1(embeddings)
        embeddings = torch.cat((embeddings, cli_emb), dim=1)  # (B, 128)
        
        # Compute survival predictions
        logits = self._fc2(embeddings)
        hazards = torch.sigmoid(logits)                  # Hazard probabilities
        scores = torch.cumprod(1 - hazards, dim=1)       # Survival probabilities
        
        return hazards, scores


def build_model(config):
    print("Init Model: {}\n".format(config.MODEL.NAME))
    if config.MODEL.NAME == 'GroupMSA':
        model = GroupmsaOmics()
    elif config.MODEL.NAME == 'CIMP':
        model=CIMP()
   
    return model


def build_loss():
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    sur_ce_loss = survival_loss

    return {"CELoss": ce_loss, "BCELoss": bce_loss, "SURCELoss": sur_ce_loss}


def survival_loss(hazards, S, Y, c, alpha=0.5, eps=1e-7):
    Y=Y.unsqueeze(1)
    c=c.unsqueeze(1)
 
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
   
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    
    return loss
   