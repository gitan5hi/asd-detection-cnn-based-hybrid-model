import torch
import torch.nn as nn   # contains neural network

# TEMPORAL ATTENTION MODULE
class TemporalAttention(nn.Module):               # helps models focus on the important frames in the sequence
    def __init__(self, input_dim):              # input_dim = number of features = 768
        super(TemporalAttention, self).__init__()
        
        ## neural network that calculates importance
        self.attention = nn.Sequential(           
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)    ##inputs one score per time step (how important is this moment?)
        )

    def forward(self, x):   # what happens when data passes through the model
        """
        x shape: (B, T, F)
        """
        # Compute attention scores
        attn_weights = self.attention(x)   # (B, T, 1)
        
        # Normalize scores
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum of features
        context = torch.sum(attn_weights * x, dim=1)  # (B, F)
        
        return context, attn_weights

## FUSION MODEL

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        
        # Total features after fusion
        self.fusion_dim = 256 * 3  # 768
        
        # Attention layer
        self.attention = TemporalAttention(self.fusion_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, flow_feat, skel2d_feat, skel3d_feat):
        """
        Inputs:
        flow_feat  : (B, T, 256)
        skel2d_feat: (B, T, 256)
        skel3d_feat: (B, T, 256)
        """

        # STEP 1: FUSION
        fused = torch.cat([flow_feat, skel2d_feat, skel3d_feat], dim=-1)
        # Shape: (B, T, 768)

        # STEP 2: ATTENTION
        
        context, attn_weights = self.attention(fused)
        # context: (B, 768)

        # STEP 3: CLASSIFICATION
        output = self.classifier(context)  # (B, 2)

        return output, attn_weights