import torch
import torch.nn as nn

# Define the SelfAttention layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Project input feature maps into query, key, and value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)

        # Calculate attention weights
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)

        # Apply attention to the value
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Update the input feature map with the attended values
        out = self.gamma * out + x

        return out
    
# Define a simple model using SelfAttention
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.self_attention = SelfAttention(8)
        self.conv2 = nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.self_attention(x)
        x = self.conv2(x)
        return x