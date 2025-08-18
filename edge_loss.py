import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelEdgeLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SobelEdgeLoss, self).__init__()
        self.reduction = reduction

        # Define Sobel kernels for X and Y (3x3)
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]]]], dtype=torch.float32)

        # Register as buffers so they're moved with the model
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        # Both pred and target should be [B, C, H, W] in range [0, 1]
        # rescaling from [-1,1 ] to [0,1]
        pred = (pred+1)/2
        target = (target+1)/2

        def sobel_edges(img):
            # Apply sobel filter per channel (grouped conv)
            grad_x = F.conv2d(img, self.sobel_x.expand(img.shape[1], -1, -1, -1), padding=1, groups=img.shape[1])
            grad_y = F.conv2d(img, self.sobel_y.expand(img.shape[1], -1, -1, -1), padding=1, groups=img.shape[1])
            return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        pred_edges = sobel_edges(pred)
        target_edges = sobel_edges(target)

        loss = F.l1_loss(pred_edges, target_edges, reduction=self.reduction)
        return loss