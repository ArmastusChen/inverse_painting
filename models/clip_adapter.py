
import torch
import torch.nn as nn
import torch.nn.functional as F

class NextImageFeaturePredictor(nn.Module):
    def __init__(self, input_feature_dim=768, output_feature_dim=768, hidden_dim=1024):
        super(NextImageFeaturePredictor, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.hidden_dim = hidden_dim
        
        # Since we concatenate the current and final features, the input dimension is doubled
        self.fc1 = nn.Linear(self.input_feature_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.output_feature_dim)

    def forward(self, current_img_feature, final_img_feature):
        # Concatenate the current and final image features along the last dimension
        x = torch.cat((current_img_feature, final_img_feature), dim=-1)
        # print(current_img_feature.shape, final_img_feature.shape)
        # print(x.shape)
        x = x.view(-1, self.input_feature_dim * 2)  # Flatten the input for fully connected layers
        # print(x.shape)

        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # exit()
        
        # Reshape the output to match the desired output shape
        x = x.view(-1, 1, 768)
        return x

# # Parameters
# input_feature_dim = 768  # Feature dimension of each vector in the sequence
# output_feature_dim = 768  # Same as input for predicting next feature
# hidden_dim = 1024  # Example hidden dimension, can be adjusted

# # Initialize the model
# model = NextImageFeaturePredictor(input_feature_dim, output_feature_dim, hidden_dim)

# # Example tensors for current and final image features
# current_img_feature = torch.randn(1, 50, 768)
# final_img_feature = torch.randn(1, 50, 768)

# # Predict the next image feature
# next_img_feature = model(current_img_feature, final_img_feature)
# print(next_img_feature.shape)  # Expected output shape: (1, 50, 768)
