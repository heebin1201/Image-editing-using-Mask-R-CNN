import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

def generate_dummy_data(num_samples, input_channels, input_size):
    data = torch.randn((num_samples, input_channels, input_size, input_size))
    labels = torch.randint(0, 2, (num_samples, 9, 32, 32)).float()  # Adjust based on your input size
    return data, labels

class RPNClassificationModel(nn.Module):
    def __init__(self, backbone, num_anchors):
        super(RPNClassificationModel, self).__init__()

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Exclude the last 2 layers (avgpool and fc)
        self.cls_conv = nn.Conv2d(512, num_anchors, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        cls_scores = self.cls_conv(x)
        cls_scores = self.sigmoid(cls_scores)
        return cls_scores
    
resnet50_backbone = models.resnet50(pretrained=True)
backbone = nn.Sequential(*list(resnet50_backbone.children())[:-2])

num_anchors = 9
rpn_cls_model = RPNClassificationModel(backbone, num_anchors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                                                      
rpn_cls_model.to(device)

num_samples = 1000
input_channels = 3
input_size = 256

dummy_data, dummy_labels = generate_dummy_data(num_samples, input_channels, input_size)

dummy_data = dummy_data.to(device)
dummy_labels = dummy_labels.to(device)

dummy_dataset = TensorDataset(dummy_data, dummy_labels)
dummy_dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

criterion = nn.BCELoss()
optimizer = optim.SGD(rpn_cls_model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 1000
for epoch in range(num_epochs):
    for inputs, labels in dummy_dataloader:
        optimizer.zero_grad()
        outputs = rpn_cls_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

print("Training complete.")