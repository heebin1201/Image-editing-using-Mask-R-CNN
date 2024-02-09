
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, pool_height, pool_width, num_classes):
        super(Classifier, self).__init__()
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(512, 10, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(10, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(10, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(10 * 7 * 7, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(10 * 7 * 7, num_classes * 4)  # 수정된 부분

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.bn2(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)

        x = x.view(x.size(0), -1)
        print(x.shape)
        mrcnn_class_logits = self.linear_class(x.view(1, -1))
        print('mrcnn_class_logtis', mrcnn_class_logits.shape)
        mrcnn_probs = self.softmax(mrcnn_class_logits)
        print(mrcnn_probs.shape)

        mrcnn_bbox = self.linear_bbox(x.view(1, -1))  # 수정된 부분
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size(0), -1, 4)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]