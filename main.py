import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = models.vgg19(weights='VGG19_Weights.DEFAULT').features[:29]
        self.chosen_features = [0, 2, 5, 10, 19]
        for (index, i) in enumerate(self.model):
            if isinstance(i, nn.MaxPool2d):
                self.model[index] = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = []
        for (layer_num, layer) in enumerate(self.model):
            x = layer(x)
            if layer_num in self.chosen_features:
                features.append(x)
        return features


def load_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize((336, 640)), transforms.ToTensor()])
    image = transform(image)

    return image


original_image = load_image('Image.jpg').to(device)
masked_image = original_image.clone().requires_grad_(True).to(device)
design_image = load_image('Design.jpg').to(device)


epochs = 6000
learning_rate = 0.01
alpha = 1
beta = 0.0001
model = VGG().to(device).eval()
optimizer = optim.Adam([masked_image], lr=learning_rate)
Mse = nn.MSELoss()
for epoch in range(epochs):
    generated_features = model(masked_image)
    original_features = model(original_image)
    design_features = model(design_image)
    loss_content = style_loss = 0
    for original_feature, masked_feature, design_feature in zip(original_features, generated_features, design_features):
        loss_content += Mse(original_feature, masked_feature)
        G = (masked_feature.view(masked_feature.shape[0], -1))@(masked_feature.view(masked_feature.shape[0], -1).t())
        A = (design_feature.view(design_feature.shape[0], -1))@(design_feature.view(design_feature.shape[0], -1).t())
        style_loss += Mse(G, A)
    total_loss = alpha * loss_content + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(epoch, total_loss)
    if epoch % 10 == 0:
        with torch.no_grad():
            image = masked_image.clone()
            transform = transforms.Compose([transforms.ToPILImage()])
            image = transform(image)
            image.save('Result.jpg')


