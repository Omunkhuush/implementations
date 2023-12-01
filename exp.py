import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models.attention import SimpleModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a sample image
image_path = './exp_ganzo.png'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Preprocess the image
input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Create an instance of the model
model = SimpleModel().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (you might need a larger dataset for meaningful training)
num_epoch = 5000
for epoch in range(num_epoch):
    # Forward pass
    output = model(input_image)

    # Compute loss
    loss = criterion(output, input_image)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')

# Visualize the original and reconstructed images on the GPU
with torch.no_grad():
    reconstructed_image = model(input_image).cpu()  # Move reconstructed image to CPU

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(transforms.ToPILImage()(input_image.cpu().squeeze(0)))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Reconstructed Image')
plt.imshow(transforms.ToPILImage()(reconstructed_image.squeeze(0)))
plt.axis('off')

plt.show()