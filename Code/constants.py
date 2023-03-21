from torchvision import transforms

batch_size = 4096
latent_dim = 128
lr = 0.001
epochs = 100
img_size = 64

TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
])

VAL_TRANSFORM = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
])