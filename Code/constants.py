from torchvision import transforms

batch_size = 256
latent_dim = 128
lr = 0.001
epochs = 100
img_size = 64

'''TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
])'''

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406)) # range [0, 1] to [-1, 1]
])

VAL_TRANSFORM = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
])