from torchvision import transforms 

transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5), (0.5)) # grayscale channel normalization
])

