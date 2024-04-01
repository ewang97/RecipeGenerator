from train import load_checkpoint
from dataloader import *
from model import *
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse



def generate_recipe(img_path, model, device, dataset):

    model.eval()
    transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])

    model.eval()
    test_img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
   
    print(" ".join(model.recipe_generate(test_img.to(device), dataset.vocab)))

    model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a recipe for an image")

    parser.add_argument("--img_path", dest="img", 
                        action="store", default="./Food Images/-em-stracciatella-tortoni-em-cake-with-espresso-fudge-sauce-242605.jpg")
    parser.add_argument("--checkpoint", dest="checkpoint", 
                        action="store", default="my_checkpoint.pth.tar")
    parser.add_argument("--gpu", dest="gpu", 
                        action="store", default="gpu")
    args = parser.parse_args()
    
    if args.gpu == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset(recipe_csv='Food Ingredients and Recipe Dataset with Image Name Mapping.csv',
                            img_dir='Food Images', transform=transforms);
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 3e-4
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    checkpoint = torch.load(args.checkpoint)
    load_checkpoint(checkpoint, model, optimizer)
    print("Recipe Output: ")
    print(generate_recipe(args.img, model, device, dataset))