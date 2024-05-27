
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from dataloader import CustomDataset, CollateFn
from model import CNNtoRNN

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint ...")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint ... ")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


def train():
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(256),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])
    # valid_transforms = transforms.Compose([transforms.Resize(255),
    #                                     transforms.CenterCrop(size=224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485,0.456,0.406],
    #                                                         [0.229,0.224,0.225])])
    # test_transforms = transforms.Compose([transforms.Resize(255),
    #                                     transforms.CenterCrop(size=224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485,0.456,0.406],
    #                                                         [0.229,0.224,0.225])])
    
    torch_data = CustomDataset(recipe_csv='Food Ingredients and Recipe Dataset with Image Name Mapping.csv',
                                    img_dir='Food Images', transform=train_transforms)

    train_dls = torch.utils.data.DataLoader(torch_data, 
                                           batch_size=64, shuffle=False,
                                           collate_fn = CollateFn(pad_idx=torch_data.vocab.stoi["<PAD>"]))
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(torch_data.vocab)
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 5

    writer = SummaryWriter('runs/latest')
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=torch_data.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for param in model.encoderCNN.parameters():
       param.requires_grad = False

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


    model.train()

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)

        for idx, (imgs, recipe_tokens) in tqdm(
            enumerate(train_dls), total=len(train_dls), leave=False
        ):
            imgs = imgs.to(device)
            recipe_tokens = recipe_tokens.to(device)

            outputs = model(imgs, recipe_tokens[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), recipe_tokens.reshape(-1)
            )
            
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
    
    if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

if __name__ == "__main__":
    train()