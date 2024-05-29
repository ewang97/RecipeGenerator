import torch
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu, SmoothingFunction
from train import load_checkpoint
from torch.utils.data import Subset
from dataloader import *
from model import *

# Function to compute BLEU score for a batch of predictions
def compute_bleu_score(predictions, references):
    total_bleu_score = 0.0
    smoothing_function = SmoothingFunction().method1

    for pred, ref in zip(predictions, references):
        total_bleu_score += sentence_bleu([ref], pred, smoothing_function=smoothing_function)
    
    return total_bleu_score / len(predictions)

# Validation loop
def validate(model, dataloader, criterion, dataset):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for idx, (imgs, recipe_tokens) in tqdm(
            enumerate(dataloader), total=len(dataloader), leave=False
        ):
            imgs = imgs.to(device)
            recipe_tokens = recipe_tokens.to(device)
            outputs = model(imgs, recipe_tokens[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), recipe_tokens.reshape(-1)
            )
            total_loss += loss.item()

            # outputs and targets are tokenized sentences
            predictions = outputs.argmax(dim=-1).tolist()
            references = recipe_tokens

            all_predictions.extend(predictions)
            all_references.extend(references)

    average_bleu_score = compute_bleu_score(all_predictions, all_references)
    average_loss = total_loss / len(dataloader)
    return average_loss, average_bleu_score

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset(recipe_csv='Food Ingredients and Recipe Dataset with Image Name Mapping.csv',
                            img_dir='Food Images', transform=transform)
    train_dls = torch.utils.data.DataLoader(dataset, 
                                           batch_size=64, shuffle=False,
                                           collate_fn = CollateFn(pad_idx=dataset.vocab.stoi["<PAD>"]))
    
    indices = [x for x in range(0,len(dataset),100)]  # List of indices you want to select
    validation_subset = Subset(dataset, indices)
    
    valid_dl = torch.utils.data.DataLoader(validation_subset, 
                                           batch_size=1, shuffle=False,
                                           collate_fn = CollateFn(pad_idx=dataset.vocab.stoi["<PAD>"]))
    
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 3e-4
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint = torch.load('my_checkpoint.pth.tar')
    load_checkpoint(checkpoint, model, optimizer)

    print("Average Loss, Average BLEU score: ")
    print(validate(model,valid_dl,criterion,validation_subset))