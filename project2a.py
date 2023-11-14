import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from mingpt.trainer import Trainer
import os
set_seed(3407)

# Custom dataset class for the Red Pajama dataset


class RedPajamaDataset(Dataset):
    def __init__(self, data, max_length=1024):
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token_id = 50256
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        # Tokenize the text
        tokens = self.tokenizer.encode(
            text, add_special_tokens=True, max_length=self.max_length, truncation=True, return_tensors='pt', padding=True)
        # Split the tokens into chunks of max_length
        # Shift the tokens to get targets (excluding the [CLS] token)
        target_tokens = tokens[:, 1:].clone()  # Exclude the [CLS] token
        # Exclude the last token to match the shifted targets
        tokens = tokens[:, :-1]

        return tokens, target_tokens


def batch_end_callback(trainer):
    if trainer.iter_num % 10 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


if __name__ == '__main__':

    # load in the dataset, on the supercomputer this will be the path to the pile
    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-1T-Sample", 'plain_text', streaming=False)
    dataset = dataset['train']
    print('Loaded Dataset')
    data = RedPajamaDataset(dataset)
    print('Instatiated Dataset Class')

    # load in an instance of the model
    checkpoints = os.listdir('./checkpoints')
    checkpoints.sort()
    # create a GPT instance
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = data.vocab_size
    model_config.block_size = 1023
    model_config.checkpoint = 'checkpoints/' + \
        checkpoints[-1] if checkpoints else None  # This is a change
    model = GPT(model_config)

    # create a trainer object
    train_config = Trainer.get_default_config()
    # the model we're using is so small that we can go a bit faster
    train_config.learning_rate = 5e-4
    train_config.max_iters = 1000 + \
        model.iter_num if model_config.checkpoint else 1000  # This is a change
    train_config.num_workers = 0
    train_config.checkpoint_iters = 100     # This is a change
    train_config.batch_size = 1
    trainer = Trainer(train_config, model, data)

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
