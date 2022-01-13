
import argparse
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

class LitAutoEncoder(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 64),
      nn.ReLU(),
      nn.Linear(64, 3))
		self.decoder = nn.Sequential(
      nn.Linear(3, 64),
      nn.ReLU(),
      nn.Linear(64, 28 * 28))

	def forward(self, x):
		embedding = self.encoder(x)
		return embedding

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)    
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)

# data
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitAutoEncoder()

# training

trainer = pl.Trainer(gpus=-1, num_nodes=1, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
    

    
from model import PedalNet
from prepare import prepare


def main(args):
    """
    This trains the PedalNet model to match the output data from the input data.

    When you resume training from an existing model, you can override hparams such as
        max_epochs, batch_size, or learning_rate. Note that changing num_channels,
        dilation_depth, num_repeat, or kernel_size will change the shape of the WaveNet
        model and is not advised.

    """

    prepare(args)
    model = PedalNet(vars(args))
    trainer = pl.Trainer(
        resume_from_checkpoint=args.model if args.resume else None,
        gpus=None if args.cpu or args.tpu_cores else args.gpus,
        tpu_cores=args.tpu_cores,
        log_every_n_steps=100,
        max_epochs=args.max_epochs,
    )

    trainer.fit(model)
    trainer.save_checkpoint(args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", nargs="?", default="data/ts9_test1_in_FP32.wav")
    parser.add_argument("out_file", nargs="?", default="data/ts9_test1_out_FP32.wav")
    parser.add_argument("--sample_time", type=float, default=100e-3)

    parser.add_argument("--num_channels", type=int, default=12)
    parser.add_argument("--dilation_depth", type=int, default=10)
    parser.add_argument("--num_repeat", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=1500)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--tpu_cores", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--model", type=str, default="models/pedalnet/pedalnet.ckpt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
