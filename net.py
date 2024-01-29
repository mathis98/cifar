import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torchvision.models import resnet18

class Net(pl.LightningModule):
	def __init__(self):
		super(Net, self).__init__()

		model = resnet18(weights=None, num_classes=10)
		model.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
		model.maxpool = nn.Identity()

		self.model = model

	def forward(self, x):

		x = self.model(x)
		# To probabilities
		output = F.log_softmax(x, dim=1)

		return output

	def training_step(self, batch, batch_idx):
		data, target = batch

		output = self(data)
		loss = F.nll_loss(output, target)

		self.log('train-loss', loss, prog_bar=True)

		return loss

	def test_step(self, batch, batch_idx):
		data, target = batch

		output = self(data)
		preds = torch.argmax(output, dim=1)
		loss = F.nll_loss(output, target)
		acc = accuracy(preds,target)

		self.log('test-loss', loss, prog_bar=True)

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters())

		return optimizer