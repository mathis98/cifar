import lightning.pytorch as pl
from torchvision import datasets, transforms
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import matplotlib.pyplot as plt
from net import Net

categories = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean, std),
])

inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# for i in range(5):
# 	img, elem = test_data[i]
# 	img = inv_normalize(img)

# 	plt.figure(figsize = (2,2))
# 	plt.imshow(img.permute(1,2,0))
# 	plt.title(categories[elem])
# 	plt.show()

net = Net()

trainer = pl.Trainer(
	max_epochs=14, 
	callbacks=[
		ModelCheckpoint(),
	]
,)

trainer.fit(net, train_loader)

trainer.test(ckpt_path='best', dataloaders=test_loader)