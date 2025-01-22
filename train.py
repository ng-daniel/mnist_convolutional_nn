import torch
from torchmetrics import Accuracy
import data, model, engine, utils

"""
The main file for running the program and starting the training process.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01

train_data, test_data = data.download_MNIST()
train_dataloader, test_dataloader = data.make_dataloaders(
    train_data,
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model_VGG = model.VGG().to(device)

loss_fn = torch.nn.CrossEntropyLoss().to(device)
acc_fn = Accuracy(task='multiclass', num_classes=10).to(device)
optimizer = torch.optim.SGD(params = model_VGG.parameters(), lr = LEARNING_RATE)

results = engine.train(
    epochs=NUM_EPOCHS,
    model=model_VGG,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_function=loss_fn,
    acc_function=acc_fn,
    optimizer=optimizer,
    device=device
)

model_eval = utils.evaluate_model(model_VGG,
                                  test_dataloader,
                                  loss_fn,
                                  acc_fn,
                                  device)
print(model_eval)

utils.save_model(model_VGG, 
                 'models/',
                 'mnist_VGG.pth')