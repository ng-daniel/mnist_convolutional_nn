import torch
from tqdm.auto import tqdm
from utils import evaluate_model

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn,
               acc_function,
               optimizer: torch.optim,
               device: str):
  """
  Trains the model on data.

  Args:
    model: The model to train.
    dataloader: The data to train the model on.
    loss_function: Function to calculate loss.
    acc_function: Function to calculate accuracy.
    optimizer: Optimization function to improve/train model.
    device: 'cuda' or 'cpu'

  Returns: 
    A python tuple containing the average loss and average accuracy
    across all batches of the training dataloader.
  """
  model.train()
  loss = acc = 0
  for batch,(X,y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    # calculate batch logits and predictions
    logits = model(X)
    preds = logits.softmax(dim=1).argmax(dim=1)
    # calculate batch loss and accuracy
    batch_loss = loss_function(logits.squeeze(), y)
    batch_acc = acc_function(preds, y)

    loss += batch_loss
    acc += batch_acc

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

  loss = loss / len(dataloader)
  acc = acc / len(dataloader)

  return loss, acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn,
              acc_function,
              device: str):
  """
  Tests the model on data it hasn't been trained on.

  Args:
    model: The model to test.
    dataloader: The data to test the model on.
    loss_function: Function to calculate loss.
    acc_function: Function to calculate accuracy.
    device: 'cuda' or 'cpu'

  Returns: 
    A python tuple containing the average loss and average accuracy
    across all batches of the test dataloader.
  """
  model.eval()
  loss = acc = 0
  for batch,(X,y) in enumerate(dataloader):
    with torch.inference_mode():
      X, y = X.to(device), y.to(device)
      logits = model(X)
      preds = logits.softmax(dim=1).argmax(dim=1)

      batch_loss = loss_function(logits, y)
      batch_acc = acc_function(preds, y)

      loss += batch_loss
      acc += batch_acc

  loss = loss / len(dataloader)
  acc = acc / len(dataloader)

  return loss, acc


def train(epochs: int,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn,
          acc_function,
          optimizer: torch.optim,
          device: str):
  """Trains and evaluates the model.

  Runs the train and test step functions across a set number of epochs.
  
  Args:
    epochs: The number of times to run the train and test functions.
    model: The model to train.
    train_dataloader: The data to train the model on.
    test_dataloader: The data to test the model on.
    loss_function: Function to calculate loss.
    acc_function: Function to calculate accuracy.
    optimizer: Optimization function to improve/train model.
    device: 'cuda' or 'cpu'
  
  Returns:
    A dictionary containing the model's name and lists of loss and accuracy metrics
    for both training and testing steps across the number of epochs.
  
  """
  #initialize results dictionary
  results = {
      "model_name":type(model).__name__,
      "train_loss":[],
      "train_acc":[],
      "test_loss":[],
      "test_acc":[]
  }

  #add initial metrics to results
  initial_train = evaluate_model(model,
                                 train_dataloader,
                                 loss_function,
                                 acc_function,
                                 device)
  initial_test = evaluate_model(model,
                                test_dataloader,
                                loss_function,
                                acc_function,
                                device)
  results["train_loss"].append(initial_train['loss'].item())
  results["train_acc"].append(initial_train['accuracy'].item())
  results["test_loss"].append(initial_test['loss'].item())
  results["test_acc"].append(initial_test['accuracy'].item())

  #pass model through training and testing over several epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model,
                                      train_dataloader,
                                      loss_function,
                                      acc_function,
                                      optimizer,
                                      device)
    test_loss, test_acc = test_step(model,
                                    test_dataloader,
                                    loss_function,
                                    acc_function,
                                    device)
    print("--------")
    print(f"Epoch:{epoch}")
    print(f"train_loss:{train_loss}")
    print(f"train_acc:{train_acc}")
    print(f"test_loss:{test_loss}")
    print(f"test_acc:{test_acc}")

    results["train_loss"].append(train_loss.item())
    results["train_acc"].append(train_acc.item())
    results["test_loss"].append(test_loss.item())
    results["test_acc"].append(test_acc.item())
  
  return results