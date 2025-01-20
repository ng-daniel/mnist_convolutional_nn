import torch

def evaluate_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_function: torch.nn,
                   acc_function,
                   device: str):
  """
  Passes an entire dataloader through the model.

  Args:
    model: The model to evaluate.
    dataloader: The data to test the model on.
    loss_function: Function to calculate loss.
    acc_function: Function to calculate accuracy.
    device: 'cuda' or 'cpu'

  Returns: 
    A python dictionary containing the model's name, mean loss, and
    mean accuracy across all batches. 
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

  return {
      'model_name' : type(model).__name__,
      'loss' : loss,
      'accuracy' : acc
  }