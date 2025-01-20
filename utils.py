import torch
from pathlib import Path

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

def save_model(model: torch.nn.Module,
               path: str,
               model_name: str):
  """
  Function to save pytorch model as a python dictionary in the
  specified directory.

  Args:
    model: the pytorch model to save
    path: the path of the directory to save in
    model_name: the name and type of the file

  """
  directory_path = Path(path)
  directory_path.mkdir(parents=True, exist_ok=True)
  model_dir_path = directory_path / model_name
  assert model_name.endswith('.pth') or model_name.endswith('.pt')
  torch.save(model.state_dict(), model_dir_path)