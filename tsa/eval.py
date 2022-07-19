import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(test_iter, criterion, model, config, ts):
    """
    Evaluate the model on the given test set.

    Args:
        test_iter: (DataLoader): test dataset iterator
        criterion: loss function
        model: model to use
        config: config
    """
    predictions, targets, attentions = [], [], []
    eval_loss = 0.0

    model.eval()
    for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating"):
        with torch.no_grad():
            feature, y_hist, target = batch
            output, att = model(feature.to(device), y_hist.to(device), return_attention=True)

            loss = criterion(output.to(device), target.to(device)).item()
            if config.training.reg1:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if 'bias' not in name])
                loss += config.training.reg_factor1 * torch.norm(params, 1)
            if config.training.reg2:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if 'bias' not in name])
                loss += config.training.reg_factor2 * torch.norm(params, 2)
            eval_loss += loss

            predictions.append(output.squeeze(1).cpu())
            targets.append(target.squeeze(1).cpu())
            attentions.append(att.cpu())

    predictions, targets = torch.cat(predictions), torch.cat(targets)

    if config.general.do_eval:
        preds, targets = ts.invert_scale(predictions), ts.invert_scale(targets)

        plt.figure()
        plt.plot(preds, linewidth=.3)
        plt.plot(targets, linewidth=.3)
        plt.savefig("{}/preds.png".format(config.general.output_dir))

        torch.save(targets, os.path.join(config.general.output_dir, "targets.pt"))
        torch.save(predictions, os.path.join(config.general.output_dir, "predictions.pt"))
        torch.save(attentions, os.path.join(config.general.output_dir, "attentions.pt"))

    results = get_eval_report(eval_loss / len(test_iter), predictions, targets)
    file_eval = os.path.join(config.general.output_dir, "eval_results.txt")
    with open(file_eval, "w") as f:
        f.write("********* EVAL REPORT ********\n")
        for key, val in results.items():
            f.write("  %s = %s\n" % (key, str(val)))

    return results


def get_eval_report(eval_loss: float, predictions: torch.Tensor, targets: torch.Tensor):
    """
    Evaluates the accuracy.

    Args:
        eval_loss: (float): loss vlue
        predictions: (torch.Tensor): tensor of predictions
        targets: (torch.Tensor): tensor of targets
    """
    residuals = np.mean(predictions.numpy() - targets.numpy())
    MSE = F.mse_loss(targets.squeeze(), predictions.squeeze()).item()
    return {"MSE": MSE, "residuals": residuals, "loss": eval_loss}
