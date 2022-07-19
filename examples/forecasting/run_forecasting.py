import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate

from tsa import AutoEncForecast, train, evaluate
from tsa.utils import load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_path="./", config_name="config")
def run(cfg):
    ts = instantiate(cfg.data)
    train_iter, test_iter, nb_features = ts.get_loaders()

    model = AutoEncForecast(cfg.training, input_size=nb_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    if cfg.general.do_train:
        train(train_iter, test_iter, model, criterion, optimizer, cfg, ts)
    if cfg.general.do_eval and cfg.general.get("ckpt", False):
        model, _, loss, epoch = load_checkpoint(cfg.general.ckpt, model, optimizer, device)
        evaluate(test_iter, loss, model, cfg, ts)


if __name__ == "__main__":
    run()
