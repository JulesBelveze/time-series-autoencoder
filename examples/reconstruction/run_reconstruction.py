import hydra
import pandas as pd
import torch
import torch.nn as nn

from tsa.utils import load_checkpoint
from tsa import TimeSeriesDataset, AutoEncForecast, train, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_path="./", config_name="config")
def run(args):
    df = pd.read_csv("data/AirQualityUCI.csv", index_col=args.data.index_col)

    ts = TimeSeriesDataset(
        data=df,
        categorical_cols=args.data.categorical_cols,
        target_col=args.data.label_col,
        seq_length=args.data.seq_len,
        prediction_window=args.data.prediction_window
    )
    train_iter, test_iter, nb_features = ts.get_loaders(batch_size=args.data.batch_size)

    model = AutoEncForecast(args, input_size=nb_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)

    if args.general.do_eval and args.general.ckpt:
        model, _, loss, epoch = load_checkpoint(args.general.ckpt, model, optimizer, device)
        evaluate(test_iter, loss, model, args, ts)
    elif args.general.do_train:
        train(train_iter, test_iter, model, criterion, optimizer, args, ts)


if __name__ == "__main__":
    run()
