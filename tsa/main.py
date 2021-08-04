import os
import torch
import argparse
import pandas as pd
import torch.nn as nn
from .config import config
from .dataset import TimeSeriesDataset
from .model import AutoEncForecast
from .train import train
from .eval import evaluate


def parse_args():
    """
    Parse command line arguments.

    Args:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=config["batch_size"], type=int, help="batch size")
    parser.add_argument("--output-size", default=config["output_size"], type=int,
                        help="size of the ouput: default value to 1 for forecasting")
    parser.add_argument("--label-col", default=config["label_col"], type=str, help="name of the target column")
    parser.add_argument("--input-att", default=config["input_att"], type=lambda x: (str(x).lower() == "true"),
                        help="whether or not activate the input attention mechanism")
    parser.add_argument("--temporal-att", default=config["temporal_att"], type=lambda x: (str(x).lower() == "true"),
                        help="whether or not activate the temporal attention mechanism")
    parser.add_argument("--seq-len", default=config["seq_len"], type=int, help="window length to use for forecasting")
    parser.add_argument("--hidden-size-encoder", default=config["hidden_size_encoder"], type=int,
                        help="size of the encoder's hidden states")
    parser.add_argument("--hidden-size-decoder", default=config["hidden_size_decoder"], type=int,
                        help="size of the decoder's hidden states")
    parser.add_argument("--reg-factor1", default=config["reg_factor1"], type=float,
                        help="contribution factor of the L1 regularization if using a sparse autoencoder")
    parser.add_argument("--reg-factor2", default=config["reg_factor2"], type=float,
                        help="contribution factor of the L2 regularization if using a sparse autoencoder")
    parser.add_argument("--reg1", default=config["reg1"], type=lambda x: (str(x).lower() == "true"),
                        help="activate/deactivate L1 regularization")
    parser.add_argument("--reg2", default=config["reg2"], type=lambda x: (str(x).lower() == "true"),
                        help="activate/deactivate L2 regularization")
    parser.add_argument("--denoising", default=config["denoising"], type=lambda x: (str(x).lower() == "true"),
                        help="whether or not to use a denoising autoencoder")
    parser.add_argument("--do-train", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not to train the model")
    parser.add_argument("--do-eval", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not evaluating the mode")
    parser.add_argument("--data-path", default='nflx.csv', help="path to data file")
    parser.add_argument("--output-dir", default=config["output_dir"], help="name of folder to output files")
    parser.add_argument("--ckpt", default=None, help="checkpoint path for evaluation")
    return parser.parse_args()


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model state from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]
    epoch = checkpoint["epoch"]
    return model, optimizer, loss, epoch


def run(args):
    """
    Run the model.

    Args:
    """
    config.update(args)
    index_col_name = config["index_col"] if "index_col" in config.keys() else None
    df = pd.read_csv(config["data_path"], index_col=index_col_name)

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    ts = TimeSeriesDataset(
        data=df,
        categorical_cols=config["categorical_cols"],
        target_col=config["label_col"],
        seq_length=config["seq_len"],
        prediction_window=config["prediction_window"]
    )
    train_iter, test_iter, nb_features = ts.get_loaders(batch_size=config["batch_size"])

    model = AutoEncForecast(config, input_size=nb_features).to(config["device"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    if config["do_eval"] and config["ckpt"]:
        model, _, loss, epoch = load_checkpoint(config["ckpt"], model, optimizer, config["device"])
        evaluate(test_iter, loss, model, config, ts)
    elif config["do_train"]:
        train(train_iter, test_iter, model, criterion, optimizer, config, ts)


if __name__ == "__main__":
    args = parse_args()
    run(vars(args))
