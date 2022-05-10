import torch


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model state from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]
    epoch = checkpoint["epoch"]
    return model, optimizer, loss, epoch
