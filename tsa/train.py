import os
from json import dump

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from .eval import evaluate


def train(train_iter, test_iter, model, criterion, optimizer, config, ts):
    """
    Training function.

    Args:
        train_iter: (DataLoader): train data iterator
        test_iter: (DataLoader): test data iterator
        model: model
        criterion: loss to use
        optimizer: optimizer to use
        config:
    """
    tb_writer_train = SummaryWriter(logdir=config['output_dir'], filename_suffix='train')
    tb_writer_test = SummaryWriter(logdir=config['output_dir'], filename_suffix='test')

    if not os.path.exists(config['output_dir']):
        os.makedirs(config["output_dir"])

    with open(os.path.join(config['output_dir'], "config.json"), 'w+') as f:
        # removing non serializable types
        c = {key: value for key, value in config.items() if key not in ["device"]}
        dump(c, f)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lrs_step_size'], gamma=0.5)

    global_step, logging_loss = 0, 0.0
    train_loss = 0.0
    for epoch in tqdm(range(config['num_epochs']), unit='epoch'):
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            model.train()
            optimizer.zero_grad()

            feature, y_hist, target = batch
            output = model(feature.to(config["device"]), y_hist.to(config["device"]))
            loss = criterion(output.to(config["device"]), target.to(config["device"]))

            if config['reg1']:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if 'bias' not in name])
                loss += config['reg_factor1'] * torch.norm(params, 1)
            if config['reg2']:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if 'bias' not in name])
                loss += config['reg_factor2'] * torch.norm(params, 2)

            if config['gradient_accumulation_steps'] > 1:
                loss = loss / config['gradient_accumulation_steps']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            train_loss += loss.item()

            if (i + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                global_step += 1

                if global_step % config['logging_steps'] == 0:
                    if config['eval_during_training']:
                        results = evaluate(test_iter, criterion, model, config, ts)
                        for key, val in results.items():
                            tb_writer_test.add_scalar("eval_{}".format(key), val, global_step)

                    tb_writer_train.add_scalar("train_loss", (train_loss - logging_loss) / config['logging_steps'],
                                               global_step)
                    tb_writer_train.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    logging_loss = train_loss

            if global_step % config['save_steps'] == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': model.encoder.state_dict(),
                    'decoder_state_dict': model.decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion
                }, '{}/checkpoint-{}.ckpt'.format(config['output_dir'], global_step))
