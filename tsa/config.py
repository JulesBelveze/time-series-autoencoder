import torch


class Config(dict):
    def __init__(self, **kwargs):
        """
        Initialize an instance of this class.

        Args:

        """
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set(self, key, value):
        """
        Sets the value to the value.

        Args:
            key: (str):
            value:
        """
        self[key] = value
        setattr(self, key, value)


config = Config(
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    categorical_cols=["Time"],  # name of columns containing categorical variables
    label_col=["T"],  # name of target column
    index_col="Date",

    output_size=1,  # for forecasting

    num_epochs=100,
    batch_size=16,
    lr=1e-5,
    reg1=True,
    reg2=False,
    reg_factor1=1e-4,
    reg_factor2=1e-4,
    seq_len=10,  # previous timestamps to use
    prediction_window=1,  # number of timestamps to forecast
    hidden_size_encoder=128,
    hidden_size_decoder=128,
    input_att=True,
    temporal_att=True,
    denoising=False,
    directions=1,

    max_grad_norm=0.1,
    gradient_accumulation_steps=1,
    logging_steps=100,
    lrs_step_size=5000,

    output_dir="output",
    save_steps=5000,
    eval_during_training=True
)
