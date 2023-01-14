from models import WavTrans_main


def create_model(opts):
    if opts.model_type == 'WavTrans':
        model = WavTrans_main.RecurrentModel(opts)

    else:
        raise NotImplementedError

    return model
