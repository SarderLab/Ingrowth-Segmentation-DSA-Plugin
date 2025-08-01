import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import resnet_encoders

def create_model(model_details, n_classes, args):
    
    if model_details['architecture']=='Unet++':
        model =  smp.UnetPlusPlus(
            encoder_name = model_details['encoder'],
            encoder_weights = None, #model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            classes = n_classes,
            activation = model_details['active']
            )
        
        encoders = {}
        encoders.update(resnet_encoders)
        name = model_details['encoder']        
        depth = 5
        Encoder = encoders[name]["encoder"]
        params = encoders[name]["params"]
        params.update(depth=depth)
        model.encoder = Encoder(**params)

        # Load the state dictionary
        state_dict = torch.load(args.resnet)

        # Update the encoder's state
        model.encoder.load_state_dict(state_dict)

        model.encoder.set_in_channels(model_details['in_channels'])

        
    else:
        model = None
        print("Model architecture not supported")
    
    return model
