import os
import torch
import torch.onnx 
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class PetModel(pl.LightningModule):
    
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        self.register_buffer("std", torch.tensor([127.5]).view(1, 1, 1, 1))
        self.register_buffer("mean", torch.tensor([1.0]).view(1, 1, 1, 1))

    def forward(self, image):
        # normalize image here
        image = image / self.std - self.mean
        mask = self.model(image)
        return mask.sigmoid()


if __name__ == '__main__':

    model = PetModel("UNet", "timm-mobilenetv3_small_100", in_channels=1, out_classes=1)
    
    state_dict = torch.load('../examples/qrcode-unet-mbv3-100.1.pth')
    model.load_state_dict(state_dict)

    with torch.no_grad():
        model.eval()

        dummy_input = torch.randn(1, 1, 320, 320, requires_grad=True)

        # Export the model   
        torch.onnx.export(model,         # model being run 
                        dummy_input,       # model input (or a tuple for multiple inputs) 
                        "qrcode-unet-mbv3-100.onnx",       # where to save the model  
                        export_params=True,  # store the trained parameter weights inside the model file 
                        opset_version=11,    # the ONNX version to export the model to 
                        do_constant_folding=True,  # whether to execute constant folding for optimization 
                        input_names = ['input'],   # the model's input names 
                        output_names = ['output'], # the model's output names 
                        dynamic_axes={'input' : {0 : 'N', 2: 'H', 3: 'W'},    # variable length axes 
                                    'output' : {0 : 'N', 2: 'H', 3: 'W'}}) 