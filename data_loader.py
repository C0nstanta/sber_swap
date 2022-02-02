import os

def d_loader(load_weights=False, load_onnx=False):
    #**Clone github & download models**
    os.system('git clone https://github.com/sberbank-ai/sber-swap.git')
    os.chdir('/content/sber-swap')

    os.system('wget -P ./arcface_model https://github.com/sberbank-ai/sber-swap/releases/download/arcface/iresnet.py')

    if load_onnx:
        # load landmarks detector
        os.system('wget -P ./insightface_func/models/antelope https://github.com/sberbank-ai/sber-swap/releases/download/antelope/glintr100.onnx')
        os.system('wget -P ./insightface_func/models/antelope https://github.com/sberbank-ai/sber-swap/releases/download/antelope/scrfd_10g_bnkps.onnx')

    if load_weights:
        # load arcface
        os.system('wget -P ./arcface_model https://github.com/sberbank-ai/sber-swap/releases/download/arcface/backbone.pth')
    
        # load model itself
        os.system('wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_2blocks.pth')

        # load super res model
        os.system('wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/super-res/10_net_G.pth')


if __name__ == "__main__":
    # load all modells & sber code
    d_loader(load_weights=True, load_onnx=True)