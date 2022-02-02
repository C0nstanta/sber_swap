import cv2
import torch
import time
import os
from utils.inference.image_processing import (crop_face, 
                                            get_final_image, 
                                            show_images
                                            )
from utils.inference.video_processing import (read_video, 
                                            get_target, 
                                            get_final_video, 
                                            add_audio_from_another_video, 
                                            face_enhancement
                                            )
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions




def model_init(AEI_Net_path='weights/G_unet_2blocks.pth', netArc_weight_path='arcface_model/backbone.pth'):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

    # main model for generation
    G = AEI_Net(backbone='unet', num_blocks=2, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(AEI_Net_path, map_location=torch.device('cpu')))
    G = G.cuda()
    G = G.half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load(netArc_weight_path))
    netArc=netArc.cuda()
    netArc.eval()

    # model to get face landmarks
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

    # model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
    use_sr = True
    if use_sr:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        #opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()
    return netArc, G, app, model, handler



def image_loader(source_path=None, target_path=None, target_type='image', path_to_video=None, app=None, out_video_path=None):
    # choose source image as a photo -- preferable a selfie of a person
    target_type = target_type # "video", "image"
    source_path =  source_path
    target_path = target_path 
    path_to_video = 'examples/videos/nggyup.mp4' 

    source_full = cv2.imread(source_path)
    out_video_path = "examples/results/result.mp4"
    crop_size = 224 # don't change this

    # check, if we can detect face on the source image
    try:    
        source = crop_face(source_full, app, crop_size)[0]
        source = [source[:, :, ::-1]]
        print("Everything is ok!")
    except TypeError:
        print("Bad source images")

    # read video
    fps = None
    if target_type == 'image':
        target_full = cv2.imread(target_path)
        full_frames = [target_full]
    else:
        full_frames, fps = read_video(path_to_video)
    target = get_target(full_frames, app, crop_size)
    return full_frames, source, target, crop_size, fps


def inference_start(model=None, full_frames=None, fps=None, handler=None, 
    source=None, target=None, netArc=None, G=None, app=None, b_size=40, 
    crop_size=224, target_type='image', out_video_path="examples/results/result.mp4",
    show_image=True, target_path='source_image_path'):
    batch_size = b_size
    START_TIME = time.time()
    final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(full_frames,
                                                        source,
                                                        target,
                                                        netArc,
                                                        G,
                                                        app,
                                                        set_target = False,
                                                        crop_size=crop_size,
                                                        BS=batch_size)
    use_sr = True
    if use_sr:
        final_frames_list = face_enhancement(final_frames_list, model)

    if target_type == 'video':
        get_final_video(final_frames_list,
                        crop_frames_list,
                        full_frames,
                        tfm_array_list,
                        out_video_path,
                        fps, 
                        handler)
        
        add_audio_from_another_video(path_to_video, out_video_path, "audio")

        print(f'Full pipeline took {time.time() - START_TIME}')
        print(f"Video saved with path {out_video_path}")
    else:
        result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler)
        cv2.imwrite('examples/results/result.png', result)    
    return result
 
if __name__ == "__main__":
    source_file_path='/content/drive/MyDrive/Memes/CTO.png'
    target_file_path = '/content/drive/MyDrive/Memes/template_clusters copy/2mode_girl/1521610.png'
    path_to_video = 'examples/videos/nggyup.mp4' 
    out_video_path = "examples/results/result.mp4"

    
    netArc, G, app, model, handler = model_init()
    

    # Our transform circle (last 2 func)
    import time
    start_time = time.time()
    
    full_frames, source, target, crop_size, fps = image_loader(source_path=source_file_path, 
                                target_path=target_file_path, target_type='image', 
                                path_to_video=path_to_video, app=app, 
                                out_video_path=out_video_path)
    
    result = inference_start(model=model, full_frames=full_frames, fps=fps, handler=handler, 
                source=source, target=target, netArc=netArc, G=G, app=app, b_size=40, 
                crop_size=224, target_type='image', out_video_path=out_video_path, 
                target_path=source_file_path, show_image=True)
    
    total_time = time.time() - start_time
    print("Transform time: ", total_time)