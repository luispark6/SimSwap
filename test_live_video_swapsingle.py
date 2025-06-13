import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
import line_profiler
import time



def main():
        
    # Setup
    opt = TestOptions().parse()
    crop_size = opt.crop_size
    torch.nn.Module.dump_patches = True
    mode = 'ffhq' if crop_size == 512 else 'None'

    # Initialize face detection, model, etc.
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    model = create_model(opt)
    model.eval()
    spNorm = SpecificNorm()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)
    # Prepare ID image (pic_a) — stays constant
    with torch.no_grad():
        img_a_whole = cv2.imread(opt.pic_a_path)
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_id = transformer(img_a_pil).unsqueeze(0).cuda()
        img_id = F.interpolate(img_id, size=(112, 112))
        latent_id = F.normalize(model.netArc(img_id), p=2, dim=1)

    # Optional: load mask model
    if opt.use_mask:
        from parsing_model.model import BiSeNet
        net = BiSeNet(n_classes=19).cuda()
        net.load_state_dict(torch.load('./parsing_model/checkpoint/79999_iter.pth'))
        net.eval()
    else:
        net = None
    # print("HIIIIIIIIIII")
    # print(type(net))
    # return type(net)

    # Live video
    cap = cv2.VideoCapture(0)  # Webcam
    assert cap.isOpened(), "Camera couldn't open."
    fps=0
    fps_update_interval = 0.5
    frame_count = 0


    print("Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time
        print(fps)
        with torch.no_grad():
            try:
                img_b_align_crop_list, b_mat_list = app.get(frame, crop_size) #this is the face detector
                b_tensors = []
                results = []

                for b_crop in img_b_align_crop_list:
                    b_tensor = torch.from_numpy(cv2.cvtColor(b_crop, cv2.COLOR_BGR2RGB)) \
                                .permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
                    
                    
                    result = model(None, b_tensor, latent_id, None, True)[0] #this is the forward pass for the face swap
                    results.append(result)
                    b_tensors.append(b_tensor)

                output_img = reverse2wholeimage(
                    b_tensors, results, b_mat_list, crop_size, frame, logoclass,
                    None, opt.no_simswaplogo, pasring_model=net,
                    use_mask=opt.use_mask, norm=spNorm
                )

                cv2.imshow("SimSwap Live", output_img)
            except Exception as e:
                print(f"Frame skipped: {e}")
                cv2.imshow("SimSwap Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
main()