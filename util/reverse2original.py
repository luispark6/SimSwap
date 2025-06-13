import cv2
import numpy as np
# import  time
import torch
from torch.nn import functional as F
import torch.nn as nn
import line_profiler


def get_gaussian_kernel2d(kernel_size=21, sigma=3, channels=1):
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d /= gauss_1d.sum()

    # Outer product to get 2D kernel
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    gauss_2d = gauss_2d.expand(channels, 1, kernel_size, kernel_size)
    return gauss_2d


def gpu_gaussian_blur(img_tensor, kernel_size=21, sigma=3):
    channels = img_tensor.shape[1]
    kernel = get_gaussian_kernel2d(kernel_size, sigma, channels).to(img_tensor.device)
    padding = kernel_size // 2
    blurred = F.conv2d(img_tensor, kernel, padding=padding, groups=channels)
    return blurred


def torch_erode(mask_tensor, kernel_size=40):
    padding = kernel_size // 2
    eroded = -F.max_pool2d(-mask_tensor, kernel_size, stride=1, padding=padding)
    return eroded
import numpy as np

def encode_segmentation_rgb(segmentation, no_neck=True):
    face_part_ids = {1, 2, 3, 4, 5, 6, 10, 12, 13} if no_neck else {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14}
    mouth_id = 11

    # Vectorized masks
    face_map = np.isin(segmentation, list(face_part_ids)) * 255
    mouth_map = (segmentation == mouth_id) * 255

    return np.stack([face_map.astype(np.uint8), mouth_map.astype(np.uint8)], axis=2)


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def postprocess(swapped_face, target, target_mask,smooth_mask):
    # target_mask = cv2.resize(target_mask, (self.size,  self.size))

    mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
    face_mask_tensor = mask_tensor[0] + mask_tensor[1]
    
    soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
    soft_face_mask_tensor.squeeze_()

    soft_face_mask = soft_face_mask_tensor.cpu().numpy()
    soft_face_mask = soft_face_mask[:, :, np.newaxis]

    result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
    result = result[:,:,::-1]# .astype(np.uint8)
    return result



# @line_profiler.profile
def reverse2wholeimage(b_align_crop_tenor_list,swaped_imgs, mats, crop_size, oriimg, logoclass, save_path = '', \
                    no_simswaplogo = False,pasring_model =None,norm = None, use_mask = False):

    target_image_list = []
    img_mask_list = []
    if use_mask:
        smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
    else:
        pass

    # print(len(swaped_imgs))
    # print(mats)
    # print(len(b_align_crop_tenor_list))
    for swaped_img, mat ,source_img in zip(swaped_imgs, mats,b_align_crop_tenor_list):
        swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
        img_white = np.full((crop_size,crop_size), 255, dtype=float)

        # inverse the Affine transformation matrix
        mat_rev = np.zeros([2,3])
        div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
        mat_rev[0][0] = mat[1][1]/div1
        mat_rev[0][1] = -mat[0][1]/div1
        mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
        div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
        mat_rev[1][0] = mat[1][0]/div2
        mat_rev[1][1] = -mat[0][0]/div2
        mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

        orisize = (oriimg.shape[1], oriimg.shape[0])
        if use_mask:
            source_img_norm = norm(source_img)
            source_img_512  = F.interpolate(source_img_norm,size=(512,512))
            print(type(pasring_model))
            with torch.no_grad():
                out = pasring_model(source_img_512)[0]
            parsing = torch.argmax(out.squeeze(0), dim=0).byte().cpu().numpy()
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            tgt_mask = encode_segmentation_rgb(vis_parsing_anno)
            if tgt_mask.sum() >= 5000:
                # face_mask_tensor = tgt_mask[...,0] + tgt_mask[...,1]
                target_mask = cv2.resize(tgt_mask, (crop_size,  crop_size))
                # print(source_img)
                target_image_parsing = postprocess(swaped_img, source_img[0].cpu().detach().numpy().transpose((1, 2, 0)), target_mask,smooth_mask)
                

                target_image = cv2.warpAffine(target_image_parsing, mat_rev, orisize)
                # target_image_parsing = cv2.warpAffine(swaped_img, mat_rev, orisize)
            else:
                target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)[..., ::-1]
        else:
            target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
        # source_image   = cv2.warpAffine(source_img, mat_rev, orisize)

        img_white = cv2.warpAffine(img_white, mat_rev, orisize)


        img_white[img_white>20] =255

        img_mask = img_white

        # if use_mask:
        #     kernel = np.ones((40,40),np.uint8)
        #     img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        # else:
        # Input: mask as [1, 1, H, W] float tensor on GPU
        img_mask_tensor = torch.from_numpy(img_mask).unsqueeze(0).unsqueeze(0).float().cuda()
        img_mask_eroded = torch_erode(img_mask_tensor, kernel_size=41)
        img_mask = img_mask_eroded.squeeze().cpu().numpy()
        # blur_size = tuple(2*i+1 for i in kernel_size)
        img_mask_tensor = torch.from_numpy(img_mask).unsqueeze(0).unsqueeze(0).float().cuda()  # [1, 1, H, W]
        img_mask_blurred = gpu_gaussian_blur(img_mask_tensor, kernel_size=41, sigma=5.0)  # You can tune these

        img_mask = img_mask_blurred.squeeze().cpu().numpy()


        # kernel = np.ones((10,10),np.uint8)
        # img_mask = cv2.erode(img_mask,kernel,iterations = 1)



        img_mask /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])

        # pasing mask

        # target_image_parsing = postprocess(target_image, source_image, tgt_mask)

        arr = np.asarray(target_image, dtype=np.float32)
        if not use_mask:
            arr = arr[..., ::-1].copy()
        np.multiply(arr, 255, out=arr)
        target_image = arr 


        
        img_mask_list.append(img_mask)
        target_image_list.append(target_image)
        

    # target_image /= 255
    # target_image = 0
     # ---- GPU Parallel Blending with Sequential Dependency ----
    img_tensor = torch.tensor(np.array(oriimg, dtype=float), dtype=torch.float32).to('cuda')  # [H, W, C]

    for img_mask_np, target_image_np in zip(img_mask_list, target_image_list):
        img_mask = torch.tensor(img_mask_np, dtype=torch.float32).to('cuda')  # [H, W, 1]
        target_image = torch.tensor(target_image_np, dtype=torch.float32).to('cuda')  # [H, W, C]

        if img_mask.ndim == 2:
            img_mask = img_mask.unsqueeze(-1)

        img_tensor = img_mask * target_image + (1 - img_mask) * img_tensor

    final_img = img_tensor.clamp(0, 255).byte().cpu().numpy()

    # final_img = img.astype(np.uint8)

    if not no_simswaplogo:
        final_img = logoclass.apply_frames(final_img)
    return final_img
