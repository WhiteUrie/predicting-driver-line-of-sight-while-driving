import os
import time
import torch

import warnings
import numpy as np
import json
from glob import glob
from utils.options import parser
from utils.bulid_models import build_model
from utils.build_datasets import build_dataset

from torch.utils.data import DataLoader
import imageio as io
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
# thay doan code nay de chay test duoc nhieu video 
# warnings.simplefilter("ignore")
# os.environ['NUMEXPR_MAX_THREADS'] = '64'

# args = parser.parse_args()
# torch.cuda.set_device(int(args.gpu))

# doan them vao 
warnings.simplefilter("ignore")
os.environ['NUMEXPR_MAX_THREADS'] = '64'

# Thư mục chứa frame của video cần test
parser.add_argument(
    '--input_dir',
    type=str,
    default='test_video/frames',
    help='Thư mục chứa các frame ảnh đầu vào'
)

# Thư mục lưu kết quả riêng cho từng video
parser.add_argument(
    '--output_dir',
    type=str,
    default='visualResults',
    help='Thư mục lưu kết quả saliency'
)

args = parser.parse_args()
torch.cuda.set_device(int(args.gpu))

def img2tensor(img_path, rgb=True):
    if rgb:
        # Giữ nguyên ảnh gốc để lưu kết quả ở độ phân giải ban đầu.
        img_ori = io.imread(img_path)

        # Bảo đảm ảnh đầu vào có đúng 3 kênh RGB.
        if img_ori.ndim == 2:
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2RGB)
        elif img_ori.shape[2] == 4:
            img_ori = img_ori[:, :, :3]

        # Chỉ resize bản sao đưa vào mô hình; ảnh gốc không bị thu nhỏ.
        img_model = cv2.resize(
            img_ori,
            args.img_shape,
            interpolation=cv2.INTER_CUBIC
        )

        img_ = img_model.astype('float32') / 255.0
        img_ = img_.transpose(2, 0, 1)
        img_ = np.ascontiguousarray(img_)
        img_tesor = torch.from_numpy(img_)

    else:
        # Giữ nguyên ảnh xám gốc để lưu kết quả.
        img_ori = io.imread(img_path, 0)

        img_model = cv2.resize(
            img_ori,
            args.img_shape,
            interpolation=cv2.INTER_CUBIC
        )

        img_ = img_model.astype('float32') / 255.0
        img_ = np.ascontiguousarray(img_)
        img_tesor = torch.from_numpy(img_)

    return img_ori, img_tesor


def _to_uint8_rgb(img):
    """Chuyển ảnh RGB về uint8 mà vẫn giữ đúng kích thước."""
    img = np.asarray(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    if np.issubdtype(img.dtype, np.floating):
        if img.size > 0 and np.nanmax(img) <= 1.0:
            img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def _write_image(saveFullName, img_bgr):
    """Lưu ảnh với chất lượng cao theo đúng phần mở rộng."""
    extension = os.path.splitext(saveFullName)[1].lower()

    if extension == '.png':
        cv2.imwrite(
            saveFullName,
            img_bgr,
            [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )
    elif extension in ('.jpg', '.jpeg'):
        cv2.imwrite(
            saveFullName,
            img_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )
    else:
        cv2.imwrite(saveFullName, img_bgr)


def savePlot(img_ori=None, img_sal=None, saveFullName=None):
    """Lưu ảnh gốc, heatmap hoặc ảnh chồng mà không làm giảm độ phân giải."""
    if img_ori is not None:
        img_ori = _to_uint8_rgb(img_ori)

    if img_sal is not None:
        img_sal = np.asarray(img_sal, dtype=np.float32)

    if img_ori is not None and img_sal is not None:
        # Bảo đảm heatmap có cùng kích thước với ảnh gốc.
        if img_sal.shape[:2] != img_ori.shape[:2]:
            img_sal = cv2.resize(
                img_sal,
                (img_ori.shape[1], img_ori.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )

        saliency_uint8 = cv2.normalize(
            img_sal,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            saliency_uint8,
            cv2.COLORMAP_JET
        )

        img_bgr = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)

        # Giữ ảnh nền rõ hơn; heatmap chiếm 30%.
        result = cv2.addWeighted(
            img_bgr,
            0.6,
            heatmap,
            0.4,
            0
        )

        _write_image(saveFullName, result)

    elif img_ori is not None:
        img_bgr = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)
        _write_image(saveFullName, img_bgr)

    elif img_sal is not None:
        saliency_uint8 = cv2.normalize(
            img_sal,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            saliency_uint8,
            cv2.COLORMAP_JET
        )

        _write_image(saveFullName, heatmap)


def onevisual(model, input_data, saveFolder, save_ori=True, save_pre=True):
    """
    type of input_data: str, tensor
    example:
            str : '0001/0001.jpg'
            list: [index, nump-->in_img, nump-->tar_img, nump-->out_img]
    """
    if isinstance(input_data, str):
        # Lưu đầu ra dạng PNG để tránh nén mất dữ liệu như JPG.
        file_stem = os.path.splitext(os.path.basename(input_data))[0]
        saveBaseName = file_stem + '.png'
        saveFolder = f'{saveFolder}/temp/'
        os.makedirs(saveFolder, exist_ok=True)

        img, img_tesor = img2tensor(input_data)
        img_tesor = img_tesor.cuda()
        img_tesor = img_tesor.unsqueeze(0)

        
        pre_mask, _ = model(img_tesor)

        pre_mask = pre_mask.detach().cpu().numpy()
        if len(pre_mask.shape) == 4:
            pre_mask = pre_mask.squeeze(0).squeeze(0)

        # Phóng heatmap từ kích thước mô hình về đúng kích thước ảnh gốc.
        # cv2.resize nhận thứ tự (width, height).
        pre_mask = cv2.resize(
            pre_mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        assert pre_mask.shape[:2] == img.shape[:2], (
            'Kích thước heatmap không khớp với ảnh gốc.'
        )

        # save ori
        if save_ori:
            saveFullName = f'{saveFolder}/ori_{saveBaseName}'
            savePlot(img_ori=img, saveFullName=saveFullName)
        # save pre
        if save_pre:
            saveFullName = f'{saveFolder}/sal_{saveBaseName}'
            savePlot(img_sal=pre_mask, saveFullName=saveFullName)
        # save ori+pre
        saveFullName = f'{saveFolder}/add_{saveBaseName}'
        savePlot(img_ori=img, img_sal=pre_mask, saveFullName=saveFullName)

    else:
        # save_ori = False
        # save_pre = False
        saveBaseNames = [json.loads(line) for line in open(args.root + '/test.json')]
        index, in_img, tar_img, out_img = input_data[0], input_data[1], input_data[2], input_data[3]
        saveBaseName = saveBaseNames[index]
        # print(saveBaseName)
        saveBaseNameList = saveBaseName.split('/')
        # print(saveBaseNameList)
        in_img = in_img.squeeze(0).transpose(1, 2, 0)

        # save ori
        if save_ori:
            saveFolder_ = f'{saveFolder}/{args.category}/ori/{saveBaseNameList[0]}'
            os.makedirs(saveFolder_, exist_ok=True)
            saveFullName = f'{saveFolder_}/{saveBaseNameList[1]}'
            savePlot(img_ori=in_img, saveFullName=saveFullName)
        # save pre
        if save_pre:
            saveFolder_ = f'{saveFolder}/{args.category}/pre/{saveBaseNameList[0]}'
            os.makedirs(saveFolder_, exist_ok=True)
            saveFullName = f'{saveFolder_}/{saveBaseNameList[1]}'
            savePlot(img_sal=out_img, saveFullName=saveFullName)
        # save ori+pre
        saveFolder_ = f'{saveFolder}/{args.category}/add/{saveBaseNameList[0]}'
        os.makedirs(saveFolder_, exist_ok=True)
        saveFullName = f'{saveFolder_}/{saveBaseNameList[1]}'
        savePlot(img_ori=in_img, img_sal=out_img, saveFullName=saveFullName)


def visualization(model, input_data, saveFolder='visualResults', save_ori=True, save_pre=True):
    """
    type of input_data: str, list, .txt, dataloader
    example:
            str : '0001/0001.jpg'
            list: ['0001/0001.jpg', '0001/0002.jpg', ......]
            dataloader: by our build_dataset.
    """
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise ValueError(f"Path: {input_data} does not exist.")

        onevisual(model, input_data=input_data, saveFolder=saveFolder, save_ori=True, save_pre=True)

    elif isinstance(input_data, list):
        for path in input_data:
            if not isinstance(path, str):
                raise TypeError(f"The element must be string, but {path} is {type(path)}")
            if not os.path.exists(path):
                raise ValueError(f"Path: {input_data} does not exist.")
        for path in input_data:
            onevisual(model, input_data=path, saveFolder=saveFolder, save_ori=True, save_pre=True)       

    elif isinstance(input_data, DataLoader):
        for i, (in_img, tar_img) in tqdm(enumerate(input_data), total=len(input_data), desc="Visualization Processing"):
            tar_img = tar_img.squeeze(0).squeeze(0)
            in_img = in_img.cuda()
            out_img, _ = model(in_img)
            out_img = out_img.squeeze(0).squeeze(0)

            in_img = in_img.detach().cpu().numpy()
            tar_img = tar_img.detach().cpu().numpy()
            out_img = out_img.detach().cpu().numpy()

            visual_np = [i, in_img, tar_img, out_img]
            onevisual(model, input_data=visual_np, saveFolder=saveFolder, save_ori=True, save_pre=True)
    else:
        raise TypeError("Parameter input_data must be a string path, a list of string paths, a txt file or a DataLoader.")


if __name__ == '__main__':
    # #  test------>next
    _, _, test_loader = build_dataset(args=args)
    ckpts = f'ckpts/{args.category}/{args.test_weight}/'
    file_name = ckpts + f'model_best_{args.network}.tar'

    checkpoint = torch.load(file_name, map_location="cpu")

    model = build_model(args=args).cuda()
    model.load_state_dict(checkpoint['state_dict'])  # single-gpu
    # print(model.device)
    # input_str = '/data/dataset/BDDA/images/0002/001.jpg'
    # input_list = ['/data/dataset/BDDA/images/0002/002.jpg', '/data/dataset/BDDA/images/0002/003.jpg']
    
    # cat video sua doan nay 
    #input_dataloader = test_loader
    #saveFolder = 'visualResults'
    #visualization(model, input_dataloader, saveFolder)
    
    #doan code them 
    # input_list = sorted(glob("test_video/frames/*.jpg"))
    # saveFolder = "visualResults"
    # visualization(model, input_list, saveFolder)
    
        # Chuyển model sang chế độ đánh giá
    model.eval()

    # Tìm tất cả ảnh JPG, JPEG và PNG trong input_dir
    input_list = []

    input_list.extend(
        glob(os.path.join(args.input_dir, "*.jpg"))
    )
    input_list.extend(
        glob(os.path.join(args.input_dir, "*.jpeg"))
    )
    input_list.extend(
        glob(os.path.join(args.input_dir, "*.png"))
    )

    input_list = sorted(input_list)

    # Kiểm tra thư mục có ảnh hay không
    if len(input_list) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy frame ảnh trong thư mục: {args.input_dir}"
        )

    saveFolder = args.output_dir
    os.makedirs(saveFolder, exist_ok=True)

    print(f"Tìm thấy {len(input_list)} frame.")
    print(f"Thư mục đầu vào: {args.input_dir}")
    print(f"Kết quả được lưu tại: {args.output_dir}/temp")

    visualization(
        model,
        input_list,
        saveFolder
    )
