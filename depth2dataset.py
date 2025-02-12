import argparse
import os
import numpy as np
from tqdm import tqdm
import math

def YUV2RGB(Y,U,V, isYUV420 = True):
    """Y: (frame_num, height, width), U/V: (frame_num, height//2, width//2)"""
    FRAME_NUM, IMG_HEIGHT, IMG_WIDTH = Y.shape[-3], Y.shape[-2], Y.shape[-1]
    bgr_data = np.zeros((FRAME_NUM, 3, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    if (isYUV420):
        V = np.repeat(V, 2, 1)
        V = np.repeat(V, 2, 2)
        U = np.repeat(U, 2, 1)
        U = np.repeat(U, 2, 2)

    c = (Y-np.array([16])) * 298
    d = U - np.array([128])
    e = V - np.array([128])

    r = (c + 409 * e + 128) // 256
    g = (c - 100 * d - 208 * e + 128) // 256
    b = (c + 516 * d + 128) // 256

    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)

    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)

    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    bgr_data[:, 2, :, :] = r
    bgr_data[:, 1, :, :] = g
    bgr_data[:, 0, :, :] = b
    # return (n,3,h,w)
    return bgr_data


def get_frame_res(seq_content):
    """input (N,3,H,W) , return (N-1,3,H,W)"""
    FRAME_NUM = seq_content.shape[0]
    frame_res = np.zeros((FRAME_NUM-1, 3, seq_content.shape[-2], seq_content.shape[-1]))
    for frame_idx in range(FRAME_NUM - 1):
        frame_res[frame_idx] = seq_content[frame_idx + 1] - seq_content[frame_idx]
    return frame_res

def load_ifo_from_cfg(cfg_path):
    fp = open(cfg_path)
    input_path = None
    bit_depth = None
    width = None
    height = None
    frame_num = None
    for line in fp:
        if "InputFile" in line:
            line = line.rstrip('\n').replace(' ', '').split('#')[0]
            loc = line.find(':')
            input_path = line[loc+1:]
        elif "InputBitDepth" in line:
            bit_depth = int(line.rstrip('\n').replace(' ', '').split('#')[0].split(':')[-1])
        elif "SourceWidth" in line:
            width = int(line.rstrip('\n').replace(' ', '').split('#')[0].split(':')[-1])
        elif "SourceHeight" in line:
            height = int(line.rstrip('\n').replace(' ', '').split('#')[0].split(':')[-1])
        elif "FramesToBeEncoded" in line:
            frame_num = int(line.rstrip('\n').replace(' ', '').split('#')[0].split(':')[-1])
    if (input_path is None) or (bit_depth is None) or (width is None) or (height is None):
        print("Format of CFG error !!!!!!!!")
        return
    return input_path, bit_depth, width, height,frame_num


def import_yuv420(file_path, width, height, frm_num, sub_sample_ratio=1, is10bit=False, without_precision_loss=False):
    fp = open(file_path, 'rb')
    pix_num = width * height
    sub_frm_num = (frm_num + sub_sample_ratio - 1) // sub_sample_ratio  # actual frame number after downsampling
    if is10bit:
        data_type = np.uint16
    else:
        data_type = np.uint8
    y_temp = np.zeros(pix_num*sub_frm_num, dtype=data_type)
    u_temp = np.zeros(pix_num*sub_frm_num // 4, dtype=data_type)
    v_temp = np.zeros(pix_num*sub_frm_num // 4, dtype=data_type)
    for i in range(0, frm_num, sub_sample_ratio):
        if is10bit:
            fp.seek(i * pix_num * 3, 0)
        else:
            fp.seek(i * pix_num * 3 // 2, 0)
        subi = i // sub_sample_ratio
        y_temp[subi*pix_num : (subi+1)*pix_num] = np.fromfile(fp, dtype=data_type, count=pix_num, sep='')
        u_temp[subi*pix_num//4 : (subi+1)*pix_num//4] = np.fromfile(fp, dtype=data_type, count=pix_num//4, sep='')
        v_temp[subi*pix_num//4 : (subi+1)*pix_num//4] = np.fromfile(fp, dtype=data_type, count=pix_num//4, sep='')
    fp.close()
    y = y_temp.reshape((sub_frm_num, height, width))
    u = u_temp.reshape((sub_frm_num, height//2, width//2))
    v = v_temp.reshape((sub_frm_num, height//2, width//2))
    # if is10bit and ~without_precision_loss:
    #     y = np.clip((y + 2) / 4, 0, 255).astype(np.uint8)
    #     u = np.clip((u + 2) / 4, 0, 255).astype(np.uint8)
    #     v = np.clip((v + 2) / 4, 0, 255).astype(np.uint8)
    return y, u, v  # return frm_num * H * W

def es_rdo(mark_map, depth = 0, mt_depth = 0, pos_x = 0, pos_y = 0):
    """mark_map: (64,64) ~ (4,4)"""
    if mark_map[0,0,depth] == 2000 or depth == 7 or mt_depth >=3 :
        return
    global single_direction_map, single_depth_map
    height, width = mark_map.shape[0], mark_map.shape[1]
    partitio_type = mark_map[0,0,depth] 
    depth += 1
    if partitio_type == 1:
        # QT
        es_rdo(mark_map[:height//2, :width//2], depth, mt_depth=0, pos_x=pos_x, pos_y=pos_y)  # 1
        es_rdo(mark_map[:height//2, width//2:], depth, mt_depth=0, pos_x=pos_x, pos_y=pos_y + width//2)  # 2
        es_rdo(mark_map[height//2:, :width//2], depth, mt_depth=0, pos_x=pos_x + height//2, pos_y=pos_y)  # 3
        es_rdo(mark_map[height//2:, width//2:], depth, mt_depth=0, pos_x=pos_x + height//2, pos_y=pos_y + width//2)  # 4
    elif partitio_type == 2:
        # BTH
        single_direction_map[mt_depth, pos_x:pos_x+height, pos_y:pos_y+width] = 1
        single_depth_map[mt_depth, pos_x:pos_x+height, pos_y:pos_y+width] = 1
        mt_depth += 1
        es_rdo(mark_map[:height//2, :], depth, mt_depth, pos_x=pos_x, pos_y=pos_y)
        es_rdo(mark_map[height//2:, :], depth, mt_depth, pos_x=pos_x + height//2, pos_y=pos_y)
    elif partitio_type == 3:
        # BTV
        single_direction_map[mt_depth, pos_x:pos_x+height, pos_y:pos_y+width] = -1
        single_depth_map[mt_depth, pos_x:pos_x+height, pos_y:pos_y+width] = 1
        mt_depth += 1
        es_rdo(mark_map[:, :width//2], depth, mt_depth, pos_x=pos_x, pos_y=pos_y)
        es_rdo(mark_map[:, width//2:], depth, mt_depth, pos_x=pos_x, pos_y=pos_y + width//2)
    elif partitio_type == 4:
        # TTH
        single_direction_map[mt_depth, pos_x:pos_x+height, pos_y:pos_y+width] = 1
        single_depth_map[mt_depth, pos_x:pos_x+height, pos_y:pos_y+width] = 2
        single_depth_map[mt_depth, pos_x+height//4:pos_x+height//4*3, pos_y:pos_y+width] = 1 
        mt_depth += 1
        es_rdo(mark_map[:height//4, :], depth, mt_depth, pos_x=pos_x, pos_y=pos_y)
        es_rdo(mark_map[height//4:height//4*3, :], depth, mt_depth, pos_x=pos_x + height//4, pos_y=pos_y)
        es_rdo(mark_map[height//4*3:, :], depth, mt_depth, pos_x=pos_x + height//4*3, pos_y=pos_y)
    elif partitio_type == 5:
        # TTV
        single_direction_map[mt_depth, pos_x:pos_x+height, pos_y:pos_y+width] = -1
        single_depth_map[mt_depth, pos_x:pos_x+height, pos_y:pos_y+width] = 2
        single_depth_map[mt_depth, pos_x:pos_x+height, pos_y+width//4:pos_y+width//4*3] = 1
        mt_depth += 1
        es_rdo(mark_map[:, :width//4], depth, mt_depth, pos_x=pos_x, pos_y=pos_y)
        es_rdo(mark_map[:, width//4:width//4*3], depth, mt_depth, pos_x=pos_x, pos_y=pos_y + width//4)
        es_rdo(mark_map[:, width//4*3:], depth,mt_depth, pos_x=pos_x, pos_y=pos_y + width//4*3)



def get_depth_label(depth_file_path, frame_width, frame_height, frame_num, block_size=128, inter_mode=True):
    """return Depth Map(include QT,BT,MT), partition map label"""
    global single_direction_map, single_depth_map
    block_ratio = block_size // 64
    frame_height_cropped, frame_width_cropped = frame_height // block_size * block_size, frame_width // block_size * block_size
    MAX_MT_DEPTH = 4

    qt_depth_map = np.zeros((frame_num, 1, frame_width_cropped, frame_height_cropped), dtype=int)
    mtt_depth_map = np.zeros((frame_num, MAX_MT_DEPTH, frame_width_cropped, frame_height_cropped), dtype=int)
    mtt_direction_map = np.zeros((frame_num, MAX_MT_DEPTH, frame_width_cropped, frame_height_cropped), dtype=int)
    
    size_map = np.ones((frame_num, 2, frame_width_cropped, frame_height_cropped), dtype=int) * 128 
    f_id_list = []
    for f_id in range(frame_num):
        if inter_mode and f_id % 32 == 0:
            continue
        print('f_id: ', f_id)
        f_flag = False
        ctu_x, ctu_y = 0, 0
        with open(depth_file_path,'r') as f:
            for line in f.readlines():
                if f_flag and 'POC' not in line:
                    block_params_list = line.strip("\n").split(",")
                    block_params_list = [int(ele) for ele in block_params_list]
                    pos_x, pos_y, block_height, block_width, partition_mode = block_params_list[1], block_params_list[2], block_params_list[3], block_params_list[4], block_params_list[-1]

                    if (pos_x >= frame_width_cropped or pos_y >= frame_height_cropped):
                        # skip partial CTU
                        continue
                    elif pos_x < ctu_x + 128 and pos_y < ctu_y + 128:
                        if partition_mode == 1:
                            # quad split
                            qt_depth_map[f_id, 0, pos_x:pos_x + block_width, pos_y:pos_y + block_height] += 1
                        else:
                            mt_depth = 0
                            for i in range(MAX_MT_DEPTH):
                                if mtt_depth_map[f_id, i, pos_x, pos_y] == 0:
                                    mt_depth = i
                                    break
                            if mtt_depth_map[f_id, i, pos_x, pos_y] != 0:
                                raise Exception("wrong.")
                            if partition_mode == 2:
                                # btv
                                mtt_depth_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y:pos_y + block_height] = 1
                                mtt_direction_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y:pos_y + block_height] = -1
                            elif partition_mode == 3:
                                # bth
                                mtt_depth_map[f_id, mt_depth, pos_x:pos_x + block_width // 4, pos_y:pos_y + block_height] = 1
                                mtt_depth_map[f_id, mt_depth, pos_x + block_width // 4:pos_x + block_width // 4 * 3, pos_y:pos_y + block_height] = 2
                                mtt_depth_map[f_id, mt_depth, pos_x + block_width // 4 * 3 :pos_x + block_width, pos_y:pos_y + block_height] = 1
                                mtt_direction_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y:pos_y + block_height] = 1
                            elif partition_mode == 4:
                                # tth
                                mtt_depth_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y:pos_y + block_height // 4] = 1
                                mtt_depth_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y + block_height // 4: pos_y + block_height // 4 * 3] = 2
                                mtt_depth_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y + block_height // 4 * 3 :pos_y + block_height] = 1
                                mtt_direction_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y:pos_y + block_height] = -1
                            elif partition_mode == 5:
                                # ttv
                                mtt_depth_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y:pos_y + block_height] = 2
                                mtt_direction_map[f_id, mt_depth, pos_x:pos_x + block_width, pos_y:pos_y + block_height] = 1
                        # size 
                        size_map[f_id, 0, pos_x:pos_x + block_width, pos_y:pos_y + block_height] = block_width
                        size_map[f_id, 1, pos_x:pos_x + block_width, pos_y:pos_y + block_height] = block_height       
                    else:
                        # new CTU, refresh params.
                        ctu_x, ctu_y = pos_x, pos_y
                if 'POC' in line:
                    if line.strip('\n') == 'POC: %d'%f_id:
                        f_flag = True
                        f_id_list.append(f_id)
                    else:
                        f_flag = False
    qt_depth_map = np.transpose(qt_depth_map, axes=(0, 1, 3, 2))
    mtt_depth_map = np.transpose(mtt_depth_map, axes=(0, 1, 3, 2))
    mtt_direction_map = np.transpose(mtt_direction_map, axes=(0, 1, 3, 2))

    size_percent = np.zeros((6, 6))
    for f_id in tqdm(f_id_list):
        if inter_mode and f_id % 32 == 0:
            continue
        for i in range(size_map.shape[-1]):
            for j in range(size_map.shape[-2]):
                block_width, block_height = size_map[f_id, 0, j, i], size_map[f_id, 1, j, i]
                size_percent[int(math.log2(block_width)) - 2, int(math.log2(block_height)) - 2] += 1
    size_percent = size_percent / np.sum(size_percent)
    return qt_depth_map, mtt_depth_map[:,:5], mtt_direction_map[:,:5], size_percent

            
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_seq_info(seq_name):
    seq_dir = os.path.join('Z:\\', 'data', 'origCfp', 'BVI_DVC', 'Video_10bit') if seq_name[0] == 'A' else os.path.join(
        'Z:\\', 'data', 'origCfp', 'BVI_DVC', 'Video_8bit')
    seq_path = os.path.join(seq_dir, seq_name + '.yuv')
    if args.train:
        frame_width, frame_height = int(seq_name.split('_')[1].split('x')[0]), int(seq_name.split('_')[1].split('x')[1])
        bit_depth = 10 if seq_name[0] == 'A' else 8
        frame_num = 64  
        frame_rate = int(seq_name.split('_')[2].split('fps')[0])
    is10bit = False
    if bit_depth == 10:
        is10bit = True
    return frame_height, frame_width, frame_num, frame_rate, is10bit, seq_path


def make_depth_dataset_func(depth_file_path, frame_width, frame_height, base_name):
    qt_depth_map, mtt_depth_map, mtt_direction_map = get_depth_label(depth_file_path, frame_width, frame_height, frame_num=1, block_size=128)
    qt_label = qt_depth_map[:,0,::16,::16].astype(np.int8)
    mtt_label = np.stack([mtt_depth_map, mtt_direction_map], axis=2)[:,:,:,::4,::4].astype(np.int8)    
    with open(os.path.join(args.train_dataset_dir, 'qt_label', base_name + '.npy'), 'wb') as f:
        np.save(f, qt_label)
    with open(os.path.join(args.train_dataset_dir, 'mtt_label', base_name + '.npy'), 'wb') as f:
        np.save(f, mtt_label)
        
def write_seq_name_from_dir(Train_sequence_dir):
    seq_list = os.listdir(Train_sequence_dir)
    with open('sequences_name.txt','w') as f:
        for seq_name in seq_list:
           f.write(seq_name.strip('.yuv') + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qp', type=int, default=None, help='option: 22 27 32 37')
    parser.add_argument('--sequences_dir', type=str, help='Path to the training sequences')
    parser.add_argument('--cfg_dir', type=str, help='Path to the configuration file of the training sequences')
    parser.add_argument('--depth_file_path', type=str, help='Path to the extracted depth file')
    parser.add_argument('--thread_num', type=int, default=0, help='frame2block multi-threading' )
    args = parser.parse_args()

    write_seq_name_from_dir(args.sequences_dir)
    seq_list = list()
    with open("sequences_name.txt") as f:
        for line in f.readlines():
            seq_list.append(line.strip('\n'))

    qp_list = [22,27,32,37] if args.qp is None else [args.qp]
    size_percent_list = []
    
    for qp in qp_list:
        for seq_id, seq_name in tqdm(enumerate(seq_list)):
            cfg_path = os.path.join(args.cfg_dir,  seq_name + '.cfg')  
            print("SEQ:", seq_name)
            seq_path, bit_depth, frame_width, frame_height, frame_num = load_ifo_from_cfg(cfg_path)
            frame_num = 32
            if not os.path.exists(os.path.join(args.train_dataset_dir, 'Q' + str(qp), 'qt_label')):
                os.makedirs(os.path.join(args.train_dataset_dir,  'Q' + str(qp), 'qt_label'))
            if not os.path.exists(os.path.join(args.train_dataset_dir,  'Q' + str(qp), 'mtt_label')):
                os.makedirs(os.path.join(args.train_dataset_dir, 'Q' + str(qp), 'mtt_label'))
            depth_file_path = os.path.join(args.depth_file_path, seq_name + '_Depth.txt')
            qt_depth_map, mtt_depth_map, mtt_direction_map, size_percent = get_depth_label(depth_file_path, frame_width, frame_height, frame_num=frame_num, block_size=128)
            size_percent_list.append(size_percent)
            qt_label = qt_depth_map[:,0,::16,::16].astype(np.int8)
            mtt_label = np.stack([mtt_depth_map, mtt_direction_map], axis=2)[:,:,:,::4,::4].astype(np.int8)
            del qt_depth_map, mtt_depth_map, mtt_direction_map
