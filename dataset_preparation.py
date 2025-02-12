import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def write_seq_name_from_dir(Train_sequence_dir):
    seq_list = os.listdir(Train_sequence_dir)
    with open('sequences_name.txt','w') as f:
        for seq_name in seq_list:
           f.write(seq_name.strip('.yuv') + '\n')

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



def make_dataset(qp, seq_name, seq_dir, exe_name, bit_depth, enc_mode):
    """only for depth label
    enc_mode: LDP LDB RA"""
    cur_path = os.getcwd()
    exe_dir = os.path.join(cur_path, "codec")
    log_path = os.path.join(cur_path, "log", "QP" + str(qp))
    out_path = os.path.join(cur_path, 'output', enc_mode, 'QP' + str(qp), "train_dataset")
    depth_path = os.path.join(cur_path, "DepthSaving")
    seq_path = os.path.join(seq_dir, seq_name + '.yuv')
    frame_num = args.test_frm_num
    per_cfg_path = os.path.join(args.cfg_dir,  seq_name + '.cfg')  
    is10bit = False
    if bit_depth == 10:
        is10bit = True
    bin_name = seq_name + ".bin"
    enc_log_name = 'enc_' + seq_name + '.log'
    enc_cfg_name = 'encoder_randomaccess_vtm.cfg'
    if os.path.exists(os.path.join(depth_path, seq_name + '_Depth.txt')):
        os.remove(os.path.join(depth_path, seq_name + '_Depth.txt'))
    if os.path.exists(os.path.join(log_path, enc_log_name)):
        os.remove(os.path.join(log_path, enc_log_name))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if args.train:
        encoder_order = os.path.join(exe_dir, exe_name) + ' -c ' + os.path.join(args.cfg_dir, enc_cfg_name)\
            + ' -c ' + per_cfg_path + " -f "+ str(frame_num)  + ' -i ' + seq_path + " -q " +str(qp)  \
                         + ' -b ' + os.path.join(out_path, seq_name + '.bin') \
                        +  " >> " + os.path.join(log_path, enc_log_name)
    else:
        encoder_order = os.path.join(exe_dir, exe_name) + ' -c ' + os.path.join(args.cfg_dir, enc_cfg_name)\
             + ' -c ' + per_cfg_path + " -f "+ str(frame_num)  + ' -i ' + seq_path + " -q " +str(qp)  \
                         + ' -b ' + os.path.join(out_path, seq_name + '.bin') \
                        +  " >> " + os.path.join(log_path, enc_log_name)
    print(encoder_order)
    os.system(encoder_order)


if __name__ == '__main__':
    # Encode each sequence and generate partition depth files, e.g., DepthSaving/BasketballDrillText_832x480_50_Depth.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('--qp', type=int, default=None, help='option: 22 27 32 37')
    parser.add_argument('--sequences_dir', type=str, help='Path to the training sequences')
    parser.add_argument('--cfg_dir', type=str, help='Path to the configuration file of the training sequences')
    parser.add_argument('--test_frm_num', type=int, default=32)
    parser.add_argument('--bit_depth', type=int, default=8)

    args = parser.parse_args()
    qp_list = [32,27,22,37] if args.qp is None else [args.qp]
    exe_name = 'print_encoder.exe'

    write_seq_name_from_dir(args.sequences_dir)
    seq_list = list()
    with open("sequences_name.txt") as f:
        for line in f.readlines():
            seq_list.append(line.strip('\n'))

        po = ThreadPoolExecutor(max_workers=35)
        future_list = []
        for qp in qp_list:
            for seq_name in seq_list:
                # make_dataset(qp, seq_name, args.sequences_dir, exe_name, args.bit_depth, args.enc_mode)
                future_list.append(po.submit(make_dataset, seq_name, args.sequences_dir, exe_name, args.bit_depth, args.enc_mode))
        while True:
            is_done = True
            for future in future_list:
                is_done &= future.done()
            if is_done:
                break
            time.sleep(5)
        po.shutdown()