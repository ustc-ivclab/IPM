'''
Form partition map to partition structure
Input: partition map
Output: partition appearance + QT depth map + direction map in the .txt form
(the partition data form in this code version is not the best form for encoder implementation, \
    a little complicated but you can focus on the legalization process of output partition map)
Author: Aolin Feng
'''

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from random import choice
from einops import rearrange
#import os
import torch
# from matplotlib import pyplot as plt
# from .Metrics import eli_structual_error
# import time
import random
# VVC prior information
# qt depth 0->3
# bt depth 0->5 0->4 (chroma)
# direction 0 1 2

def check_square_unity(mat):  # input 4*4 tensor
    num0 = len(torch.where(mat == 0)[0])
    if num0 >= 0 and num0 <= 12:  # 0 in the minority
        mat = torch.where(mat == 0, torch.full_like(mat, 1).cuda(), mat)
        # process 4 sub-mats
        for i in [0, 2]:
            for j in [0, 2]:
                sum_sub_mat = torch.sum(mat[i:i + 2, j:j + 2])
                if sum_sub_mat <= 10 and sum_sub_mat >= 5: # 1 and 2 or 3 mixed
                    sub_num1 = len(torch.where(mat[i:i + 2, j:j + 2] == 1)[0])
                    if sub_num1 < 3:
                        mat[i:i + 2, j:j + 2] = torch.where(mat[i:i + 2, j:j + 2] == 1, (torch.ones((2, 2)) * 2).cuda(), mat[i:i + 2, j:j + 2])
                    else:
                        mat[i:i + 2, j:j + 2] = torch.ones((2, 2)).cuda()
    elif num0 > 12 and num0 < 16:
        mat = torch.zeros((4, 4)).cuda()
    return mat

def eli_structual_error(out_batch):
    N = out_batch.shape[0]
    pooled_batch = torch.clamp(torch.round(F.max_pool2d(out_batch, 2)), min=0, max=3)
    for num in range(N):
        pooled_batch[num][0] = check_square_unity(pooled_batch[num][0])
    post_batch = F.interpolate(pooled_batch, scale_factor=2)
    del pooled_batch
    return post_batch

def th_round(input_batch, thd):
    input_batch = np.where(input_batch >= thd, np.full_like(input_batch, 1), input_batch)
    input_batch = np.where(input_batch <= -thd, np.full_like(input_batch, -1), input_batch)
    input_batch = np.where((input_batch > -thd) & (input_batch < thd), np.full_like(input_batch, 0),
                                 input_batch)
    return input_batch

def th_round2(input_batch, thd_1, thd_2):
    # for qt map, 0 1 [2] 3, thd=0.2
    input_batch = np.where((input_batch >= (1 + thd_2)) * (input_batch  <= (3 - thd_2)), np.full_like(input_batch, 2), input_batch)
    input_batch = np.where((input_batch >= (1 - thd_1)) * (input_batch  <= (1 + thd_1)), np.full_like(input_batch, 1), input_batch)
    input_batch = np.round(input_batch)
    return input_batch


def th_round3(input_batch, thd):
    # for mtt depth map, region for 2
    input_batch = np.where((input_batch >= (1 + thd)), np.full_like(input_batch, 2), input_batch)
    input_batch = np.round(input_batch)
    return input_batch

def th_round4(input_batch, thd_1, thd_2):
    # for mtt depth map, region for 1
    input_batch = np.where((input_batch <= (1 + thd_2)) * (input_batch >= (1 - thd_1)), np.full_like(input_batch, 1), input_batch)
    input_batch = np.where((input_batch >= (1 + thd_2)) * (input_batch <= 2.5), np.full_like(input_batch, 2), input_batch)
    input_batch = np.where((input_batch <= (1 - thd_1)), np.full_like(input_batch, 0), input_batch)
    input_batch = np.round(input_batch)
    return input_batch

def delete_tree(root):
    if len(root.children) == 0:  # no child
        del root
    else:
        children = root.children
        for child in children:
            delete_tree(child)

# Split_Mode and Search are set for generate combination modes
# example: input [[1,2], [3,4,5]]; output [[1,3],[1,4],[1,5],[2,3],[2,4],[2,5]]

class Split_Node():
    def __init__(self, split_type):
        self.split_type = split_type
        self.children = []

class Search():
    def __init__(self, cus_candidate_mode_list):
        self.split_root = Split_Node(0)
        self.parent_list = []
        self.cus_mode_list = cus_candidate_mode_list
        self.partition_modes = []
        self.cus_modes = []

    def get_cus_mode_tree(self):
        self.parent_list = [self.split_root]
        while len(self.cus_mode_list) != 0:  # transverse every cu
            parent_temp = []
            for parent in self.parent_list:
                for split_type in self.cus_mode_list[0]:
                    child = Split_Node(split_type)
                    parent.children.append(child)
                    parent_temp.append(child)
            self.parent_list = parent_temp
            self.cus_mode_list.pop(0)

    def bfs(self, node):
        self.cus_modes.append(node.split_type)
        if len(node.children) == 0:
            temp = self.cus_modes[1:]
            self.partition_modes.append(temp)
            self.cus_modes.pop(-1)
        else:
            for child in node.children:
                self.bfs(child)
            self.cus_modes.pop(-1)

    def get_partition_modes(self):
        self.get_cus_mode_tree()
        self.bfs(self.split_root)
        return self.partition_modes

class Map_Node():
    def __init__(self, qt_map, bt_map, dire_map, cus, depth, bt_depth, out_mt_map, out_dire_map, parent=None, early_skip=True):
        self.qt_map = qt_map
        self.bt_map = bt_map
        self.dire_map = dire_map
        self.depth = depth
        self.cus = cus  # [x, y, h, w] list
        self.children = []
        self.parent = parent
        self.bt_depth = bt_depth
        self.out_mt_map = out_mt_map
        self.out_dire_map = out_dire_map
        if early_skip:
            self.early_skip_list = []

def find_max_key(dictionary):
    max_value = max(dictionary.values())
    for key, value in dictionary.items():
        if value == max_value:
            return key
        
class Map_to_Partition():
    """Convert Partition maps to Split flags to Partition vectors
    params:
    acc_level: options {0,1,2,3}, 
    
    """
    def __init__(self, qt_map, msbt_map, msdire_map, chroma_factor, lamb1=0.85, lamb2=[0.2, 0.2], lamb3=1.5, lamb4=0.3, \
                 lamb5=0.7, lamb6=0.3, lamb7=[0.5, 0.5, 0.5], lamb8 = [0.3, 0.2, 0.1],  block_size=64, no_dir=False, early_skip=False, debug_mode=False, acc_level=3):
        self.early_skip = early_skip
        self.no_dir = no_dir 
        self.qt_map = th_round2(qt_map, thd_1=lamb7[0],thd_2=lamb7[1]) 
        self.qt_map = self.qt_map.clip(max=3)
        self.msbt_map = th_round4(msbt_map, thd_1=lamb7[1], thd_2=lamb7[2])  # for QP37
        self.ori_qt_map = qt_map
        self.ori_msbt_map = msbt_map
        self.msdire_map = th_round(msdire_map, thd=0.5)  # threshold round
        self.ori_msdire_map = msdire_map
        
        self.block_ratio = block_size // 64
        self.chroma_factor = chroma_factor  # luma=1 choma=2

        # return p[0], p[1] and d
        self.par_vec = np.zeros((2, self.block_ratio * 16 + 1, self.block_ratio * 16 + 1), dtype=np.uint8)  # partition vector mat, p[0] hor, p[1] ver
        self.out_msdire_map = np.zeros((3, 16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)  
        self.out_qt_map = np.zeros((8 * self.block_ratio, 8 * self.block_ratio))
        self.out_bt_map = np.zeros_like(msbt_map)
        self.debug_mode = debug_mode
         
        self.cur_leaf_nodes = []  # store leaf nodes of Map Tree

        # lamb indicates several kinds of thresholds
        self.lamb1 = lamb1  # control no partition based on depth map
        self.lamb2 = lamb2  # 
        self.lamb3 = lamb3  # control hor or ver
        self.lamb4 = lamb4  # control number of minus
        self.lamb5 = lamb5  # control number of zero
        self.lamb6 = lamb6  # judge whether qt or not early, by default 0.3
        
        self.lamb8 = lamb8  # excludes non-split modes 
        self.time = 0
        
        self.acc_level = acc_level
        
    def split_cur_map(self, x, y, h, w, split_type):
        # split current cu [x,y,h,w]
        # split_type: 0 1 2 3 4 no bth btv tth ttv
        if split_type == 0:
            return [[x, y, h, w]]
        elif split_type == 1:  # bth
            return [[x, y, h//2, w], [x+h//2, y, h//2, w]]
        elif split_type == 2:  # btv
            return [[x, y, h, w//2], [x, y+w//2, h, w//2]]
        elif split_type == 3:  # tth
            return [[x, y, h//4, w], [x+h//4, y, h//2, w], [x+(h*3)//4, y, h//4, w]]
        elif split_type == 4:  # ttv
            return [[x, y, h, w//4], [x, y+w//4, h, w//2], [x, y+(w*3)//4, h, w//4]]
        elif split_type == 5:  # qt
            return [[x, y, h//2, w//2], [x, y+w//2, h//2, w//2], [x+h//2, y, h//2, w//2], [x+h//2, y+w//2, h//2, w//2]]
        else:
            print("Unknown split type!")

    def early_skip_chk(self, x, y, h, w, split_type):
        if split_type == 0:
            return [0]
        elif split_type == 1 or split_type == 2:  # bth
            return [1, 1]
        elif split_type == 3 or split_type == 4:  # tth
            return [1, 1, 1]
        elif split_type == 5:  # qt
            return [1, 1, 1, 1]
        else:
            print("Unknown split type!")

    def can_split_mode_list(self, x, y, h, w, cur_bt_map, depth, cur_qt_map, cur_bt_depth):
        """
        Output a list of candidate split types for the current CU (Coding Unit).
        Note: There is no QT (Quad Tree) partition here.

        Parameters:
        - x, y: Coordinates of the top-left corner of the current CU.
        - h, w: Height and width of the current CU.
        - cur_bt_map: The current bit map of the CU.
        - depth: Current depth of the CU.
        - cur_qt_map: The current Quad Tree map.
        - cur_bt_depth: The current bit depth of the CU.

        Returns:
        - A list of candidate split types for the CU.
        """

        # 0. Minimum CU edge length should be >= 4
        if h <= 1 and w <= 1:
            return [0]
        
        # If the current bit depth exceeds 3, no further splitting is possible
        if cur_bt_depth >= 3:
            return [0]
        
        # Compute the difference between QT maps
        comp_map_qt = self.qt_map[x // 2:(x + h) // 2, y // 2:(y + w) // 2] - cur_qt_map[x // 2:(x + h) // 2, y // 2:(y + w) // 2]
        
        # Compute the difference between MTT maps
        comp_map_mtt = self.msbt_map[cur_bt_depth, x:x + h, y:y + w] - cur_bt_map[x:x + h, y:y + w]
        
        # Count zeros in MTT depth map and QT map
        count_zero_mtt = len(np.where((comp_map_mtt).round() == 0)[0])
        count_zero_qt = len(np.where((comp_map_qt).round() == 0)[0])
        
        # If enough zeros are present, do not partition
        if (count_zero_qt >= self.lamb1 * h * w // 4) and (count_zero_mtt >= self.lamb1 * h * w):
            return [0]
        
        # Set terminal flag for QT if enough zeros are present
        qt_terminal = count_zero_qt >= self.lamb1 * h * w // 4
        
        # Check for condition to decide on QT map
        if cur_qt_map[x // 2, y // 2] == 0:
            ratio = self.lamb2[0]
        else:
            ratio = self.lamb2[1]
        
        # If a certain number of QT values exceeds a threshold, attempt a QT split
        if (np.sum(self.qt_map[x // 2:(x + h) // 2, y // 2:(y + w) // 2] > cur_qt_map[x // 2:(x + h) // 2, y // 2:(y + w) // 2]) >= ratio * h * w // 4) and depth <= 2:
            res_map = depth + 1 - self.qt_map[x // 2:(x + h) // 2, y // 2:(y + w) // 2].clip(min=0, max=depth + 1)
            self.qt_map[x // 2:(x + h) // 2, y // 2:(y + w) // 2] += res_map
            res_map = depth + 1 - self.ori_qt_map[x // 2:(x + h) // 2, y // 2:(y + w) // 2].clip(min=0, max=depth + 1)
            self.ori_qt_map[x // 2:(x + h) // 2, y // 2:(y + w) // 2] += res_map
            return [5]

        # If the accumulator level is zero or depth is greater than or equal to the current bit depth, do not partition
        if self.acc_level == 0 or cur_bt_depth >= self.acc_level:
            return [0]
        
        direction = 0  # 0 = Unknown, 1 = Horizontal, 2 = Vertical
        count_dire_nonzero = 0
        
        # If direction map is available, calculate the horizontal and vertical unit numbers
        if not self.no_dir:
            count_hor = len(np.where(self.msdire_map[cur_bt_depth, x:x + h, y:y + w] == 1)[0])
            count_ver = len(np.where(self.msdire_map[cur_bt_depth, x:x + h, y:y + w] == -1)[0])
            count_dire_nonzero = (count_hor + count_ver) / (h * w)
            direction = 1 if count_hor > self.lamb3 * count_ver else 2
        
        # Exclude partition if MTT difference exceeds threshold
        if (len(np.where((comp_map_mtt).round() != 0)[0]) > self.lamb8[cur_bt_depth] * h * w):
            exclude_non_split = True
        else:
            exclude_non_split = False

        initial_split_list = []

        # Try potential split modes
        for split_mode in [1, 2, 3, 4, 5]:
            if (split_mode == 1 or split_mode == 5) and (h // (2 * self.chroma_factor) == 0 or h % (2 * self.chroma_factor) != 0):
                continue  # Skip horizontal split if the height is too small or not divisible by chroma factor
            if (split_mode == 2 or split_mode == 5) and (w // (2 * self.chroma_factor) == 0 or w % (2 * self.chroma_factor) != 0):
                continue  # Skip vertical split if the width is too small or not divisible by chroma factor
            if split_mode == 3 and (h // (4 * self.chroma_factor) == 0 or h % (4 * self.chroma_factor) != 0):
                continue  # Skip TT horizontal split if the height is too small or not divisible by chroma factor
            if split_mode == 4 and (w // (4 * self.chroma_factor) == 0 or w % (4 * self.chroma_factor) != 0):
                continue  # Skip TT vertical split if the width is too small or not divisible by chroma factor
            if (split_mode == 1 or split_mode == 3) and direction == 2:
                continue  # Skip horizontal mode with vertical texture
            if (split_mode == 2 or split_mode == 4) and direction == 1:
                continue  # Skip vertical mode with horizontal texture
            if (split_mode == 3 or split_mode == 4) and depth == 0:
                continue  # Only binary or quadtree partitions allowed for CTU

            if split_mode == 5 and (depth >= 3 or direction != 0 or cur_bt_depth != 0 or qt_terminal or exclude_non_split):
                continue  # Skip if depth exceeds 3 or if other conditions prevent partitioning
            initial_split_list.append(split_mode)

        # If QT prediction is required, force TT mode
        tt_flag = False
        if cur_bt_depth == 0:
            previous_msbt = 0
        else:
            previous_msbt = self.msbt_map[cur_bt_depth - 1, x:x + h, y:y + w]

        if (self.msbt_map[cur_bt_depth, x:x + h, y:y + w] - previous_msbt).max() == 2 and (3 in initial_split_list or 4 in initial_split_list):
            if 0 in initial_split_list:
                initial_split_list.remove(0)
            if 1 in initial_split_list:
                initial_split_list.remove(1)
            if 2 in initial_split_list:
                initial_split_list.remove(2)
            if 5 in initial_split_list:
                initial_split_list.remove(5)
            tt_flag = True

        # Further checking for each sub-CU after the initial split
        temp_cost_dict = {}
        candidate_mode_list = []
        for split_mode in initial_split_list:
            sub_map_xyhw = self.split_cur_map(x, y, h, w, split_mode)
            map_temp = np.zeros_like(cur_bt_map, dtype=np.int8)
            map_temp[:, :] = cur_bt_map[:, :]
            map_temp_qt = np.zeros_like(cur_qt_map, dtype=np.int8)
            map_temp_qt[:, :] = cur_qt_map[:, :]
            
            # Evaluate the cost for this split mode
            temp_cost = 0
            score_list = []
            for sub_map_id in range(len(sub_map_xyhw)):  # traverse all sub-blocks
                [sub_x, sub_y, sub_h, sub_w] = sub_map_xyhw[sub_map_id]
                if split_mode == 5:
                    map_temp_qt[sub_x // 2:(sub_x + sub_h) // 2, sub_y // 2:(sub_y + sub_w) // 2] += 1
                    comp_map = self.qt_map[sub_x // 2:(sub_x + sub_h) // 2, sub_y // 2:(sub_y + sub_w) // 2] - map_temp_qt[sub_x // 2:(sub_x + sub_h) // 2, sub_y // 2:(sub_y + sub_w) // 2]
                    num_pixel = sub_h * sub_w // 4
                else:
                    map_temp[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                    comp_map = self.msbt_map[cur_bt_depth, sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] - map_temp[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w]
                    num_pixel = sub_h * sub_w

                count_zero = len(np.where(comp_map >= 0)[0])  # Count of zero and positive values
                score_list.append(count_zero / num_pixel)
                temp_cost += count_zero
            
            temp_cost_dict[str(split_mode)] = temp_cost

            # Add valid split modes based on calculated cost and score
            if sum(score_list) / len(score_list) >= 0.6:
                candidate_mode_list.append(split_mode)

        # Return the list of valid candidate split modes
        if len(candidate_mode_list) == 0:
            if exclude_non_split and len(temp_cost_dict) > 0:
                max_key = max(temp_cost_dict, key=temp_cost_dict.get)
                candidate_mode_list = [int(max_key)]
            else:
                candidate_mode_list = [0]

        return candidate_mode_list



    def get_candidate_map_tree(self, map_node):
        if map_node.depth >= 7:
            return
        cur_cus = map_node.cus
        cu_num = len(cur_cus)
        cur_bt_map = map_node.bt_map
        cur_dire_map = map_node.dire_map
        cur_depth = map_node.depth
        cur_qt_map = map_node.qt_map
        cur_bt_depth_list = map_node.bt_depth
        cur_out_dire_map = map_node.out_dire_map
        cur_out_mt_map = map_node.out_mt_map
        if self.early_skip:
            early_skip_list = map_node.early_skip_list
        cus_candidate_mode_list = []
        for i in range(cu_num):
            cus_candidate_mode_list.append([])  # store all candidate split modes of every CU
        for cu_id in range(cu_num):  # traverse all CUs in current map
            if self.early_skip and early_skip_list[cu_id] == 0:
                cus_candidate_mode_list[cu_id] = [0]
                continue
            [cu_x, cu_y, cu_h, cu_w] = cur_cus[cu_id]
            candidate_mode_list = self.can_split_mode_list(cu_x, cu_y, cu_h, cu_w, cur_bt_map, cur_depth, cur_qt_map, cur_bt_depth_list[cu_id])
            if len(candidate_mode_list) == 0:  
                return
            cus_candidate_mode_list[cu_id] += candidate_mode_list

        s = Search(cus_candidate_mode_list)
        partition_modes = s.get_partition_modes()


        if False not in [(set(modes) == set([0])) for modes in partition_modes]:
            return
        
        if len(partition_modes) >= 1024:
            if cu_num <= 32:
                partition_modes = random.sample(partition_modes, 256)
            else:
                # return
                partition_modes = random.sample(partition_modes, 8)

        for cus_modes in partition_modes:  # traverse all possible combination cu split modes
            child_qt_map = np.zeros_like(cur_qt_map, dtype=np.int8)
            child_bt_map = np.zeros_like(cur_bt_map, dtype=np.int8)
            child_dire_map = np.zeros_like(cur_dire_map, dtype=np.int8)
            child_bt_map[:, :] = cur_bt_map                
            child_qt_map[:, :] = cur_qt_map
            # out map
            child_out_mt_map = np.zeros_like(cur_out_mt_map)
            child_out_mt_map[:,:,:] = cur_out_mt_map[:,:,:]
            child_out_dire_map = np.zeros_like(cur_out_dire_map)
            child_out_dire_map[:,:,:] = cur_out_dire_map[:,:,:]
            # child_dire_map[:, :] = cur_dire_map
            child_cus = []
            child_bt_depths = []
            if self.early_skip:
                early_skip_list = []
            for cu_id in range(cu_num):  # traverse all cu
                [cu_x, cu_y, cu_h, cu_w] = cur_cus[cu_id]  # location and size of current CU
                cu_mode = cus_modes[cu_id]  # split mode of current CU
                child_map_xyhw = self.split_cur_map(cu_x, cu_y, cu_h, cu_w, cu_mode)
                child_cus += child_map_xyhw
                cu_bt_depth = cur_bt_depth_list[cu_id]
                child_bt_depth = [cu_bt_depth + 1 if cu_mode in [1,2,3,4] else cu_bt_depth for i in range(len(child_map_xyhw))]
                child_bt_depths += child_bt_depth
                
                if self.early_skip:
                    early_skip_list += self.early_skip_chk(cu_x, cu_y, cu_h, cu_w, cu_mode)
                if cu_mode == 0:  # no partition
                    child_dire_map[cu_x:cu_x+cu_h, cu_y:cu_y+cu_w] = 0
                    if cu_bt_depth <= 2:
                        child_out_dire_map[cu_bt_depth, cu_x:cu_x+cu_h, cu_y:cu_y+cu_w] = 0
                    continue
                elif cu_mode == 1 or cu_mode == 3:  # horizontal
                    child_dire_map[cu_x:cu_x + cu_h, cu_y:cu_y + cu_w] = 1
                    child_out_dire_map[cu_bt_depth, cu_x:cu_x+cu_h, cu_y:cu_y+cu_w] = 1
                elif cu_mode == 2 or cu_mode == 4:  # vertical
                    child_dire_map[cu_x:cu_x + cu_h, cu_y:cu_y + cu_w] = -1
                    child_out_dire_map[cu_bt_depth, cu_x:cu_x+cu_h, cu_y:cu_y+cu_w] = -1

                for sub_block_id in range(len(child_map_xyhw)):  # traverse all sub-blocks
                    [sub_x, sub_y, sub_h, sub_w] = child_map_xyhw[sub_block_id]
                    if cu_mode == 5:
                        child_qt_map[sub_x//2:(sub_x + sub_h)//2, sub_y//2:(sub_y + sub_w)//2] += 1
                        # child_bt_map[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 2
                    else:
                        child_bt_map[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                        child_out_mt_map[cu_bt_depth, sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] = 1
                    if (cu_mode == 3 or cu_mode == 4) and (sub_block_id != 1):
                        # depth +2 in the first and last parts
                        child_bt_map[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                        child_out_mt_map[cu_bt_depth, sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                
            child_map_node = Map_Node(bt_map=child_bt_map, dire_map=child_dire_map, \
                    cus=child_cus, parent=map_node, depth=cur_depth+1, qt_map=child_qt_map, \
                        early_skip=self.early_skip, bt_depth=child_bt_depths, out_mt_map=child_out_mt_map, out_dire_map=child_out_dire_map)
            if self.early_skip:
                child_map_node.early_skip_list = early_skip_list
            self.get_candidate_map_tree(child_map_node)
            map_node.children.append(child_map_node)

    def get_leaf_nodes(self, map_node):
        if len(map_node.children) == 0:  # no children node
            self.cur_leaf_nodes.append(map_node)
        else:
            for child_node in map_node.children:
                self.get_leaf_nodes(child_node)

    def print_tree(self, map_node, depth):
        print('**********************')
        print('node', depth)
        print(map_node.mtt_depth)
        print(map_node.bt_map)
        print(map_node.cus)
        print(len(map_node.children))
        print('**********************')
        if len(map_node.children) != 0:
            for child_node in map_node.children:
                self.print_tree(child_node, depth+1)

    def set_bt_partition_vector(self, x, y, h, w):
        init_qt_map = np.zeros((8 * self.block_ratio, 8 * self.block_ratio), dtype=np.int8)
        init_bt_map = np.zeros((16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        init_dire_map = np.zeros((16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        init_out_mt_map = np.zeros((3, 16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        init_out_dire_map = np.zeros((3, 16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        map_root = Map_Node(qt_map=init_qt_map, bt_map=init_bt_map, dire_map=init_dire_map, depth=0, cus=[[x, y, h, w]], early_skip = self.early_skip, bt_depth=[0], out_mt_map=init_out_mt_map, out_dire_map = init_out_dire_map)
        if self.early_skip:
            map_root.early_skip_list = [1]
        self.get_candidate_map_tree(map_root)  # build Map Tree

        # self.print_tree(map_root, 0)
        self.cur_leaf_nodes = []
        self.get_leaf_nodes(map_root)  # get lead nodes list of Map Tree
        
        self.ori_msbt_map = self.msbt_map
        self.ori_msdire_map = self.msdire_map
        error_list = []
        if len(self.cur_leaf_nodes) > 1:
            for node4 in self.cur_leaf_nodes:
                qt_map = node4.qt_map
                bt_map = node4.out_mt_map
                dire_map = node4.out_dire_map
                error = np.sum(np.abs(qt_map - self.ori_qt_map)) + \
                        np.sum(np.abs(bt_map - self.ori_msbt_map))
              
                error_list.append(error)
            min_index = error_list.index(min(error_list))
        else:
            min_index = 0
        best_node4 = self.cur_leaf_nodes[min_index]
        # best dir map
        
        
        # best qt map
        self.out_qt_map = best_node4.qt_map
        self.out_msdire_map = best_node4.out_dire_map
        self.out_bt_map = best_node4.out_mt_map
        
        # best_bt_map = self.cur_leaf_nodes[min_index].bt_map  # best bt map
        best_cus = self.cur_leaf_nodes[min_index].cus  # best partition
        # print('*********************************************************')
        # print('error_list', error_list, np.min(error_list))
        # print('x, y, h, w', x, y, h, w)
        # print(best_bt_map[x:x + h, y:y + w])
        # print(self.cur_leaf_nodes[2].cus)
        delete_tree(map_root)
        for cu in best_cus:
            [cu_x, cu_y, cu_h, cu_w] = cu
            for i_w in range(cu_w):  # set CU horizontal edges
                self.par_vec[0, cu_x, cu_y + i_w] = 1
                self.par_vec[0, cu_x + cu_h, cu_y + i_w] = 1
            for i_h in range(cu_h):  # set CU vertical edges
                self.par_vec[1, cu_x + i_h, cu_y] = 1
                self.par_vec[1, cu_x + i_h, cu_y + cu_w] = 1

    def get_partition(self):
        self.set_bt_partition_vector(0, 0, 32, 32)
        return self.par_vec, self.out_msdire_map, self.out_qt_map



def map_to_partition_qtmtt(qt_map, bt_map, dire_map, qp, chroma_factor, block_size=128, early_skip=True, debug_mode=False, no_dir=False, acc_level=3, frm_id=None, block_x=None, block_y=None):
    """
    Arguments:
    - qt_map: The quantization table map.
    - bt_map: The binary tree map.
    - dire_map: The direction map.
    - qp: The quantization parameter.
    - chroma_factor: The chroma factor for the block.
    - block_size: The block size for partitioning, default is 128.
    - early_skip: Whether to skip later partitioning steps that do not affect previous layers, default is True.
    - debug_mode: If True, outputs the optimized QT map, BT map, and direction map, default is False.
    - no_dir: If True, disables direction-based decisions, default is False.
    - acc_level: The accuracy level for partitioning, default is 3.
    - frm_id: Frame ID, used for multi-threading.
    - block_x: X-coordinate of the block.
    - block_y: Y-coordinate of the block.
    
    Returns:
    - Depending on the `debug_mode`, returns either the partitioned maps or the partitioned maps along with additional information like frame ID and block coordinates.

    Description:
    - The function decides the best partitioning method for a given block based on various parameters like the QP, chroma factor, and the maps provided. 
    - It uses lamb parameters to control various factors such as early termination, QT completion, direction balancing, and thresholds for QT and MTT.
    """
    
    # Define the lamb parameters based on the QP value.
    if qp == 37:
        lamb_params = {
            'lamb1': 0.85, 'lamb2': [0.7, 0.85], 'lamb3': 1, 'lamb7': [0.5, 0.5, 0.4, 0.5], 
            'lamb5': [0.3, 0.2, 0.1], 'lamb8': [0.3, 0.2, 0.1]
        }
    elif qp == 32:
        lamb_params = {
            'lamb1': 0.85, 'lamb2': [0.4, 0.2], 'lamb3': 1, 'lamb7': [0.5, 0.2, 0.5, 0.5], 
            'lamb5': [0.3, 0.2, 0.1], 'lamb8': [0.3, 0.2, 0.1]
        }
    elif qp == 27:
        lamb_params = {
            'lamb1': 0.85, 'lamb2': [0.4, 0.9], 'lamb3': 1, 'lamb7': [0.3, 0.2, 0.5, 0.5], 
            'lamb5': [0.3, 0.2, 0.1], 'lamb8': [0.3, 0.2, 0.1]
        }
    elif qp == 22:
        lamb_params = {
            'lamb1': 0.85, 'lamb2': [0.4, 0.9], 'lamb3': 1, 'lamb7': [0.3, 0.2, 0.5, 0.5], 
            'lamb5': [0.3, 0.2, 0.1], 'lamb8': [0.3, 0.2, 0.1]
        }

    # Initialize the partitioning process with the given parameters.
    partition = Map_to_Partition(qt_map, bt_map, dire_map, chroma_factor, block_size=128, 
                                 early_skip=early_skip, debug_mode=debug_mode, no_dir=no_dir, 
                                 acc_level=acc_level, **lamb_params)
    
    # Get the partitioned maps and depth information.
    p, d, q = partition.get_partition()  # Partition map, MTT direction map, QT depth map

    # If debug mode is enabled, return the detailed partition maps.
    if debug_mode:
        return partition.out_qt_map, partition.out_bt_map, partition.out_msdire_map
    else:
        # If frame ID is provided, return the partitioned maps along with additional information.
        if frm_id is None:
            return p[0][:16 * block_ratio, :16 * block_ratio], p[1][:16 * block_ratio, :16 * block_ratio], d, q
        else:
            # Multi-threaded processing, returning additional frame-related information.
            return p[0][:16 * block_ratio, :16 * block_ratio], p[1][:16 * block_ratio, :16 * block_ratio], d, q, frm_id, block_x, block_y



def get_sequence_partition_for_VTM(qt_map, bt_map, dire_map, is_luma, save_path, frm_num, frm_width, frm_height, blockSize=64):
    # partition maps --> partition edge vector + sequence qt depth map + sequence direction map
    # dire_map = np.where(dire_map < 0, np.ones_like(dire_map) * 2, dire_map)
    chroma_factor = 2
    if is_luma:
        chroma_factor = 1
    if save_path is not None:
        out_file = open(save_path, 'w')
    block_num_in_height = frm_height // blockSize
    block_num_in_width = frm_width // blockSize
    seq_partition_hor_mat = np.zeros((frm_num, block_num_in_height * 16, block_num_in_width * 16)) # store whether is partition edge or not (1 or 0) for edges of all the basic unit (4*4)
    seq_partition_ver_mat = np.zeros((frm_num, block_num_in_height * 16, block_num_in_width * 16))
    seq_qt_map = np.zeros((frm_num, block_num_in_height * 8, block_num_in_width * 8))
    seq_dire_map = np.zeros((frm_num, 3, block_num_in_height * 16, block_num_in_width * 16))
    for frm_id in range(frm_num):
        print("Frame ", frm_id)
        frm_block_id = frm_id * block_num_in_height * block_num_in_width
        for block_x in range(block_num_in_height):
            for block_y in range(block_num_in_width):
                block_id = frm_block_id + block_x * block_num_in_width + block_y
                hor_mat, ver_mat, out_dire_map, valid_qt_map = map_to_parititon_noQT(bt_map[block_id], dire_map[block_id], chroma_factor)
                seq_partition_hor_mat[frm_id, block_x * 16:(block_x + 1) * 16, block_y * 16:(block_y + 1) * 16] = hor_mat
                seq_partition_ver_mat[frm_id, block_x * 16:(block_x + 1) * 16, block_y * 16:(block_y + 1) * 16] = ver_mat
                seq_qt_map[frm_id, block_x * 8:(block_x + 1) * 8, block_y * 8:(block_y + 1) * 8] = valid_qt_map
                seq_dire_map[frm_id, :, block_x * 16:(block_x + 1) * 16, block_y * 16:(block_y + 1) * 16] = out_dire_map
        if save_path is not None:
            hor_vec = seq_partition_hor_mat[frm_id].reshape(-1).astype(np.uint8)
            ver_vec = seq_partition_ver_mat[frm_id].reshape(-1).astype(np.uint8)
            qtdepth_vec = seq_qt_map[frm_id].reshape(-1).astype(np.uint8)
            dire_vec = seq_dire_map[frm_id].reshape(-1).astype(np.int8)
            for i in range(hor_vec.size):  # horizontal edge vector
                out_file.write(str(hor_vec[i]) + '\n')
            for i in range(ver_vec.size):  # vertical edge vector
                out_file.write(str(ver_vec[i]) + '\n')
            for i in range(qtdepth_vec.size):  # qt depth vector
                out_file.write(str(qtdepth_vec[i]) + '\n')
            for i in range(dire_vec.size):  # direction vector
                out_file.write(str(dire_vec[i]) + '\n')
            # print(hor_vec.size)
            # print(qtdepth_vec.size)
            # print(dire_vec.size)
    if save_path is not None:
        out_file.close()

