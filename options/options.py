import logging
import os
import os.path as osp

import yaml
from .utils import OrderedYaml

Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode='rb') as f:
        opt = yaml.load(f, Loader=Loader)
    if is_train:
        if not os.path.exists(os.path.join(opt['path']['cp_path'], opt['model_type'], opt['enc_mode'], 'QP' + str(opt['qp']))):
            os.makedirs(os.path.join(opt['path']['cp_path'], opt['model_type'], opt['enc_mode'], 'QP' + str(opt['qp'])))
        opt['path']['cp_path'] = os.path.join(opt['path']['cp_path'], opt['model_type'], opt['enc_mode'], 'QP' + str(opt['qp']))
        print("checkpoints path: ", os.path.join(opt['path']['cp_path'], opt['model_type'], opt['enc_mode'], 'QP' + str(opt['qp'])))

        mean_list = (len(opt['train']['mtt_layer_weight']) / sum(opt['train']['mtt_layer_weight']))
        opt['train']['mtt_layer_weight'] = [ele * mean_list for ele in opt['train']['mtt_layer_weight']]
        print("normalized weight of mtt_layers: ", opt['train']['mtt_layer_weight'])

        mean_list = (len(opt['train']['qt_mtt_weight']) / sum(opt['train']['qt_mtt_weight']))
        opt['train']['qt_mtt_weight'] = [ele * mean_list for ele in opt['train']['qt_mtt_weight']]
        print("weight for qt: %.2f" % opt['train']['qt_mtt_weight'][0], "\t for mtt_layer: %.2f" % opt['train']['qt_mtt_weight'][1])

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get('pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'], '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'], '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
