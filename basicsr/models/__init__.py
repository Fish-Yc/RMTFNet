import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'basicsr.models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model



# # 扫描`models`目录下所有以`_model.py`结尾的文件
# model_folder = osp.dirname(osp.abspath(__file__))
# model_filenames = [
#     osp.splitext(osp.basename(v))[0]  # 提取文件名（不含后缀）
#     for v in scandir(model_folder)    # 遍历目录下的文件
#     if v.endswith('_model.py')         # 筛选以`_model.py`结尾的文件
# ]
#
# # 动态导入所有模型模块
# _model_modules = [
#     importlib.import_module(f'basicsr.models.{file_name}')
#     for file_name in model_filenames
# ]
#
# def build_model(opt):
#     """根据配置构建模型"""
#     opt = deepcopy(opt)  # 深拷贝配置，避免原始配置被修改
#     model = MODEL_REGISTRY.get(opt['model_type'])(opt)  # 从注册表获取模型类并实例化
#     logger = get_root_logger()
#     logger.info(f'Model [{model.__class__.__name__}] is created.')
#     return model