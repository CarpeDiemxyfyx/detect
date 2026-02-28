"""
自定义模块注册工具
将 SADR 和 BDFR 模块注册到 Ultralytics YOLO 框架中，
使其能在 YAML 配置文件中直接引用

使用方法:
    在训练/推理脚本中, 在加载模型之前调用:
        from models.register_modules import register_custom_modules
        register_custom_modules()
    
    或者直接:
        import models.register_modules  # 导入即自动注册
"""
import sys
import os

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from models.modules.sadr import SADR
from models.modules.bdfr import BDFR
from models.modules.tvad import TVAD


def register_custom_modules():
    """
    将自定义模块注册到 ultralytics 的模块查找表中
    
    Ultralytics YOLO 在解析 YAML 配置时, 会在以下位置查找模块类:
    1. ultralytics.nn.modules 内置模块
    2. 通过 task.py 中的 parse_model() 函数查找
    
    我们通过将自定义模块添加到 ultralytics.nn.modules 的命名空间来实现注册
    
    注: TVAD 为纯推理阶段模块, 不参与YAML模型定义, 但仍注册以保持一致性
    """
    try:
        import ultralytics.nn.modules as nn_modules
        
        # 注册 SADR
        if not hasattr(nn_modules, 'SADR'):
            setattr(nn_modules, 'SADR', SADR)
        
        # 注册 BDFR
        if not hasattr(nn_modules, 'BDFR'):
            setattr(nn_modules, 'BDFR', BDFR)
        
        # 注册 TVAD (推理阶段模块)
        if not hasattr(nn_modules, 'TVAD'):
            setattr(nn_modules, 'TVAD', TVAD)
        
        # 同时注册到 ultralytics.nn.modules.__all__ (如果存在)
        if hasattr(nn_modules, '__all__'):
            all_list = list(nn_modules.__all__)  # tuple → list 兼容
            for name in ['SADR', 'BDFR', 'TVAD']:
                if name not in all_list:
                    all_list.append(name)
            nn_modules.__all__ = tuple(all_list)  # 写回原类型
        
        # 注册到 tasks.py 的解析函数使用的模块查找空间
        try:
            from ultralytics.nn import tasks
            if hasattr(tasks, '__dict__'):
                tasks.__dict__['SADR'] = SADR
                tasks.__dict__['BDFR'] = BDFR
                tasks.__dict__['TVAD'] = TVAD
        except ImportError:
            pass
        
        # 同时在 ultralytics.nn 层级注册
        try:
            import ultralytics.nn as nn_pkg
            for name, cls in [('SADR', SADR), ('BDFR', BDFR), ('TVAD', TVAD)]:
                if not hasattr(nn_pkg, name):
                    setattr(nn_pkg, name, cls)
        except ImportError:
            pass
            
        return True
        
    except ImportError as e:
        print(f"[!] Warning: ultralytics not found, skipping registration: {e}")
        print("    Install with: pip install ultralytics")
        return False


# 自动注册 (导入此模块时即执行)
_registered = register_custom_modules()
