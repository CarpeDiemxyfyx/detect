"""
训练脚本
基于改进YOLOv11的道路异常事件检测模型训练

功能:
- 从 YAML 配置文件读取全部训练超参数
- 命令行参数可覆盖配置文件中的任意值
- 支持选择不同模型配置 (baseline / sadr / bdfr / full)
- 自动注册自定义模块
- 支持断点续训
- 训练日志记录

使用方法:
    # 使用配置文件训练 (推荐)
    python scripts/train.py --cfg configs/train_full.yaml

    # 配置文件 + 命令行覆盖部分参数
    python scripts/train.py --cfg configs/train_full.yaml --epochs 100 --batch 8

    # 快捷方式: 直接指定预设名 (自动映射到 configs/train_<name>.yaml)
    python scripts/train.py --config full --epochs 200 --batch 16

    # 断点续训
    python scripts/train.py --resume runs/road_anomaly/yolov11m_improved/weights/last.pt
"""
import os
import sys
import argparse
from pathlib import Path
from copy import deepcopy

import yaml
import warnings

# 抑制 PyTorch 确定性算法警告 (SADR 的 AdaptiveAvgPool2d 触发, 每epoch重复)
warnings.filterwarnings('ignore', message='.*does not have a deterministic implementation.*')

# 添加项目根目录到 sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 注册自定义模块 (必须在加载YOLO之前, 导入即自动注册)
from models.register_modules import register_custom_modules  # noqa: F401
register_custom_modules()

from ultralytics import YOLO


# --config 快捷名 → 配置文件路径的映射
CONFIG_SHORTCUTS = {
    'baseline': 'configs/train_baseline.yaml',
    'sadr':     'configs/train_sadr.yaml',
    'bdfr':     'configs/train_bdfr.yaml',
    'full':     'configs/train_full.yaml',
}

# 默认配置 (当配置文件中缺少某项时使用)
DEFAULT_CFG = {
    'model': {
        'yaml': 'models/yolov11m-road-anomaly.yaml',
        'pretrained': 'yolo11m.pt',
        'name': 'yolov11m_improved',
        'desc': 'YOLOv11m + SADR + BDFR (完整改进)',
    },
    'data': {
        'dataset': 'dataset/road_anomaly.yaml',
        'imgsz': 640,
    },
    'train': {
        'epochs': 200,
        'batch': 4,           # RTX 4060 (8GB) 适配
        'device': '0',
        'workers': 4,
        'amp': True,
        'patience': 30,
        'save_period': 10,
        'cos_lr': True,
        'close_mosaic': 10,
        'val': True,
        'plots': True,
        'exist_ok': True,
    },
    'optimizer': {
        'type': 'AdamW',
        'lr0': 0.0005,        # √(4/16)×0.001, AdamW小batch缩放
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
    },
    'augment': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.001,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.1,
    },
    'loss': {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    },
    'output': {
        'project': 'runs/road_anomaly',
    },
}


def deep_update(base: dict, override: dict) -> dict:
    """递归合并字典, override 覆盖 base 中的值"""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(cfg_path: str) -> dict:
    """
    从 YAML 文件加载配置, 与默认配置合并.
    缺失项自动用 DEFAULT_CFG 补全.
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.is_absolute():
        cfg_path = Path(PROJECT_ROOT) / cfg_path

    if not cfg_path.exists():
        print(f"[!] 配置文件不存在: {cfg_path}")
        print(f"    可用预设: {list(CONFIG_SHORTCUTS.keys())}")
        sys.exit(1)

    with open(cfg_path, 'r', encoding='utf-8') as f:
        user_cfg = yaml.safe_load(f) or {}

    # 合并: DEFAULT_CFG ← 配置文件
    cfg = deep_update(DEFAULT_CFG, user_cfg)
    cfg['_source'] = str(cfg_path)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """
    命令行参数覆盖配置文件中的值.
    仅当用户显式传入时才覆盖 (通过 _explicitly_set 标记判断).
    """
    cli_map = {
        # 命令行参数名 → (配置文件路径, 类型)
        'epochs':     ('train.epochs',        int),
        'batch':      ('train.batch',         int),
        'imgsz':      ('data.imgsz',          int),
        'lr':         ('optimizer.lr0',        float),
        'patience':   ('train.patience',       int),
        'device':     ('train.device',         str),
        'workers':    ('train.workers',        int),
        'pretrained': ('model.pretrained',     str),
    }

    for cli_name, (cfg_key, _) in cli_map.items():
        if cli_name not in args._explicitly_set:
            continue
        value = getattr(args, cli_name, None)
        if value is None:
            continue
        # 按 "." 分层写入
        keys = cfg_key.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    return cfg


def print_config(cfg: dict):
    """美观地打印当前训练配置"""
    print(f"\n{'='*64}")
    print(f"  📄 配置文件:  {cfg.get('_source', 'N/A')}")
    print(f"  🏷  模型:      {cfg['model']['desc']}")
    print(f"  📐 结构:      {cfg['model']['yaml']}")
    print(f"  📦 预训练:    {cfg['model']['pretrained']}")
    print(f"  📊 数据集:    {cfg['data']['dataset']}")
    print(f"  {'─'*60}")
    print(f"  epochs={cfg['train']['epochs']}  batch={cfg['train']['batch']}  "
          f"imgsz={cfg['data']['imgsz']}  device={cfg['train']['device']}")
    print(f"  lr0={cfg['optimizer']['lr0']}  optimizer={cfg['optimizer']['type']}  "
          f"patience={cfg['train']['patience']}")
    print(f"  mosaic={cfg['augment']['mosaic']}  mixup={cfg['augment']['mixup']}  "
          f"copy_paste={cfg['augment']['copy_paste']}")
    print(f"  box={cfg['loss']['box']}  cls={cfg['loss']['cls']}  "
          f"dfl={cfg['loss']['dfl']}")
    print(f"{'='*64}\n")


def _materialize_lazy_modules(trainer):
    """
    处理自定义模块 (SADR/BDFR) 的延迟初始化.
    
    问题背景:
      Ultralytics 的 parse_model 对自定义模块不执行 width_multiple 通道缩放,
      而是把 YAML 参数原样传入构造函数. 因此 SADR/BDFR 采用延迟构建策略:
      在首次 forward 时按实际输入通道数构建子模块. 但优化器在首次 forward
      之前就已创建, 延迟构建的参数不在优化器的参数组中.
    
    解决方案:
      本回调在 on_pretrain_routine_end 阶段 (优化器创建后、训练循环开始前) 执行 dummy
      forward 触发所有延迟构建, 然后将新增参数按类型分组加入优化器.
    """
    import torch
    import torch.nn as nn
    
    model = trainer.model
    
    # 检查是否存在未构建的延迟模块
    has_lazy = any(
        hasattr(m, '_built') and not m._built
        for m in model.modules()
    )
    if not has_lazy:
        return
    
    device = next(model.parameters()).device
    imgsz = trainer.args.imgsz
    
    # 记录 dummy forward 之前的参数 ID
    params_before = {id(p) for p in model.parameters()}
    
    # dummy forward 触发延迟初始化
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        try:
            model(dummy)
        except Exception as e:
            print(f"  ⚠️ 延迟初始化 dummy forward 失败: {e}")
            if was_training:
                model.train()
            return
    if was_training:
        model.train()
    
    # 按 Ultralytics 标准将新增参数分为三组:
    #   param_groups[0] = BN 权重 (无 weight_decay)
    #   param_groups[1] = 卷积/线性层权重 (有 weight_decay)
    #   param_groups[2] = 偏置项 (无 weight_decay)
    g_bn, g_weight, g_bias = [], [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if id(param) in params_before or not param.requires_grad:
                continue
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                g_bn.append(param)
            elif name == 'bias':
                g_bias.append(param)
            else:
                g_weight.append(param)
    
    total_new = len(g_bn) + len(g_weight) + len(g_bias)
    if total_new == 0:
        return
    
    # 注入已有优化器的对应参数组
    if len(trainer.optimizer.param_groups) >= 3:
        trainer.optimizer.param_groups[0]['params'].extend(g_bn)
        trainer.optimizer.param_groups[1]['params'].extend(g_weight)
        trainer.optimizer.param_groups[2]['params'].extend(g_bias)
    else:
        all_new = g_bn + g_weight + g_bias
        trainer.optimizer.add_param_group({'params': all_new, 'lr': trainer.args.lr0})
    
    total_params = sum(p.numel() for p in g_bn + g_weight + g_bias)
    print(f"\n  ✅ 自定义模块延迟初始化完成:")
    print(f"     BN 参数: {len(g_bn)} 组 | 权重: {len(g_weight)} 组 | 偏置: {len(g_bias)} 组")
    print(f"     共 {total_params:,} 个新增可训练参数已加入优化器\n")


def train(cfg: dict):
    """执行训练"""

    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)

    print_config(cfg)

    # 加载模型
    model = YOLO(cfg['model']['yaml'])

    # 加载预训练权重
    pretrained = cfg['model']['pretrained']
    if os.path.exists(pretrained):
        model = model.load(pretrained)
        print(f"  ✅ 已加载预训练权重: {pretrained}")
    else:
        print(f"  ⚠️  预训练权重不存在: {pretrained}, 使用随机初始化")
        print(f"      可从 ultralytics 下载: yolo11m.pt")

    # 构建 model.train() 参数字典
    train_args = {
        'data':             cfg['data']['dataset'],
        'imgsz':            cfg['data']['imgsz'],

        # 训练基本参数
        'epochs':           cfg['train']['epochs'],
        'batch':            cfg['train']['batch'],
        'device':           cfg['train']['device'],
        'workers':          cfg['train']['workers'],
        'amp':              cfg['train']['amp'],
        'patience':         cfg['train']['patience'],
        'save_period':      cfg['train']['save_period'],
        'cos_lr':           cfg['train']['cos_lr'],
        'close_mosaic':     cfg['train']['close_mosaic'],
        'val':              cfg['train']['val'],
        'plots':            cfg['train']['plots'],
        'exist_ok':         cfg['train']['exist_ok'],

        # 优化器
        'optimizer':        cfg['optimizer']['type'],
        'lr0':              cfg['optimizer']['lr0'],
        'lrf':              cfg['optimizer']['lrf'],
        'momentum':         cfg['optimizer']['momentum'],
        'weight_decay':     cfg['optimizer']['weight_decay'],
        'warmup_epochs':    cfg['optimizer']['warmup_epochs'],
        'warmup_momentum':  cfg['optimizer']['warmup_momentum'],

        # 数据增强
        **cfg['augment'],

        # 损失函数权重
        'box':              cfg['loss']['box'],
        'cls':              cfg['loss']['cls'],
        'dfl':              cfg['loss']['dfl'],

        # 输出路径 (绝对路径, 避免 Ultralytics runs_dir 嵌套)
        'project':          str(Path(PROJECT_ROOT) / cfg['output']['project']),
        'name':             cfg['model']['name'],

        'save':             True,
    }

    # 注册回调: 模型和优化器就绪后, 训练循环开始前,
    # 触发 SADR/BDFR 延迟初始化并将新增参数加入优化器
    model.add_callback("on_pretrain_routine_end", _materialize_lazy_modules)

    results = model.train(**train_args)

    # 训练完成后评估
    print(f"\n{'='*64}")
    print("  🎯 训练完成! 执行最终评估...")
    print(f"{'='*64}")

    best_pt = Path(PROJECT_ROOT) / cfg['output']['project'] / cfg['model']['name'] / 'weights' / 'best.pt'
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(
        data=cfg['data']['dataset'],
        imgsz=cfg['data']['imgsz'],
        batch=cfg['train']['batch'],
        device=cfg['train']['device'],
        plots=True,
    )

    print(f"\n  📈 最终评估结果:")
    print(f"    mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"    mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"    Precision:     {metrics.box.mp:.4f}")
    print(f"    Recall:        {metrics.box.mr:.4f}")

    return results


class ExplicitArgParser(argparse.ArgumentParser):
    """
    扩展 ArgumentParser, 记录哪些参数是用户在命令行中显式传入的,
    以此区分 "用户传了 --epochs 200" 和 "默认值 200", 避免覆盖配置文件.
    """
    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args, namespace)
        # 找出用户显式传入的参数
        explicitly_set = set()
        for action in self._actions:
            if action.dest == 'help':
                continue
            for opt in action.option_strings:
                # 检查原始命令行中是否包含该参数
                if args is None:
                    import sys as _sys
                    check_args = _sys.argv[1:]
                else:
                    check_args = args
                if opt in check_args:
                    explicitly_set.add(action.dest)
                    break
        ns._explicitly_set = explicitly_set
        return ns


def main():
    print("[✓] 自定义模块注册完成 (SADR / BDFR / TVAD)")
    
    parser = ExplicitArgParser(
        description='道路异常事件检测 - 训练脚本',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 配置文件 (二选一)
    parser.add_argument('--cfg', type=str, default=None,
                        help='YAML 配置文件路径, 例如:\n'
                             '  --cfg configs/train_full.yaml')
    parser.add_argument('--config', type=str, default=None,
                        choices=['baseline', 'sadr', 'bdfr', 'full'],
                        help='快捷预设名 (自动映射到 configs/train_<name>.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='断点续训权重路径')

    # 可选命令行覆盖参数
    parser.add_argument('--epochs',     type=int,   default=None, help='覆盖训练轮数')
    parser.add_argument('--batch',      type=int,   default=None, help='覆盖 batch size')
    parser.add_argument('--imgsz',      type=int,   default=None, help='覆盖输入图像尺寸')
    parser.add_argument('--lr',         type=float, default=None, help='覆盖初始学习率')
    parser.add_argument('--patience',   type=int,   default=None, help='覆盖早停耐心值')
    parser.add_argument('--device',     type=str,   default=None, help='覆盖训练设备')
    parser.add_argument('--workers',    type=int,   default=None, help='覆盖 dataloader workers')
    parser.add_argument('--pretrained', type=str,   default=None, help='覆盖预训练权重路径')

    args = parser.parse_args()

    # --- 断点续训 ---
    if args.resume:
        os.chdir(PROJECT_ROOT)
        print(f"\n{'='*64}")
        print(f"  🔄 恢复训练: {args.resume}")
        print(f"{'='*64}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    # --- 确定配置文件路径 ---
    if args.cfg:
        cfg_path = args.cfg
    elif args.config:
        cfg_path = CONFIG_SHORTCUTS[args.config]
    else:
        # 都没指定, 默认用 full
        cfg_path = CONFIG_SHORTCUTS['full']
        print(f"  ℹ️  未指定配置, 默认使用: {cfg_path}")

    # --- 加载配置 + 命令行覆盖 ---
    cfg = load_config(cfg_path)
    cfg = apply_cli_overrides(cfg, args)

    # --- 开始训练 ---
    train(cfg)


if __name__ == "__main__":
    main()
