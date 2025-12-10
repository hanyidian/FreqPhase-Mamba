# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 确保sam2包可以被正确导入
import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到Python路径，确保可以导入sam2
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 初始化hydra配置模块
try:
    from hydra import initialize_config_module
    initialize_config_module("sam2_configs", version_base="1.2")
except Exception as e:
    print(f"Warning: Failed to initialize hydra config module: {e}")
