#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from modules.dlc_pytorch.models.detectors.base import (
    DETECTORS,
    BaseDetector,
)
from modules.dlc_pytorch.models.detectors.fasterRCNN import FasterRCNN
from modules.dlc_pytorch.models.detectors.ssd import SSDLite
