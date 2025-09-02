# Legacy constants - use src.config.settings for new code
import warnings
warnings.warn("data/constants.py is deprecated. Use src.config.settings instead.", DeprecationWarning)

GLASS = 0
PAPER = 1
CARDBOARD = 2
PLASTIC = 3
METAL = 4
TRASH = 5

DIM1 = 384
DIM2 = 512