"""Side-effect module: configures nnUNet_* environment variables on import.

Import this BEFORE any module that triggers a nnunetv2 import (directly or
transitively via totalsegmentator.nnunet) so that nnunetv2/paths.py reads
the variables on its first import and does not emit
"nnUNet_* is not defined" warnings.
"""

from totalsegmentator.config import setup_nnunet

setup_nnunet()
