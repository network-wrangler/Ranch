__version__ = "0.0.1"

import os
from datetime import datetime

from .logger import RanchLogger, setupLogging
#from .roadway import Roadway
from .sharedstreets import run_shst_extraction

__all__ = [
    "RanchLogger",
    "setupLogging",
    #"Roadway",
    "run_shst_extraction",
]

setupLogging(
    log_filename=os.path.join(
        os.getcwd(),
        "ranch_{}.log".format(datetime.now().strftime("%Y_%m_%d__%H_%M_%S")),
    ),
    log_to_console=True,
)