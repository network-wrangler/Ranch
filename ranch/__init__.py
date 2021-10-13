__version__ = "0.0.1"

import os
from datetime import datetime

from .logger import RanchLogger, setupLogging
from .roadway import Roadway
from .sharedstreets import run_shst_extraction, read_shst_extraction, run_shst_match
from .osm import run_osmnx_extraction
from .utils import link_df_to_geojson, point_df_to_geojson
from .parameters import Parameters
from .transit import Transit

__all__ = [
    "RanchLogger",
    "setupLogging",
    "Roadway",
    "run_shst_extraction",
    "read_shst_extraction",
    "run_shst_match",
    "run_osmnx_extraction",
    "link_df_to_geojson",
    "point_df_to_geojson",
    "Parameters",
    "Transit",
]

setupLogging(
    log_filename=os.path.join(
        os.getcwd(),
        "ranch_{}.log".format(datetime.now().strftime("%Y_%m_%d__%H_%M_%S")),
    ),
    log_to_console=True,
)