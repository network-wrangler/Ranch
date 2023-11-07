import os

from pyproj import CRS

from .logger import RanchLogger

standard_crs = CRS("epsg:4326")
alt_standard_crs = CRS("epsg:4269")


def get_base_dir(ranch_base_dir=os.getcwd()):
    d = ranch_base_dir
    for i in range(3):
        if "ranch" in os.listdir(d):
            RanchLogger.info("Lasso base directory set as: {}".format(d))
            return d
        d = os.path.dirname(d)

    msg = "Cannot find ranch base directory from {}, please input using keyword in parameters: `ranch_base_dir =` ".format(
        ranch_base_dir
    )
    RanchLogger.error(msg)
    raise (ValueError(msg))


class Parameters:
    """A class representing all the parameters"""

    def __init__(self, **kwargs):
        """
        Time period and category  splitting info
        """
        if "ranch_base_dir" in kwargs:
            self.base_dir = get_base_dir(ranch_base_dir=kwargs.get("ranch_base_dir"))
        else:
            self.base_dir = get_base_dir()

        if "settings_location" in kwargs:
            self.settings_location = kwargs.get("settings_location")
        else:
            self.settings_location = os.path.join(self.base_dir, "settings")

        self.scratch_location = os.path.join(self.base_dir, "tests", "scratch")

        self.data_interim_dir = os.path.join(self.base_dir, "data", "interim")

        self.highway_to_roadway_crosswalk_file = os.path.join(
            self.settings_location, "highway_to_roadway.csv"
        )

        self.network_type_file = os.path.join(
            self.settings_location, "network_type_indicator.csv"
        )

        self.county_taz_range = {
            "San Francisco": {"start": 1, "end": 9999},
            "San Mateo": {"start": 100001, "end": 109999},
            "Santa Clara": {"start": 200001, "end": 209999},
            "Alameda": {"start": 300001, "end": 309999},
            "Contra Costa": {"start": 400001, "end": 409999},
            "Solano": {"start": 500001, "end": 509999},
            "Napa": {"start": 600001, "end": 609999},
            "Sonoma": {"start": 700001, "end": 709999},
            "Marin": {"start": 800001, "end": 804999},
            "San Joaquin": {"start": 805001, "end": 809999},
            "external": {"start": 900001},
        }

        self.county_node_range = {
            "San Francisco": {"start": 1000000, "end": 1500000},
            "San Mateo": {"start": 1500000, "end": 2000000},
            "Santa Clara": {"start": 2000000, "end": 2500000},
            "Alameda": {"start": 2500000, "end": 3000000},
            "Contra Costa": {"start": 3000000, "end": 3500000},
            "Solano": {"start": 3500000, "end": 4000000},
            "Napa": {"start": 4000000, "end": 4500000},
            "Sonoma": {"start": 4500000, "end": 5000000},
            "Marin": {"start": 5000000, "end": 5250000},
            "San Joaquin": {"start": 5250000, "end": 5500000},
            "external": {"start": 10000000},
        }

        self.county_link_range = {
            "San Francisco": {"start": 1, "end": 1000000},
            "San Mateo": {"start": 1000000, "end": 2000000},
            "Santa Clara": {"start": 2000000, "end": 3000000},
            "Alameda": {"start": 3000000, "end": 4000000},
            "Contra Costa": {"start": 4000000, "end": 5000000},
            "Solano": {"start": 5000000, "end": 6000000},
            "Napa": {"start": 6000000, "end": 7000000},
            "Sonoma": {"start": 7000000, "end": 8000000},
            "Marin": {"start": 8000000, "end": 8500000},
            "San Joaquin": {"start": 8500000, "end": 9000000},
        }

        self.model_time_period = {
            "AM": {"start": 6, "end": 10},
            "MD": {"start": 10, "end": 15},
            "PM": {"start": 15, "end": 19},
            "NT": {"start": 19, "end": 3, "frequency_start": 19, "frequency_end": 22},
            "EA": {"start": 3, "end": 6, "frequency_start": 5, "frequency_end": 6},
        }

        self.model_time_enum_list = {
            "start_time": {
                "AM": "06:00:00",
                "MD": "10:00:00",
                "PM": "15:00:00",
                "NT": "19:00:00",
                "EA": "03:00:00",
            },
            "end_time": {
                "AM": "10:00:00",
                "MD": "15:00:00",
                "PM": "19:00:00",
                "NT": "03:00:00",
                "EA": "06:00:00",
            },
        }

        self.transit_routing_parameters = {
            "good_links_buffer_radius": 200,
            "non_good_links_penalty": 5,
            "bad_stops_buffer_radius": 100,
            "ft_penalty": {
                "residential": 2,
                "service": 3,
                "default": 1,
                "motorway": 0.9,
            },
        }

        self.standard_crs = CRS("epsg:4326")
        # do not convert if alt_standard_crs
        self.alt_standard_crs = CRS("epsg:4269")

        self.model_centroid_node_id_reserve = 3100

        self.__dict__.update(kwargs)
