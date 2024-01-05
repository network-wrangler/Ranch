from __future__ import annotations

import multiprocessing
import os
import re
import time
from typing import Dict, Optional, Union
import copy

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import partridge as ptg
import peartree as pt
from partridge.config import default_config
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon

from .logger import RanchLogger
from .parameters import Parameters
from .roadway import Roadway
from .sharedstreets import read_shst_extraction, run_shst_match
from .utils import find_closest_node, geodesic_point_buffer, ox_graph

TRANSIT_UNQIUE_SHAPE_ID = ["agency_raw_name", "shape_id"]


def multiprocessing_shst_match(input_network_file):
    run_shst_match(
        input_network_file=input_network_file,
        input_unique_id=TRANSIT_UNQIUE_SHAPE_ID,
        custom_match_option="--follow-line-direction --tile-hierarchy=8",
    )


class Transit(object):
    """
    Representation of a Transit Network.
    Usage:
      import network_wrangler as wr
      stpaul = r'/home/jovyan/work/example/stpaul'
      tc=wr.TransitNetwork.read(path=stpaul)
    """

    # PK = primary key, FK = foreign key
    SHAPES_FOREIGN_KEY = "shape_model_node_id"
    STOPS_FOREIGN_KEY = "model_node_id"

    REQUIRED_FILES = [
        "agency.txt",
        "frequencies.txt",
        "routes.txt",
        "shapes.txt",
        "stop_times.txt",
        "stops.txt",
        "trips.txt",
    ]

    def __init__(
        self,
        gtfs_dir: str = None,
        feed: DotDict = None, 
        roadway_network: Roadway = None,
        parameters: Union[Parameters, dict] = {},
    ):
        """
        Constructor
        """
        self.gtfs_dir = gtfs_dir
        self.feed = feed
        self.roadway_network = roadway_network
        self.graph: nx.MultiDiGraph = None
        self.feed_path = None
        self.good_link_dict = dict()
        self.bad_stop_dict = dict()
        self.trip_osm_link_df: pd.DataFrame() = None
        self.trip_shst_link_df: pd.DataFrame() = None
        self.bus_trip_link_df: pd.DataFrame() = None
        self.bus_stops: pd.DataFrame() = None
        self.rail_trip_link_df: pd.DataFrame() = None
        self.rail_stops: pd.DataFrame() = None
        self.shortest_path_failed_shape_list: list = []
        self.unique_rail_links_gdf: gpd.GeoDataFrame() = None
        self.unique_rail_nodes_gdf: gpd.GeoDataFrame() = None

        # will have to change if want to alter them
        if type(parameters) is dict:
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = Parameters(**parameters.__dict__)
        else:
            msg = "Parameters should be a dict or instance of Parameters: found {} which is of type:{}".format(
                parameters, type(parameters)
            )
            RanchLogger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def load_all_gtfs_feeds(
        gtfs_dir: str,
        roadway_network: Roadway,
        parameters: Dict
    ):
        """
        read from every GTFS folders in the path and combine then into one
        """

        gtfs_agencies_list = os.listdir(gtfs_dir)
        if 'gtfs_shape_for_shst_match' in gtfs_agencies_list:
            gtfs_agencies_list.remove('gtfs_shape_for_shst_match')
        if 'osm_routing.geojson' in gtfs_agencies_list:
            gtfs_agencies_list.remove('osm_routing.geojson')
        
        feed = DotDict()
        feed.agency = pd.DataFrame()
        feed.routes = pd.DataFrame()
        feed.trips = pd.DataFrame()
        feed.stops = pd.DataFrame()
        feed.stop_times = pd.DataFrame()
        feed.shapes = pd.DataFrame()
        feed.fare_attributes = pd.DataFrame()
        feed.fare_rules = pd.DataFrame()
        feed.frequencies = pd.DataFrame()

        for agency in gtfs_agencies_list:

            agency_feed = Transit.get_representative_feed_from_gtfs(
                os.path.join(gtfs_dir, agency)
            )

            # gtfs cannot read fare tables for all agencies
            fare_attributes_df = pd.DataFrame()
            fare_rules_df = pd.DataFrame()

            if "fare_attributes.txt" in os.listdir(os.path.join(gtfs_dir, agency)):
        
                fare_attributes_df = pd.read_csv(os.path.join(gtfs_dir, agency, "fare_attributes.txt"),
                                         dtype = {"fare_id" : str})

                fare_attributes_df["agency_raw_name"] = agency
    
            if "fare_rules.txt" in os.listdir(os.path.join(gtfs_dir, agency)):
        
                fare_rules_df = pd.read_csv(
                    os.path.join(gtfs_dir, agency, "fare_rules.txt"),
                    dtype = {"fare_id" : str, "route_id" : str, "origin_id" : str, "destination_id" : str,
                         " route_id" : str, " origin_id" : str, " destination_id" : str,})

                fare_rules_df["agency_raw_name"] = agency

            feed.agency = pd.concat(
                [feed.agency, agency_feed.agency], sort=False, ignore_index=True
            )
            feed.routes = pd.concat(
                [feed.routes, agency_feed.routes], sort=False, ignore_index=True
            )
            feed.trips = pd.concat(
                [feed.trips, agency_feed.trips], sort=False, ignore_index=True
            )
            feed.stops = pd.concat(
                [feed.stops, agency_feed.stops], sort=False, ignore_index=True
            )
            feed.stop_times = pd.concat(
                [feed.stop_times, agency_feed.stop_times], sort=False, ignore_index=True
            )
            feed.shapes = pd.concat(
                [feed.shapes, agency_feed.shapes], sort=False, ignore_index=True
            )
            feed.fare_attributes = pd.concat(
                [feed.fare_attributes, fare_attributes_df],
                sort=False,
                ignore_index=True,
            )
            feed.fare_rules = pd.concat(
                [feed.fare_rules, fare_rules_df], sort=False, ignore_index=True
            )
            feed.frequencies = pd.concat(
                [feed.frequencies, agency_feed.frequencies],
                sort=False,
                ignore_index=True,
            )

        input_roadway_network = copy.deepcopy(roadway_network)

        transit_network = Transit(
            gtfs_dir = gtfs_dir,
            feed = feed,
            roadway_network = input_roadway_network,
            parameters=parameters
        )

        return transit_network

    def build_standard_transit_network(
        self,
        num_most_pattern: int = 1,
        multithread_shst_match: bool = True,
        multithread_shortest_path: bool = False
    ):
        """
        one-call method for transit, instead of calling each sub-module
        """
        self.get_representative_trip_for_route(num_most_pattern)
        self.snap_stop_to_node()

        route_df = self.feed.routes.copy()
        route_df = pd.merge(
            route_df,
            self.feed.trips[["agency_raw_name", "route_id"]].drop_duplicates(),
            how="inner",
            on=["agency_raw_name", "route_id"],
        )

        # check if there is bus routes
        bus_routes_df = route_df[route_df.route_type == 3].copy()
        
        if len(bus_routes_df) > 0:
            self.route_bus_trip(
                multithread_shst_match = multithread_shst_match, 
                multithread_shortest_path = multithread_shortest_path
            )
            if len(self.bus_trip_link_df) > 0:
                self.update_bus_stop_node()
            else:
                self.bus_trip_link_df = None
        else:
            RanchLogger.info('Feed does not have bus routes')

        # check if there is rail routes
        rail_routes_df = route_df[route_df.route_type != 3].copy()

        if len(rail_routes_df) > 0:
            self.route_rail_trip()
        else:
            RanchLogger.info('Feed does not have rail routes')

        self.create_shape_node_table()
        self.create_freq_table()

        # TODO: append rail links and nodes to roadway standard

    def get_representative_feed_from_gtfs(feed_path: str):
        """
        Read GTFS feed from folder and TransitNetwork object
        """

        RanchLogger.info("Excluding weekend-only services")

        # exclude weekend only services
        if "calendar_orig.txt" in os.listdir(feed_path):
            calendar_df = pd.read_csv(os.path.join(feed_path, "calendar.txt"))

        elif "calendar.txt" in os.listdir(feed_path):
            calendar_df = pd.read_csv(os.path.join(feed_path, "calendar.txt"))
            calendar_df.to_csv(
                os.path.join(feed_path, "calendar_orig.txt"), index=False, sep=","
            )

            calendar_df["weekdays"] = calendar_df.apply(
                lambda x: x.monday + x.tuesday + x.wednesday + x.thursday + x.friday,
                axis=1,
            )
            calendar_df = calendar_df[calendar_df.weekdays > 0]

            calendar_df.drop("weekdays", axis=1).to_csv(
                os.path.join(feed_path, "calendar.txt"), index=False, sep=","
            )

        RanchLogger.info(
            "Read and get representative transit feed from: {}".format(feed_path)
        )

        feed = pt.get_representative_feed(feed_path)

        agency_gtfs_name = os.path.basename(feed_path)

        validate_feed = Transit.validate_feed(feed, agency_gtfs_name)

        return validate_feed

    def validate_feed(feed, agency_gtfs_name):
        """
        add fields and validate feed
        """

        feed.routes["agency_raw_name"] = agency_gtfs_name

        feed.stops["agency_raw_name"] = agency_gtfs_name

        feed.trips["agency_raw_name"] = agency_gtfs_name

        # when 'direction_id' is not present, fill in default
        if "direction_id" not in feed.trips.columns:  # Marguerita
            feed.trips["direction_id"] = 0

        feed.trips["direction_id"].fillna(0, inplace=True)

        feed.shapes["agency_raw_name"] = agency_gtfs_name

        feed.stop_times["agency_raw_name"] = agency_gtfs_name

        feed.agency["agency_raw_name"] = agency_gtfs_name

        feed.fare_attributes["agency_raw_name"] = agency_gtfs_name

        feed.fare_rules["agency_raw_name"] = agency_gtfs_name

        # add agency_id in routes.txt if missing
        if "agency_id" not in feed.routes.columns:
            if "agency_id" in feed.agency.columns:
                feed.agency["agency_id"] = feed.agency.agency_id.iloc[0]

        # check if shapes are missing in GTFS
        if 'shape_id' not in feed.trips.columns:
            feed.trips['shape_id'] = np.nan
        
        # prep data for creating missing shape ids based on stop patterns
        # if the stop patterns are differnt, then assume the shapes are different
        stop_times_df = feed.stop_times.copy()
        trips_df = feed.trips.copy()

        trip_stops_df = (
            stop_times_df.groupby(
                ['agency_raw_name', 'trip_id']
            )['stop_id']
            .agg(list)
            .reset_index()
        )
        trip_stops_df.rename(
            columns = {'stop_id' : 'stop_pattern'},
            inplace = True
        )
        trip_stops_df['stop_pattern'] = trip_stops_df['stop_pattern'].str.join('-')

        trips_df = pd.merge(
            trips_df,
            trip_stops_df[['agency_raw_name', 'trip_id', 'stop_pattern']],
            how = 'left',
            on = ['agency_raw_name', 'trip_id']
        )

        if (len(feed.shapes) == 0) and (len(feed.trips[feed.trips.shape_id.notnull()]) == 0):  # ACE, CCTA, VINE

            RanchLogger.info("missing shapes.txt for {}".format(agency_gtfs_name))

            group_df = (
                trips_df.groupby(["agency_raw_name", "route_id", "direction_id", 'stop_pattern'])["trip_id"]
                .first()
                .reset_index()
                .drop("trip_id", axis=1)
            )

            group_df["shape_id"] = range(1, len(group_df) + 1)
            group_df["shape_id"] = group_df["shape_id"].astype(str)

            if "shape_id" in feed.trips.columns:
                feed.trips.drop("shape_id", axis=1, inplace=True)
                trips_df.drop("shape_id", axis=1, inplace=True)

            join_df = pd.merge(
                trips_df, group_df, how="left", on=["agency_raw_name", "route_id", "direction_id", 'stop_pattern']
            )

            join_df = pd.merge(
                feed.trips, 
                join_df[["agency_raw_name", "route_id", "direction_id", 'trip_id', 'shape_id']], 
                how="left", 
                on=["agency_raw_name", "route_id", "direction_id", 'trip_id']
            )

            feed.trips['shape_id'] = join_df['shape_id']

        if len(feed.trips[feed.trips.shape_id.isnull()]) > 0:
            RanchLogger.info("missing shape_ids in trips.txt for {}".format(agency_gtfs_name))

            trips_missing_shape_df = trips_df[trips_df.shape_id.isnull()].copy()

            group_df = (
                trips_missing_shape_df.groupby(['agency_raw_name', "route_id", "direction_id", 'stop_pattern'])["trip_id"]
                .first()
                .reset_index()
                .drop("trip_id", axis=1)
            )
            group_df["shape_id"] = range(1, len(group_df) + 1)
            group_df["shape_id"] = group_df["shape_id"].apply(
                lambda x: "psudo" + str(x)
            )

            trips_missing_shape_df = pd.merge(
                trips_missing_shape_df.drop("shape_id", axis=1),
                group_df,
                how="left",
                on=['agency_raw_name', "route_id", "direction_id", 'stop_pattern'],
            )

            new_shape_id_dict = dict(
                zip(trips_missing_shape_df.trip_id,trips_missing_shape_df.shape_id)
            )
            
            feed.trips['shape_id'] = np.where(
                feed.trips.shape_id.isnull(),
                feed.trips.trip_id.map(new_shape_id_dict),
                feed.trips.shape_id
            )

        return feed

    def get_representative_trip_for_route(self, num_most_pattern = 1):

        """
        get the representative trips for each route, by direction, tod

        Args:
            num_most_pattern: number of N most frequent routing pattern to keep, default to 1

        """

        RanchLogger.info(
            "Getting representative trip for each route by time of day and direction..."
        )

        # get the first stop of each trip to determine the time period for each trip
        # process time
        stop_times_df = self.feed.stop_times.copy()
        stop_times_df["arrival_h"] = pd.to_datetime(
            stop_times_df["arrival_time"], unit="s"
        ).dt.hour
        stop_times_df["arrival_m"] = pd.to_datetime(
            stop_times_df["arrival_time"], unit="s"
        ).dt.minute
        stop_times_df["departure_h"] = pd.to_datetime(
            stop_times_df["departure_time"], unit="s"
        ).dt.hour
        stop_times_df["departure_m"] = pd.to_datetime(
            stop_times_df["departure_time"], unit="s"
        ).dt.minute

        # according to the gtfs reference, the stop sequence does not have to be consecutive, but has to always increase
        # so we can get the fisrt stop by the smallest stop sequence on the trip
        stop_times_df.sort_values(
            by=["agency_raw_name", "trip_id", "stop_sequence"],
            ascending=True,
            inplace=True,
        )
        first_stop_df = stop_times_df.drop_duplicates(
            subset=["agency_raw_name", "trip_id"]
        )

        ## identify peak, offpeak trips, based on the arrival time of first stop
        trip_df = self.feed.trips.copy()
        trip_df = pd.merge(
            trip_df, first_stop_df, how="left", on=["agency_raw_name", "trip_id"]
        )

        model_time_period = self.parameters.model_time_period

        ## AM: 6-10am, MD: 10am-3pm, PM: 3-7pm, NT 7pm-3am, EA 3-6am
        trip_df["tod"] = np.where(
            (trip_df["arrival_h"] >= model_time_period.get("pk").get("start"))
            & (trip_df["arrival_h"] < model_time_period.get("pk").get("end")),
            "pk",
            np.where(
                (trip_df["arrival_h"] >= model_time_period.get("op").get("start"))
                & (trip_df["arrival_h"] < model_time_period.get("op").get("end")),
                "op",
                "other"
            ),
        )

        # get the most frequent trip for each route, by direction, by time of day
        ## trips share the same shape_id is considered being the same
        ## first get the trip count for each shape_id
        trip_freq_df = (
            trip_df.groupby(
                ["agency_raw_name", "route_id", "tod", "direction_id", "shape_id"]
            )["trip_id"]
            .count()
            .to_frame()
        )

        trip_freq_df.rename(
            columns = {"trip_id" : "trip_num_for_shape"}, 
            inplace = True
        )

        ## then choose the most frequent shape_id for each route
        # for frequency use the total number of trips
        def agg(x):
            m = x.shape_id.iloc[np.argmax(x.trip_num_for_shape.values)]
            return pd.Series({"trip_num": x.trip_num_for_shape.sum(), "shape_id": m})

        if num_most_pattern == 0:
            trip_freq_df = (
                trip_freq_df.reset_index()
                .groupby(["agency_raw_name", "route_id", "tod", "direction_id"])
                .apply(agg)
            )
        
        else:
            # keep the n most frequent pattern
        
            # calculate total number of trip per route by time and direction
            trip_num_df = trip_freq_df.groupby(
                ['agency_raw_name', 'route_id', 'tod', 'direction_id']
            )["trip_num_for_shape"].sum().reset_index()
            trip_num_df.rename(
                columns = {"trip_num_for_shape" : "trip_num_total"}, 
                inplace = True
            )
            # sort shape freuqncy table by number of trips per shape
            trip_freq_df = trip_freq_df.sort_values(
                by = ['agency_raw_name', 'route_id', 'tod', 'direction_id', "trip_num_for_shape"], 
                ascending = False
            )
            # keep the N most frequent shape
            trip_freq_df = trip_freq_df.groupby(
                ['agency_raw_name', 'route_id', 'tod', 'direction_id']
            ).head(num_most_pattern).reset_index()
        
            trip_freq_df = pd.merge(
                trip_freq_df,#.drop("trip_id", axis = 1), 
                trip_num_df, 
                how = "left", 
                on = ['agency_raw_name', 'route_id', 'tod', 'direction_id']
            )
            
            trip_freq_df["num_pattern"] = trip_freq_df.groupby(
                ['agency_raw_name', 'route_id', 'tod', 'direction_id']
            )["shape_id"].transform("count")
            trip_freq_df["trip_num_N_most"] = trip_freq_df.groupby(
                ['agency_raw_name', 'route_id', 'tod', 'direction_id']
            )["trip_num_for_shape"].transform("sum")

        # sort trips based on #stops on them
        trip_stops_df = (
            stop_times_df.groupby(
                ['agency_raw_name', 'trip_id']
            )['stop_sequence']
            .count()
            .to_frame()
            .reset_index()
        )
        trip_stops_df.rename(
            columns = {'stop_sequence' : 'number_of_stops'},
            inplace = True
        )
        
        # retain the complete trip info of the represent trip only
        trip_df = pd.merge(
            trip_df,
            trip_freq_df.reset_index(),
            how="inner",
            on=["agency_raw_name", "route_id", "tod", "direction_id", "shape_id"],
        )

        trip_df = pd.merge(
            trip_df,
            trip_stops_df[['agency_raw_name', 'trip_id', 'number_of_stops']],
            how="left",
            on=["agency_raw_name", 'trip_id']
        )

        # keep the trip with the most stops
        trip_df.sort_values(
            by = ['agency_raw_name', 'route_id', 'shape_id', 'number_of_stops'],
            ascending = [True, True, True, False]
        )
        
        trip_df.drop_duplicates(
            subset = ["agency_raw_name", "route_id", 'shape_id', "direction_id", "tod"],
            inplace = True
        )

        # keep trips within the model time
        trip_df = trip_df[trip_df['tod'].isin(model_time_period.keys())].copy()

        self.feed.trips = trip_df

    def snap_stop_to_node(self):

        """
        map gtfs stops to roadway nodes

        Parameters:
        ------------
        feed
        drive nodes

        return
        ------------
        stops with drive nodes id
        """
        RanchLogger.info('Snapping gtfs stops to roadway node...')

        # get rid of motorway nodes
        non_motorway_links_df = self.roadway_network.links_df[
            ~self.roadway_network.links_df.roadway.isin(["motorway", "motorway_link"])
        ].copy()

        node_candidates_for_stops_df = self.roadway_network.nodes_df[
            (
                self.roadway_network.nodes_df.shst_node_id.isin(
                    non_motorway_links_df.fromIntersectionId.tolist()
                    + non_motorway_links_df.toIntersectionId.tolist()
                )
                & (self.roadway_network.nodes_df.drive_access == 1)
            )
        ].copy()

        stop_df = self.feed.stops.copy()
        stop_df["geometry"] = [
            Point(xy) for xy in zip(stop_df["stop_lon"], stop_df["stop_lat"])
        ]
        stop_df = gpd.GeoDataFrame(
            stop_df, geometry=stop_df["geometry"], crs=self.parameters.standard_crs
        )

        stop_to_node_gdf = find_closest_node(
            stop_df,
            node_candidates_for_stops_df,
            unique_id=["agency_raw_name", "stop_id"],
        )

        stop_to_node_gdf.drop(["X", "Y"], axis=1, inplace=True)
        stop_df = pd.merge(
            stop_df, stop_to_node_gdf, how="left", on=["agency_raw_name", "stop_id"]
        )

        column_list = self.feed.stops.columns.values.tolist() + [
            "osm_node_id",
            "shst_node_id",
        ]
        self.feed.stops = stop_df[column_list]

    def route_bus_link_osmnx_from_start_to_end(
        self,
        good_links_buffer_radius: Optional[float] = None,
        ft_penalty: Optional[Dict] = None,
        non_good_links_penalty: Optional[float] = None,
    ):

        """
        Route each bus trip from the start stop to end stop
        """
        if good_links_buffer_radius:
            good_links_buffer_radius = good_links_buffer_radius
        else:
            good_links_buffer_radius = self.parameters.transit_routing_parameters.get(
                "good_links_buffer_radius"
            )

        if ft_penalty:
            ft_penalty = ft_penalty
        else:
            ft_penalty = self.parameters.transit_routing_parameters.get("ft_penalty")

        if non_good_links_penalty:
            non_good_links_penalty = non_good_links_penalty
        else:
            non_good_links_penalty = self.parameters.transit_routing_parameters.get(
                "non_good_links_penalty"
            )

        trip_df = self.feed.trips.copy()
        stop_df = self.feed.stops.copy()
        stop_time_df = self.feed.stop_times.copy()
        links_gdf = self.roadway_network.links_df.copy()
        nodes_gdf = self.roadway_network.nodes_df.copy()

        # append stop info to stop times table
        stop_time_df = pd.merge(
            stop_time_df, stop_df, how="left", on=["agency_raw_name", "stop_id"]
        )

        # for each stop, get which trips are using them
        stop_trip_df = stop_time_df.drop_duplicates(
            subset=["agency_raw_name", "trip_id", "stop_id"]
        )

        RanchLogger.info(
            "Routing bus on roadway network from start to end with osmnx..."
        )

        # get route type for trips, get bus trips
        trip_df = pd.merge(
            trip_df, self.feed.routes, how="left", on=["agency_raw_name", "route_id"]
        )
        bus_trip_df = trip_df[trip_df["route_type"] == 3]

        # for trips with same shape_id, keep the one with the most #stops
        # count # stops on each trip
        num_stops_on_trips_df = (
            stop_time_df.groupby(["agency_raw_name", "trip_id"])["stop_id"]
            .count()
            .reset_index()
            .rename(columns={"stop_id": "num_stop"})
        )

        bus_trip_df = pd.merge(
            bus_trip_df,
            num_stops_on_trips_df[["agency_raw_name", "trip_id", "num_stop"]],
            how="left",
            on=["agency_raw_name", "trip_id"],
        )

        bus_trip_df.sort_values(by=["num_stop"], inplace=True, ascending=False)

        # keep the trip with most stops
        bus_trip_df.drop_duplicates(
            subset=["agency_raw_name", "shape_id"],
            keep="first",
            inplace=True,
        )

        # get stops that are on bus trips only
        stops_on_bus_trips_df = stop_trip_df[
            (
                stop_trip_df["agency_raw_name"].isin(
                    bus_trip_df["agency_raw_name"].unique()
                )
            )
            & (stop_trip_df["trip_id"].isin(bus_trip_df["trip_id"].unique()))
        ].copy()
        stops_on_bus_trips_df.drop_duplicates(
            subset=["agency_raw_name", "stop_id"], inplace=True
        )

        RanchLogger.info("Setting good link dictionary")

        # set good link dictionary based on stops
        self.set_good_links(stops_on_bus_trips_df, good_links_buffer_radius)

        # output dataframe for osmnx success
        trip_osm_link_df = pd.DataFrame()

        # loop through bus trips
        for agency_raw_name in bus_trip_df.agency_raw_name.unique():
            trip_id_list = bus_trip_df[
                bus_trip_df["agency_raw_name"] == agency_raw_name
            ]["trip_id"].tolist()

            for trip_id in trip_id_list:

                RanchLogger.info("\tRouting agency {}, trip {}".format(agency_raw_name, trip_id))
                shape_id = bus_trip_df[
                    (bus_trip_df['agency_raw_name'] == agency_raw_name) &
                    (bus_trip_df['trip_id'] == trip_id)
                ]['shape_id'].iloc[0]

                # create bounding box from shape, xmin, xmax, ymin, ymax
                shape_df = self.feed.shapes[
                    (self.feed.shapes.shape_id == shape_id) &
                    (self.feed.shapes.agency_raw_name == agency_raw_name)
                ].copy()
                if len(shape_df) > 0:
                    shape_pt_lat_min = shape_df['shape_pt_lat'].min() - 0.05
                    shape_pt_lat_max = shape_df['shape_pt_lat'].max() + 0.05
                    shape_pt_lon_min = shape_df['shape_pt_lon'].min() - 0.05
                    shape_pt_lon_max = shape_df['shape_pt_lon'].max() + 0.05

                    lon_list = [shape_pt_lon_min, shape_pt_lon_min, shape_pt_lon_max, shape_pt_lon_max]
                    lat_list = [shape_pt_lat_min, shape_pt_lat_max, shape_pt_lat_max, shape_pt_lat_min]

                    shape_polygon = Polygon(zip(lon_list, lat_list))
                else:
                    stop_times_df = self.feed.stop_times[
                        (self.feed.stop_times.trip_id == trip_id) &
                        (self.feed.stop_times.agency_raw_name == agency_raw_name)
                    ].copy()
                    stop_times_df = pd.merge(
                        stop_times_df,
                        self.feed.stops[['stop_lat', 'stop_lon', 'stop_id', 'agency_raw_name']],
                        how = 'left',
                        on = ['stop_id', 'agency_raw_name']
                    )

                    stop_pt_lat_min = stop_times_df['stop_lat'].min() - 0.05
                    stop_pt_lat_max = stop_times_df['stop_lat'].max() + 0.05
                    stop_pt_lon_min = stop_times_df['stop_lon'].min() - 0.05
                    stop_pt_lon_max = stop_times_df['stop_lon'].max() + 0.05

                    lon_list = [stop_pt_lon_min, stop_pt_lon_min, stop_pt_lon_max, stop_pt_lon_max]
                    lat_list = [stop_pt_lat_min, stop_pt_lat_max, stop_pt_lat_max, stop_pt_lat_min]

                    shape_polygon = Polygon(zip(lon_list, lat_list))

                # get roadway links and nodes within bounding box
                # exclude cycleway and footway
                links_within_polygon_gdf = links_gdf[
                    (links_gdf.geometry.within(shape_polygon)) &
                    (links_gdf.drive_access == 1)
                ].copy()
                nodes_within_polygon_gdf = nodes_gdf[
                    nodes_gdf['shst_node_id'].isin(
                        links_within_polygon_gdf.fromIntersectionId.tolist() + 
                        links_within_polygon_gdf.toIntersectionId.tolist()
                    )
                ].copy()
                
                # get the stops on the trip
                trip_stops_df = stop_time_df[
                    (stop_time_df["trip_id"] == trip_id)
                    & (stop_time_df["agency_raw_name"] == agency_raw_name)
                ].copy()

                # get the good links from good links dictionary
                good_links_list = self.get_good_link_for_trip(trip_stops_df)

                # update link weights
                links_within_polygon_gdf["length_weighted"] = np.where(
                    links_within_polygon_gdf.shstReferenceId.isin(good_links_list),
                    links_within_polygon_gdf["length"],
                    links_within_polygon_gdf["length"] * non_good_links_penalty,
                )

                # apply ft penalty
                links_within_polygon_gdf["ft_penalty"] = links_within_polygon_gdf["roadway"].map(ft_penalty)
                links_within_polygon_gdf["ft_penalty"].fillna(ft_penalty["default"], inplace=True)

                links_within_polygon_gdf["length_weighted"] = (
                    links_within_polygon_gdf["length_weighted"] * links_within_polygon_gdf["ft_penalty"]
                )

                # update graph
                G_trip = ox_graph(nodes_within_polygon_gdf, links_within_polygon_gdf)

                trip_stops_df.sort_values(by=["stop_sequence"], inplace=True)

                # from first stop node OSM id
                closest_node_to_first_stop = int(trip_stops_df.osm_node_id.iloc[0])

                # to last stop node OSM id
                closest_node_to_last_stop = int(trip_stops_df.osm_node_id.iloc[-1])

                RanchLogger.debug("Routing trip {} from stop {}, osm node {} to stop {} osm node {}".format(
                    trip_stops_df.trip_id.unique(),
                    trip_stops_df.stop_id.iloc[0],
                    closest_node_to_first_stop,
                    trip_stops_df.stop_id.iloc[-1],
                    closest_node_to_last_stop
                ))

                path_osm_link_df = Transit.get_link_path_between_nodes(
                    G_trip,
                    closest_node_to_first_stop,
                    closest_node_to_last_stop,
                    weight_field="length_weighted",
                )

                if type(path_osm_link_df) == str:
                    self.shortest_path_failed_shape_list.append(
                        '{}_{}'.format(
                            agency_raw_name, 
                            shape_id
                        )
                    )
                    continue

                path_osm_link_df['trip_id'] = trip_id
                path_osm_link_df['agency_raw_name'] = agency_raw_name

                trip_osm_link_df = trip_osm_link_df.append(
                    path_osm_link_df, ignore_index=True, sort=False
                )

        # after routing all trips, join with the links
        trip_osm_link_df = pd.merge(
            trip_osm_link_df,
            trip_df[["agency_raw_name", "trip_id", "shape_id"]],
            how="left",
            on=["agency_raw_name", "trip_id"],
        )

        trip_osm_link_df = pd.merge(
            trip_osm_link_df,
            self.roadway_network.links_df[
                ["u", "v", "wayId", "shstReferenceId", "shstGeometryId", "geometry"]
            ].drop_duplicates(subset=["u", "v"]),
            how="left",
            on=["u", "v"],
        )

        self.trip_osm_link_df = trip_osm_link_df

    def get_link_path_between_nodes(G, from_node, to_node, weight_field):
        """
        return the complete links from start node to end node using networkx routing
        """

        # routing btw from and to nodes, return the list of nodes
        try:
            node_osmid_list = nx.shortest_path(
                G, 
                from_node, 
                to_node, 
                weight_field
            )
        except:
            return 'No Path Found'

        # circular route
        if from_node == to_node:
            osm_link_df = pd.DataFrame(
                {
                    "u": [from_node],
                    "v": [from_node],
                },
            )

            return osm_link_df

        # get the links
        if len(node_osmid_list) > 1:
            osm_link_df = pd.DataFrame(
                {
                    "u": node_osmid_list[: len(node_osmid_list) - 1],
                    "v": node_osmid_list[1 : len(node_osmid_list)],
                },
            )

            return osm_link_df
        else:
            return pd.DataFrame()

    def set_good_links(self, stops, good_links_buffer_radius):

        """
        for each bus stop, get the list of good link IDs and store them to a dict
        """

        # get non-motorway links
        # maybe not a good idea?
        non_motorway_links_df = self.roadway_network.links_df[
            ~self.roadway_network.links_df.roadway.isin(["motorway", "motorway_link"])
        ].copy()

        # get drive links
        drive_links_df = self.roadway_network.links_df[
            self.roadway_network.links_df.drive_access == 1
        ].copy()

        # get the links that are within stop buffer
        stop_good_link_df = Transit.links_within_stop_buffer(
            # non_motorway_links_df,
            drive_links_df,
            stops,
            buffer_radius=good_links_buffer_radius,
        )

        good_link_dict = (
            stop_good_link_df.groupby(["agency_raw_name", "stop_id"])["shstReferenceId"]
            .apply(list)
            .to_dict()
        )

        self.good_link_dict = good_link_dict

    def get_good_link_for_trip(self, trip_stops):
        """
        for input stop IDs return a list of the good link IDs
        """

        link_shstReferenceId_list = []
        for agency_raw_name in trip_stops["agency_raw_name"].unique():
            stop_id_list = trip_stops[trip_stops["agency_raw_name"] == agency_raw_name][
                "stop_id"
            ].unique()
            for stop_id in stop_id_list:
                if self.good_link_dict.get((agency_raw_name, stop_id)):
                    link_shstReferenceId_list += self.good_link_dict.get(
                        (agency_raw_name, stop_id)
                    )

        return link_shstReferenceId_list

    def links_within_stop_buffer(drive_link_df, stops, buffer_radius):
        """
        find the links that are within buffer of nodes
        """

        stop_buffer_df = stops.copy()
        stop_buffer_df["geometry"] = stop_buffer_df.apply(
            lambda x: geodesic_point_buffer(x.stop_lat, x.stop_lon, buffer_radius),
            axis=1,
        )

        stop_buffer_df = gpd.GeoDataFrame(
            stop_buffer_df, geometry=stop_buffer_df["geometry"], crs=drive_link_df.crs
        )

        stop_buffer_link_df = gpd.sjoin(
            drive_link_df,
            stop_buffer_df[["geometry", "agency_raw_name","stop_id"]], 
            how = "left", 
            predicate = "intersects"
        )
        
        stop_buffer_link_df = stop_buffer_link_df[
            stop_buffer_link_df.stop_id.notnull()
        ]
        
        return stop_buffer_link_df

    def set_bad_stops(self, bad_stop_buffer_radius: Optional[float] = None):
        """
        for each stop location, check if the routed route is within 50 meters
        """

        if bad_stop_buffer_radius:
            bad_stop_buffer_radius = bad_stop_buffer_radius
        else:
            bad_stop_buffer_radius = self.parameters.transit_routing_parameters.get(
                "bad_stops_buffer_radius"
            )

        trip_df = self.feed.trips.copy()
        stop_df = self.feed.stops.copy()
        stop_time_df = self.feed.stop_times.copy()

        trip_osm_link_gdf = gpd.GeoDataFrame(
            self.trip_osm_link_df,
            geometry=self.trip_osm_link_df["geometry"],
            crs=self.roadway_network.links_df.crs,
        )

        # get chained stops on a trip
        chained_stop_df = stop_time_df[
            stop_time_df["trip_id"].isin(trip_df.trip_id.tolist())
        ].copy()
        chained_stop_df = pd.merge(
            chained_stop_df, stop_df, how="left", on=["agency_raw_name", "stop_id"]
        )

        dict_stop_far_from_trip = {}

        # loop through agency and bus trips
        for agency_raw_name in trip_osm_link_gdf["agency_raw_name"].unique():

            trip_id_list = trip_osm_link_gdf[
                trip_osm_link_gdf["agency_raw_name"] == agency_raw_name
            ]["trip_id"].unique()

            for trip_id in trip_id_list:

                # get the stops on the trip
                trip_stop_df = chained_stop_df[
                    (chained_stop_df["trip_id"] == trip_id)
                    & (chained_stop_df["agency_raw_name"] == agency_raw_name)
                ].copy()

                dict_stop_far_from_trip[(agency_raw_name, trip_id)] = []

                for i in range(len(trip_stop_df)):

                    single_stop_df = trip_stop_df.iloc[i : i + 1].copy()

                    trip_links_gdf = trip_osm_link_gdf[
                        (trip_osm_link_gdf["trip_id"] == trip_id)
                        & (trip_osm_link_gdf["agency_raw_name"] == agency_raw_name)
                    ].copy()

                    # get the links that are within stop buffer
                    links_list = Transit.links_within_stop_buffer(
                        trip_links_gdf,
                        single_stop_df,
                        buffer_radius=bad_stop_buffer_radius,
                    )

                    if len(links_list) == 0:
                        dict_stop_far_from_trip[
                            (agency_raw_name, trip_id)
                        ] += single_stop_df["stop_id"].tolist()

        self.bad_stop_dict = dict_stop_far_from_trip

    def route_bus_link_osmnx_between_stops(
        self,
        good_links_buffer_radius: Optional[float] = None,
        ft_penalty: Optional[Dict] = None,
        non_good_links_penalty: Optional[float] = None,
    ):
        """
        route bus trips between the bad stops
        """

        if good_links_buffer_radius:
            good_links_buffer_radius = good_links_buffer_radius
        else:
            good_links_buffer_radius = self.parameters.transit_routing_parameters.get(
                "good_links_buffer_radius"
            )

        if ft_penalty:
            ft_penalty = ft_penalty
        else:
            ft_penalty = self.parameters.transit_routing_parameters.get("ft_penalty")

        if non_good_links_penalty:
            non_good_links_penalty = non_good_links_penalty
        else:
            non_good_links_penalty = self.parameters.transit_routing_parameters.get(
                "non_good_links_penalty"
            )

        trip_df = self.feed.trips.copy()
        stop_df = self.feed.stops.copy()
        stop_time_df = self.feed.stop_times.copy()
        links_gdf = self.roadway_network.links_df.copy()
        nodes_gdf = self.roadway_network.nodes_df.copy()

        # append stop info to stop times table
        stop_time_df = pd.merge(
            stop_time_df, stop_df, how="left", on=["agency_raw_name", "stop_id"]
        )

        RanchLogger.info('Routing bus on roadway network from stop to stop with osmnx...')

        bus_trip_df = pd.merge(
            trip_df,
            self.trip_osm_link_df[['agency_raw_name', 'trip_id']].drop_duplicates(),
            how = 'inner',
            on = ['agency_raw_name', 'trip_id']
        )

        # output dataframe for osmnx success
        trip_osm_link_df = pd.DataFrame()

        # loop through for bus trips
        for agency_raw_name in bus_trip_df.agency_raw_name.unique():
            trip_id_list = bus_trip_df[
                bus_trip_df["agency_raw_name"] == agency_raw_name
            ]["trip_id"].tolist()

            for trip_id in trip_id_list:

                shape_id = bus_trip_df[
                    (bus_trip_df["agency_raw_name"] == agency_raw_name)
                    & (bus_trip_df["trip_id"] == trip_id)
                ]["shape_id"].iloc[0]

                # create bounding box from shape, xmin, xmax, ymin, ymax
                shape_df = self.feed.shapes[
                    (self.feed.shapes.shape_id == shape_id) &
                    (self.feed.shapes.agency_raw_name == agency_raw_name)
                ].copy()
                if len(shape_df) > 0:
                    shape_pt_lat_min = shape_df['shape_pt_lat'].min() - 0.05
                    shape_pt_lat_max = shape_df['shape_pt_lat'].max() + 0.05
                    shape_pt_lon_min = shape_df['shape_pt_lon'].min() - 0.05
                    shape_pt_lon_max = shape_df['shape_pt_lon'].max() + 0.05

                    lon_list = [shape_pt_lon_min, shape_pt_lon_min, shape_pt_lon_max, shape_pt_lon_max]
                    lat_list = [shape_pt_lat_min, shape_pt_lat_max, shape_pt_lat_max, shape_pt_lat_min]

                    shape_polygon = Polygon(zip(lon_list, lat_list))
                else:
                    stop_times_df = self.feed.stop_times[
                        (self.feed.stop_times.trip_id == trip_id) &
                        (self.feed.stop_times.agency_raw_name == agency_raw_name)
                    ].copy()
                    stop_times_df = pd.merge(
                        stop_times_df,
                        self.feed.stops[['stop_lat', 'stop_lon', 'stop_id', 'agency_raw_name']],
                        how = 'left',
                        on = ['stop_id', 'agency_raw_name']
                    )

                    stop_pt_lat_min = stop_times_df['stop_lat'].min() - 0.05
                    stop_pt_lat_max = stop_times_df['stop_lat'].max() + 0.05
                    stop_pt_lon_min = stop_times_df['stop_lon'].min() - 0.05
                    stop_pt_lon_max = stop_times_df['stop_lon'].max() + 0.05

                    lon_list = [stop_pt_lon_min, stop_pt_lon_min, stop_pt_lon_max, stop_pt_lon_max]
                    lat_list = [stop_pt_lat_min, stop_pt_lat_max, stop_pt_lat_max, stop_pt_lat_min]

                    shape_polygon = Polygon(zip(lon_list, lat_list))

                # get roadway links and nodes within bounding box
                # exclude cycleway and footway
                links_within_polygon_gdf = links_gdf[
                    (links_gdf.geometry.within(shape_polygon)) &
                    (links_gdf.drive_access == 1)
                ].copy()
                nodes_within_polygon_gdf = nodes_gdf[
                    nodes_gdf['shst_node_id'].isin(
                        links_within_polygon_gdf.fromIntersectionId.tolist() + 
                        links_within_polygon_gdf.toIntersectionId.tolist()
                    )
                ].copy()

                # get the stops on the trip
                trip_stops_df = stop_time_df[
                    (stop_time_df["trip_id"] == trip_id)
                    & (stop_time_df["agency_raw_name"] == agency_raw_name)
                ].copy()

                # get the links that are within stop buffer
                good_links_list = self.get_good_link_for_trip(trip_stops_df)

                # update link weights
                links_within_polygon_gdf["length_weighted"] = np.where(
                    links_within_polygon_gdf.shstReferenceId.isin(good_links_list),
                    links_within_polygon_gdf["length"],
                    links_within_polygon_gdf["length"] * non_good_links_penalty,
                )

                # apply ft penalty
                links_within_polygon_gdf["ft_penalty"] = links_within_polygon_gdf["roadway"].map(ft_penalty)
                links_within_polygon_gdf["ft_penalty"].fillna(ft_penalty["default"], inplace=True)

                links_within_polygon_gdf["length_weighted"] = (
                    links_within_polygon_gdf["length_weighted"] * links_within_polygon_gdf["ft_penalty"]
                )

                # update graph

                G_trip = ox_graph(nodes_within_polygon_gdf, links_within_polygon_gdf)

                trip_stops_df.sort_values(by=["stop_sequence"], inplace=True)

                route_by_stop_df = trip_stops_df[
                    trip_stops_df.stop_id.isin(
                        self.bad_stop_dict.get((agency_raw_name, trip_id))
                    )
                ].copy()

                route_by_stop_df = pd.concat(
                    [trip_stops_df.iloc[0:1],
                    route_by_stop_df, 
                    trip_stops_df.iloc[-1:]],
                    sort = False,
                    ignore_index = True)
                
                route_by_stop_df.drop_duplicates(subset = ["stop_id"], inplace = True)

                RanchLogger.info("\tRouting agency {}, trip {}, between stops".format(
                    trip_stops_df.agency_raw_name.unique(),
                    trip_stops_df.trip_id.unique()
                    )
                )
                
                for s in range(len(route_by_stop_df)-1):
                    # from stop node OSM id
                    closest_node_to_first_stop = int(
                        route_by_stop_df.osm_node_id.iloc[s]
                    )

                    # to stop node OSM id
                    closest_node_to_last_stop = int(
                        route_by_stop_df.osm_node_id.iloc[s + 1]
                    )

                    # osmnx routing btw from and to stops, return the list of nodes
                    RanchLogger.debug("\tRouting trip {} from stop {} node {}, to stop {} node {}".format(
                        trip_stops_df.trip_id.unique(),
                        route_by_stop_df.stop_id.iloc[s],
                        route_by_stop_df.osm_node_id.iloc[s],
                        route_by_stop_df.stop_id.iloc[s+1],
                        route_by_stop_df.osm_node_id.iloc[s+1],
                    ))

                    if closest_node_to_first_stop == closest_node_to_last_stop:
                        continue

                    path_osm_link_df = Transit.get_link_path_between_nodes(
                        G_trip,
                        closest_node_to_first_stop,
                        closest_node_to_last_stop,
                        weight_field="length_weighted",
                    )

                    if type(path_osm_link_df) == str:
                        self.shortest_path_failed_shape_list.append(
                            '{}_{}'.format(
                                agency_raw_name, 
                                shape_id
                            )
                        )
                        break
                    
                    path_osm_link_df['trip_id'] = trip_id
                    path_osm_link_df['agency_raw_name'] = agency_raw_name
                    path_osm_link_df['shape_id'] = shape_id

                    trip_osm_link_df = trip_osm_link_df.append(
                        path_osm_link_df, ignore_index=True, sort=False
                    )

        if len(trip_osm_link_df) == 0:
            self.trip_osm_link_df = trip_osm_link_df
            return
        
        # after routing all trips, join with the links
        trip_osm_link_df = pd.merge(
            trip_osm_link_df.drop("trip_id", axis=1),
            trip_df[["agency_raw_name", "trip_id", "shape_id"]],
            how="left",
            on=["agency_raw_name", "shape_id"],
        )

        # remove trips that failed to be routed with shortest path
        trip_osm_link_df['agency_shape_id'] = trip_osm_link_df['agency_raw_name'] + "_" + trip_osm_link_df['shape_id'].astype(str)
        
        trip_osm_link_df = trip_osm_link_df[
            ~(trip_osm_link_df['agency_shape_id'].isin(
                self.shortest_path_failed_shape_list
            ))
        ]
        trip_osm_link_df.drop(['agency_shape_id'], axis = 1, inplace = True)

        trip_osm_link_df = pd.merge(
            trip_osm_link_df,
            self.roadway_network.links_df[
                [
                    "u",
                    "v",
                    "fromIntersectionId",
                    "toIntersectionId",
                    "wayId",
                    "shstReferenceId",
                    "shstGeometryId",
                    "geometry",
                ]
            ].drop_duplicates(subset=["u", "v"]),
            how="left",
            on=["u", "v"],
        )

        self.trip_osm_link_df = trip_osm_link_df

    def route_gtfs_using_shortest_path(self):
        """
        method that calls methods for routing using shortest path
        """
        RanchLogger.info("Route bus trips using shortest path")
        self.route_bus_link_osmnx_from_start_to_end()
        self.set_bad_stops()
        self.route_bus_link_osmnx_between_stops()

    def match_gtfs_shapes_to_shst(
        self,
        path: Optional[str] = None,
        multithread_shst_match: bool=True
    ):
        """
        1. call the method that matches gtfs shapes to shst,
        2. clean up the match result
        """
        if path:
            path = path
        else:
            path = os.path.join(self.gtfs_dir, 'gtfs_shape_for_shst_match')
        
        if not os.path.exists(path):
            os.makedirs(path)

        RanchLogger.info("Route bus trips using shst match")

        self._match_gtfs_shapes_to_shst(path, multithread_shst_match)

        if len(self.trip_shst_link_df) > 0:
            self.trip_shst_link_df = pd.merge(
                self.trip_shst_link_df.drop(["geometry"], axis=1),
                self.roadway_network.links_df[
                    [
                        "shstReferenceId",
                        "wayId",
                        "u",
                        "v",
                        "fromIntersectionId",
                        "toIntersectionId",
                        "geometry",
                    ]
                ],
                how="left",
                on="shstReferenceId",
            )

            # if GTFS has transit that is outside of the network region
            # there will be no roadway link joined, drop those records
            self.trip_shst_link_df = self.trip_shst_link_df[
                (self.trip_shst_link_df["u"].notnull()) &
                (self.trip_shst_link_df["v"].notnull())
            ]

            if len(self.trip_shst_link_df) > 0:

                self.trip_shst_link_df["u"] = (
                    self.trip_shst_link_df["u"].fillna(0).astype(np.int64)
                )
                self.trip_shst_link_df["v"] = (
                    self.trip_shst_link_df["v"].fillna(0).astype(np.int64)
                )

                trip_shst_link_df = self.trip_shst_link_df.copy()

                trip_shst_link_df["next_agency_raw_name"] = (
                    trip_shst_link_df["agency_raw_name"]
                    .iloc[1:]
                    .append(pd.Series(trip_shst_link_df["agency_raw_name"].iloc[-1]))
                    .reset_index(drop=True)
                )

                trip_shst_link_df["next_shape_id"] = (
                    trip_shst_link_df["shape_id"]
                    .iloc[1:]
                    .append(pd.Series(trip_shst_link_df["shape_id"].iloc[-1]))
                    .reset_index(drop=True)
                )

                trip_shst_link_df["next_u"] = (
                    trip_shst_link_df["u"]
                    .iloc[1:]
                    .append(pd.Series(trip_shst_link_df["v"].iloc[-1]))
                    .reset_index(drop=True)
                )

                incomplete_trip_shst_link_df = trip_shst_link_df[
                    (
                        trip_shst_link_df.agency_raw_name
                        == trip_shst_link_df.next_agency_raw_name
                    )
                    & (trip_shst_link_df.shape_id == trip_shst_link_df.next_shape_id)
                    & (trip_shst_link_df.v != trip_shst_link_df.next_u)
                ].copy()

                incomplete_trip_shst_link_df["agency_shape_id"] = (
                    incomplete_trip_shst_link_df["agency_raw_name"]
                    + "_"
                    + incomplete_trip_shst_link_df["shape_id"].astype(str)
                )

                self.trip_shst_link_df["agency_shape_id"] = (
                    self.trip_shst_link_df["agency_raw_name"]
                    + "_"
                    + self.trip_shst_link_df["shape_id"].astype(str)
                )

                self.trip_shst_link_df = self.trip_shst_link_df[
                    ~(
                        self.trip_shst_link_df.agency_shape_id.isin(
                            incomplete_trip_shst_link_df.agency_shape_id.unique()
                        )
                    )
                ]

                self.trip_shst_link_df = pd.merge(
                    self.trip_shst_link_df,
                    self.feed.trips[["agency_raw_name", "trip_id", "shape_id"]],
                    how="left",
                    on=["agency_raw_name", "shape_id"],
                )

    def _match_gtfs_shapes_to_shst(
        self,
        path: str,
        multithread_shst_match: bool=True
    ):
        """
        1. write out geojson from gtfs shapes for shst match,
        2. run the actual match method,
        3. read the match result
        """

        # exclude rail shapes
        if "route_type" in self.feed.trips.columns:
            bus_trips_df = self.feed.trips[self.feed.trips.route_type == 3].copy()
        else:
            trips_df = pd.merge(
                self.feed.trips,
                self.feed.routes,
                how="left",
                on=["agency_raw_name", "route_id"],
            )
            bus_trips_df = trips_df[trips_df.route_type == 3].copy()

        shapes_df = self.feed.shapes.copy()
        shapes_df = pd.merge(
            shapes_df,
            bus_trips_df[["agency_raw_name", "shape_id"]].drop_duplicates(),
            how="inner",
            on=["agency_raw_name", "shape_id"],
        )

        shapes_df = gpd.GeoDataFrame(
            shapes_df,
            geometry=gpd.points_from_xy(
                shapes_df["shape_pt_lon"], shapes_df["shape_pt_lat"]
            ),
            crs=self.roadway_network.links_df.crs,
        )

        lines_from_shapes_df = (
            shapes_df.groupby(["agency_raw_name", "shape_id"])["geometry"]
            .apply(lambda x: LineString(x.tolist()))
            .reset_index()
        )
        lines_from_shapes_df = gpd.GeoDataFrame(
            lines_from_shapes_df, geometry="geometry"
        )

        match_input_file_list = []
        for index, row in lines_from_shapes_df.iterrows():
            agency_raw_name = row["agency_raw_name"]
            shape_id = row["shape_id"]
            row = row.to_frame().T
            row_gdf = gpd.GeoDataFrame(
                row, geometry=row["geometry"], crs=self.parameters.standard_crs
            )
            row_gdf.to_file(
                os.path.join(path, "lines_from_shapes_{}_{}.geojson".format(agency_raw_name.replace(' ', ''), shape_id.replace(' ', ''))),
                driver='GeoJSON'
            )

            match_input_file_list.append(
                os.path.join(path, "lines_from_shapes_{}_{}.geojson".format(agency_raw_name.replace(' ', ''), shape_id.replace(' ', '')))
            )
        
        if multithread_shst_match:
            match_pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

            match_pool.map(multiprocessing_shst_match, match_input_file_list)

        else:
            for match_input_file in match_input_file_list:
                run_shst_match(
                    input_network_file = match_input_file,
                    input_unique_id = TRANSIT_UNQIUE_SHAPE_ID,
                    output_dir = path,
                    custom_match_option = '--follow-line-direction --tile-hierarchy=8'
                )
        
        trip_shst_link_df = read_shst_extraction(path, "*.matched.geojson")

        if len(trip_shst_link_df) > 0:
            trip_shst_link_df.rename(
                columns={
                    "pp_agency_raw_name": "agency_raw_name",
                    "pp_shape_id": "shape_id",
                },
                inplace=True,
            )

        self.trip_shst_link_df = trip_shst_link_df

    def route_bus_trip(
        self,
        multithread_shst_match: bool = True,
        multithread_shortest_path: bool = False
    ):
        """
        method that routes bus trips
        1. check if already routed by shortest path, if not, call shortest path
        2. check if already routed by shst, if not, call shst match
        3. combine the two routing results
        """

        # route using shortest path
        if self.trip_osm_link_df is None:
            self.route_gtfs_using_shortest_path()
            trip_osm_link_gdf = gpd.GeoDataFrame(
                self.trip_osm_link_df, crs=self.roadway_network.links_df.crs
            )

            if len(self.trip_osm_link_df) > 0:
                trip_osm_link_gdf.drop('wayId',axis=1).to_file(
                    os.path.join(self.gtfs_dir, 'osm_routing.geojson'),
                    driver = 'GeoJSON'
                )

        # route using shts match
        if self.trip_shst_link_df is None:
            self.match_gtfs_shapes_to_shst(multithread_shst_match = multithread_shst_match)

        # get route type for trips, get bus trips
        trip_df = pd.merge(
            self.feed.trips,
            self.feed.routes,
            how="left",
            on=["agency_raw_name", "route_id"],
        )
        bus_trip_df = trip_df[trip_df["route_type"] == 3].copy()

        bus_trip_df["agency_shape_id"] = (
            bus_trip_df["agency_raw_name"] + "_" + bus_trip_df["shape_id"].astype(str)
        )

        RanchLogger.info(
            "representative trips include {} bus shapes, {} bus trips".format(
                bus_trip_df.agency_shape_id.nunique(), len(bus_trip_df)
            )
        )

        trip_shst_link_df = self.trip_shst_link_df.copy()

        if len(trip_shst_link_df) > 0:
            # keep bus shapes from shst match
            trip_shst_link_df = trip_shst_link_df[
                (
                    trip_shst_link_df.agency_shape_id.isin(
                        bus_trip_df.agency_shape_id.unique()
                    )
                )
            ]

            trip_shst_link_df["method"] = "shst match"

            RanchLogger.info(
                "shst matched {} bus shapes, {} bus trips".format(
                    trip_shst_link_df.agency_shape_id.nunique(),
                    len(trip_shst_link_df.groupby(["agency_raw_name", "trip_id"]).count()),
                )
            )
        else:
            RanchLogger.info(
                "shst matched 0 bus shapes, 0 bus trips"
            )

        if len(self.trip_osm_link_df) == 0:
            RanchLogger.info('in the remaining gtfs shapes, shorteset path method matched 0 bus shapes')
            self.bus_trip_link_df = trip_shst_link_df
            return

        # keep bus shapes in shortest path routing that are not in shst match
        trip_osm_link_df = self.trip_osm_link_df.copy()
        trip_osm_link_df["agency_shape_id"] = (
            trip_osm_link_df["agency_raw_name"]
            + "_"
            + trip_osm_link_df["shape_id"].astype(str)
        )

        if len(trip_shst_link_df) > 0:

            trip_osm_link_df = trip_osm_link_df[
                ~(
                    trip_osm_link_df.agency_shape_id.isin(
                        trip_shst_link_df.agency_shape_id.unique()
                    )
                )
            ]

        trip_osm_link_df['method'] = 'shortest path'
        
        RanchLogger.info("in the remaining gtfs shapes, shortest path method matched {} bus shapes, {} bus trips".format(
            trip_osm_link_df.agency_shape_id.nunique(),
            len(trip_osm_link_df.groupby(['agency_raw_name', 'trip_id']).count()))
        )

        if len(trip_shst_link_df) > 0:
            bus_trip_link_df = pd.concat(
                [trip_osm_link_df, trip_shst_link_df[trip_osm_link_df.columns]],
                sort=False,
                ignore_index=True,
            )
        else:
            bus_trip_link_df = trip_osm_link_df

        self.bus_trip_link_df = bus_trip_link_df

    def update_bus_stop_node(self):
        """
        after routing buses, update bus stop nodes
        match stops to the nodes that are on the bus links
        """

        RanchLogger.info("Updating stop node matching")

        # first for each stop, get what trips use them
        stop_time_df = self.feed.stop_times.copy()
        stop_df = self.feed.stops.copy()

        stop_df['geometry'] = [Point(xy) for xy in zip(stop_df['stop_lon'], stop_df['stop_lat'])]
        stop_df = gpd.GeoDataFrame(
            stop_df,
            geometry = stop_df['geometry'],
            crs = self.parameters.standard_crs
        )

        ## append stop info to stop times table
        stop_time_df = pd.merge(
            stop_time_df, stop_df, how="left", on=["agency_raw_name", "stop_id"]
        )

        stop_time_df = gpd.GeoDataFrame(
            stop_time_df,
            geometry = stop_time_df['geometry'],
            crs = self.parameters.standard_crs
        )

        # get route type for trips, get bus trips
        trip_df = pd.merge(
            self.feed.trips,
            self.feed.routes,
            how="left",
            on=["agency_raw_name", "route_id"],
        )
        bus_trip_df = trip_df[trip_df["route_type"] == 3]

        stop_time_df = pd.merge(
            stop_time_df,
            bus_trip_df[["agency_raw_name", "trip_id"]],
            how="inner",
            on=["agency_raw_name", "trip_id"],
        )

        stop_to_node_df = pd.DataFrame()

        for agency_raw_name in self.bus_trip_link_df.agency_raw_name.unique():
            agency_trip_link_df = self.bus_trip_link_df[
                self.bus_trip_link_df.agency_raw_name == agency_raw_name
            ].copy()

            for trip_id in self.bus_trip_link_df.trip_id.unique():
                
                shape_id = self.feed.trips[self.feed.trips.trip_id == trip_id].shape_id.iloc[0]

                if '{}_{}'.format(agency_raw_name, shape_id) in self.shortest_path_failed_shape_list:
                    continue

                trip_stop_df = stop_time_df[
                    (stop_time_df.trip_id == trip_id)
                    & (stop_time_df.agency_raw_name == agency_raw_name)
                ].copy()

                related_bus_trip_link_df = agency_trip_link_df[
                    (agency_trip_link_df.shape_id == shape_id)
                ].copy()

                trip_node_df = self.roadway_network.nodes_df[
                    self.roadway_network.nodes_df.osm_node_id.isin(
                        related_bus_trip_link_df.u.tolist()
                        + related_bus_trip_link_df.v.tolist()
                    )
                ].copy()

                trip_stop_df = find_closest_node(
                    trip_stop_df,
                    trip_node_df,
                    unique_id=["agency_raw_name", "stop_id", "trip_id"],
                )

                stop_to_node_df = stop_to_node_df.append(
                    trip_stop_df, sort=False, ignore_index=True
                )

        stop_to_node_df.drop(["X", "Y"], axis=1, inplace=True)

        if "osm_node_id" in stop_df.columns:
            stop_df.drop(["osm_node_id", "shst_node_id"], axis=1, inplace=True)

        stop_df = pd.merge(
            stop_df, stop_to_node_df, how="inner", on=["agency_raw_name", "stop_id"]
        )

        column_list = self.feed.stops.columns.values.tolist() + ["trip_id"]
        self.bus_stops = stop_df[column_list]

    def route_rail_trip(self):
        """
        method that creates rail routes

        1. get rail routes
        2. for each rail shape, get stops
        3. create shape line string between stops
        """

        # get route type for trips, get rail trips
        trip_df = pd.merge(
            self.feed.trips,
            self.feed.routes,
            how="left",
            on=["agency_raw_name", "route_id"],
        )
        rail_trip_df = trip_df[trip_df["route_type"] != 3].copy()

        rail_trip_df["agency_shape_id"] = (
            rail_trip_df["agency_raw_name"] + "_" + rail_trip_df["shape_id"].astype(str)
        )

        RanchLogger.info(
            "representative trips include {} rail/ferry shapes, which are total of {} trips".format(
                rail_trip_df.agency_shape_id.nunique(), len(rail_trip_df)
            )
        )

        if len(rail_trip_df) == 0:
            return None, None

        # get rail shapes
        rail_shape_df = self.feed.shapes.copy()
        rail_shape_df["agency_shape_id"] = (
            rail_shape_df["agency_raw_name"]
            + "_"
            + rail_shape_df["shape_id"].astype(str)
        )
        rail_shape_df = rail_shape_df[
            rail_shape_df.agency_shape_id.isin(rail_trip_df.agency_shape_id.tolist())
        ]

        # for rails that have the same shape_id, but different stop patterns
        # e.g. AM trip A-B-D vs MD trip A-C-D
        # we need to get A-B-C-D to create rail links

        # get rail stop times
        rail_stop_times_df = pd.merge(
            self.feed.stop_times,
            rail_trip_df[["agency_raw_name", "trip_id", "shape_id", "agency_shape_id"]],
            how="inner",
            on=["agency_raw_name", "trip_id"],
        )

        rail_stop_times_df = pd.merge(
            rail_stop_times_df,
            self.feed.stops[["agency_raw_name", "stop_id", "stop_lat", "stop_lon"]],
            how="left",
            on=["agency_raw_name", "stop_id"],
        )

        # get agency-shape-stop correspondence
        rail_stop_times_df = rail_stop_times_df.drop_duplicates(
            subset=["agency_raw_name", "shape_id", "stop_id"]
        )

        # if gtfs has no or missing shapes for rails, e.g. ACE
        # use stop times as shapes
        if len(rail_shape_df) == 0 :
            new_rail_shape_df = rail_stop_times_df.copy()
            new_rail_shape_df = new_rail_shape_df.rename(
                columns = {
                    'stop_lat':'shape_pt_lat',
                    'stop_lon':'shape_pt_lon'
                }
            )
            rail_shape_df = rail_shape_df.append(
                new_rail_shape_df,
                sort = False,
                ignore_index = True
            )
        else:
            shape_id_list = rail_trip_df.agency_shape_id.unique()
            shape_id_not_in_shape_data = [
                s for s in shape_id_list if s not in rail_shape_df.agency_shape_id.tolist()
            ]
            # if there are shapes not in the shape.txt
            if len(shape_id_not_in_shape_data) > 0:
                new_rail_shape_df = rail_stop_times_df[
                    rail_stop_times_df.agency_shape_id.isin(shape_id_not_in_shape_data)
                ].copy()
                new_rail_shape_df = new_rail_shape_df.rename(
                    columns = {
                        'stop_lat':'shape_pt_lat',
                        'stop_lon':'shape_pt_lon'
                    }
                )
                rail_shape_df = rail_shape_df.append(
                    new_rail_shape_df,
                    sort = False,
                    ignore_index = True
                )

        rail_shape_stop_df = pd.DataFrame()

        # for each rail shape
        for agency_shape_id in rail_shape_df.agency_shape_id.unique():
            # find the closest shape node for each stop
            shape_df = rail_shape_df[
                rail_shape_df.agency_shape_id == agency_shape_id
            ].copy()
            # initialize columns
            shape_df["is_stop"] = np.int(0)
            shape_df["stop_id"] = np.nan

            shape_inventory = shape_df[["shape_pt_lon", "shape_pt_lat"]].values
            tree = cKDTree(shape_inventory)

            # stops on the shape
            stop_df = rail_stop_times_df[
                rail_stop_times_df.agency_shape_id == agency_shape_id
            ].copy()
            for s in range(len(stop_df)):
                point = stop_df.iloc[s][["stop_lon", "stop_lat"]].values
                dd, ii = tree.query(point, k=1)
                shape_df.shape_pt_lon.iloc[ii] = stop_df.iloc[s]["stop_lon"]
                shape_df.shape_pt_lat.iloc[ii] = stop_df.iloc[s]["stop_lat"]
                shape_df.is_stop.iloc[ii] = 1
                shape_df.stop_id.iloc[ii] = stop_df.iloc[s]["stop_id"]

            rail_shape_stop_df = rail_shape_stop_df.append(
                shape_df, ignore_index=True, sort=False
            )

        rail_trip_link_df = pd.DataFrame()
        # create new rail links
        for agency_shape_id in rail_shape_stop_df.agency_shape_id.unique():
            shape_df = rail_shape_stop_df[
                rail_shape_stop_df.agency_shape_id == agency_shape_id
            ].copy()

            agency_raw_name = rail_shape_stop_df[
                rail_shape_stop_df.agency_shape_id == agency_shape_id
            ]["agency_raw_name"].iloc[0]

            shape_id = rail_shape_stop_df[
                rail_shape_stop_df.agency_shape_id == agency_shape_id
            ]["shape_id"].iloc[0]

            # get rail nodes based on the stop flags
            break_list = shape_df.index[shape_df.is_stop == 1].tolist()

            stop_id_list = shape_df[shape_df.is_stop == 1]["stop_id"].tolist()

            # use the gtfs shape between "stop" shapes to build the rail true shape
            for j in range(len(break_list) - 1):
                lon_list = rail_shape_stop_df.shape_pt_lon.iloc[
                    break_list[j] : break_list[j + 1] + 1
                ].tolist()
                lat_list = rail_shape_stop_df.shape_pt_lat.iloc[
                    break_list[j] : break_list[j + 1] + 1
                ].tolist()
                linestring = LineString([Point(xy) for xy in zip(lon_list, lat_list)])
                rail_trip_link_df = rail_trip_link_df.append(
                    {
                        "agency_raw_name": agency_raw_name,
                        "shape_id": shape_id,
                        "from_stop_id": stop_id_list[j],
                        "to_stop_id": stop_id_list[j + 1],
                        "geometry": linestring,
                    },
                    ignore_index=True,
                    sort=False,
                )

        # drop duplicate rail links
        unique_rail_links_df = rail_trip_link_df.drop_duplicates(
            subset=["agency_raw_name", "from_stop_id", "to_stop_id"]
        )

        # rail links geodataframe
        unique_rail_links_gdf = gpd.GeoDataFrame(
            unique_rail_links_df,
            geometry=unique_rail_links_df["geometry"],
            crs=self.roadway_network.links_df.crs,
        )

        # create new rail nodes
        rail_nodes_df = rail_shape_stop_df[rail_shape_stop_df.is_stop == 1][
            ["stop_id", "agency_raw_name", "shape_pt_lon", "shape_pt_lat"]
        ].copy()

        # create node shst id for rail nodes, use 'agency_raw_name'+'stop_id'
        rail_nodes_df["shst_node_id"] = (
            rail_nodes_df["agency_raw_name"]
            + "_"
            + rail_nodes_df["stop_id"].astype(str)
        )

        # drop duplicate rail nodes
        unique_rail_nodes_df = rail_nodes_df.drop_duplicates(
            subset=["agency_raw_name", "stop_id"]
        )

        # rail nodes geodataframe
        unique_rail_nodes_gdf = gpd.GeoDataFrame(
            unique_rail_nodes_df,
            geometry=[
                Point(xy)
                for xy in zip(
                    unique_rail_nodes_df.shape_pt_lon, unique_rail_nodes_df.shape_pt_lat
                )
            ],
            crs=self.roadway_network.links_df.crs,
        )

        rail_stops = pd.merge(
            self.feed.stops.drop(["shst_node_id", 'osm_node_id'], axis=1),
            unique_rail_nodes_df[["agency_raw_name", "stop_id", "shst_node_id"]],
            how="inner",
            on=["agency_raw_name", "stop_id"],
        )

        # assign u and v to rail_trip_link_df
        rail_trip_link_df = pd.merge(
            rail_trip_link_df,
            unique_rail_nodes_df[["agency_raw_name", "stop_id", "shst_node_id"]].rename(
                columns={
                    "stop_id": "from_stop_id",
                    "shst_node_id": "fromIntersectionId",
                }
            ),
            how="left",
            on=["agency_raw_name", "from_stop_id"],
        )

        rail_trip_link_df = pd.merge(
            rail_trip_link_df,
            unique_rail_nodes_df[["agency_raw_name", "stop_id", "shst_node_id"]].rename(
                columns={"stop_id": "to_stop_id", "shst_node_id": "toIntersectionId"}
            ),
            how="left",
            on=["agency_raw_name", "to_stop_id"],
        )

        # TODO assign county to unique rail links and nodes before appending them to the roadway network

        # TODO assign model id to rail links and nodes

        self.rail_trip_link_df = rail_trip_link_df
        self.rail_stops = rail_stops

        self.unique_rail_links_gdf = unique_rail_links_gdf
        self.unique_rail_nodes_gdf = unique_rail_nodes_gdf

        self.roadway_network.links_df = self.roadway_network.links_df.append(
            unique_rail_links_gdf, sort=False, ignore_index=True
        )

        self.roadway_network.nodes_df = self.roadway_network.nodes_df.append(
            unique_rail_nodes_gdf, sort=False, ignore_index=True
        )

    def create_freq_table(self):

        """
        create frequency table for trips
        """
        RanchLogger.info('Creating frequency reference')

        tod_numhours_dict = {}
        model_time_period = self.parameters.model_time_period

        for key in model_time_period.keys():
            if model_time_period.get(key).get("frequency_start") is None:
                tod_numhours_dict[key] = model_time_period.get(key).get(
                    "end"
                ) - model_time_period.get(key).get("start")
            else:
                tod_numhours_dict[key] = model_time_period.get(key).get(
                    "frequency_end"
                ) - model_time_period.get(key).get("frequency_start")

        freq_df = self.feed.trips[
            ["agency_raw_name", "trip_id", "tod", "direction_id", "trip_num_for_shape", "trip_num_total", "trip_num_N_most"]
        ].copy()
        freq_df["headway_secs"] = freq_df.tod.map(tod_numhours_dict)
        freq_df["headway_secs"] = freq_df.apply(
            lambda x: int(x.headway_secs * 60 * 60 / (x.trip_num_for_shape / x.trip_num_N_most * x.trip_num_total)), 
            axis=1
        )

        model_time_enum_list = self.parameters.model_time_enum_list

        freq_df["start_time"] = freq_df.tod.map(model_time_enum_list.get("start_time"))
        freq_df["end_time"] = freq_df.tod.map(model_time_enum_list.get("end_time"))

        self.feed.frequencies = freq_df

    def create_shape_node_table(self):
        """
        create complete node lists each transit traverses to replace the gtfs shape.txt
        """
        if self.bus_trip_link_df is not None:
            bus_trip_link_df = self.bus_trip_link_df.copy()
            bus_trip_link_df['agency_trip_id'] = (
                bus_trip_link_df['agency_raw_name'] + 
                "_" + 
                bus_trip_link_df['trip_id'].astype(str)
            )
        
            bus_trip_link_with_unique_shape_id = bus_trip_link_df.drop_duplicates(
                subset=['agency_raw_name', "shape_id"]
            ).agency_trip_id.tolist()

            bus_trip_link_df = bus_trip_link_df[
                bus_trip_link_df.agency_trip_id.isin(bus_trip_link_with_unique_shape_id)
            ].copy()
        else:
            bus_trip_link_df = pd.DataFrame(
                columns=[
                    "u",
                    "v",
                    "fromIntersectionId",
                    "toIntersectionId",
                    "shape_id",
                    "agency_raw_name",
                ]
            )

        if self.rail_trip_link_df is not None:
            shape_link_df = pd.concat(
                [
                    bus_trip_link_df[
                        [
                            "u",
                            "v",
                            "fromIntersectionId",
                            "toIntersectionId",
                            "shape_id",
                            "agency_raw_name",
                        ]
                    ],
                    self.rail_trip_link_df[
                        [
                            "fromIntersectionId",
                            "toIntersectionId",
                            "shape_id",
                            "agency_raw_name",
                        ]
                    ],
                ],
                sort=False,
                ignore_index=True,
            )

        if self.rail_trip_link_df is None:
            shape_link_df = bus_trip_link_df.copy()

        shape_link_df.u = shape_link_df.u.fillna(0).astype(np.int64)
        shape_link_df.v = shape_link_df.v.fillna(0).astype(np.int64)

        shape_point_df = gpd.GeoDataFrame()

        if len(shape_link_df) == 0:
            self.shape_point_df = shape_point_df
            return

        for shape_id in shape_link_df.shape_id.unique():
            shape_df = shape_link_df[shape_link_df.shape_id == shape_id]
            point_df = pd.DataFrame(
                data={
                    "agency_raw_name": shape_df["agency_raw_name"].iloc[0],
                    "shape_id": shape_id,
                    "shape_osm_node_id": shape_df.u.tolist() + [shape_df.v.iloc[-1]],
                    "shape_shst_node_id": shape_df.fromIntersectionId.tolist()
                    + [shape_df.toIntersectionId.iloc[-1]],
                    # "shape_model_node_id" : shape_df.A.tolist() + [shape_df.B.iloc[-1]],
                    "shape_pt_sequence": range(1, 1 + len(shape_df) + 1),
                }
            )

            shape_point_df = pd.concat(
                [shape_point_df, point_df], sort=False, ignore_index=True
            )

        shape_point_df = pd.merge(
            shape_point_df,
            self.roadway_network.nodes_df[["osm_node_id", "shst_node_id", "geometry"]],
            how="left",
            left_on="shape_shst_node_id",
            right_on="shst_node_id",
        )

        shape_point_df.crs = self.parameters.standard_crs
        #shape_point_df = shape_point_df.to_crs(epsg = 4326)
        
        #print(shape_point_df[shape_point_df.geometry.isnull()])
        
        shape_point_df["shape_pt_lat"] = shape_point_df.geometry.map(lambda g:g.y)
        shape_point_df["shape_pt_lon"] = shape_point_df.geometry.map(lambda g:g.x)
        
        #shape_point_df["shape_id"] = shape_point_df["shape_id"].astype(int)
        
        #shape_point_df.rename(columns = {"shst_node_id":"shape_shst_node_id"}, inplace = True)

        self.shape_point_df = shape_point_df

    def write_standard_transit(self, path: Optional[str] = None):
        """
        write out transit network in standard format
        """

        if path is None:
            path = self.gtfs_dir

        else:
            path = path

        shape_point_df = self.shape_point_df.copy()

        if len(shape_point_df) == 0:
            RanchLogger.info('None of the GTFS routes are built as standard transit network.')
            return
        
        trip_df = self.feed.trips.copy()

        trip_df = trip_df[
            trip_df.shape_id.isin(shape_point_df.shape_id.unique().tolist())
        ]

        freq_df = self.feed.frequencies.copy()
        freq_df = pd.merge(
            freq_df,
            trip_df[["agency_raw_name", "trip_id"]],
            how="inner",
            on=["agency_raw_name", "trip_id"],
        )

        if self.rail_stops is None:
            stop_df = self.bus_stops.copy()
        else:
            stop_df = pd.concat(
                [self.bus_stops, self.rail_stops], sort=False, ignore_index=True
            )

        stop_times_df = self.feed.stop_times.copy()
        stop_times_df = pd.merge(
            stop_times_df,
            trip_df[["agency_raw_name", "trip_id"]],
            how="inner",
            on=["agency_raw_name", "trip_id"],
        )

        # update time to relative time for frequency based transit system
        stop_times_df["first_arrival"] = stop_times_df.groupby(
            ["agency_raw_name", "trip_id"]
        )["arrival_time"].transform(min)
        stop_times_df["arrival_time"] = (
            stop_times_df["arrival_time"] - stop_times_df["first_arrival"]
        )
        stop_times_df["departure_time"] = (
            stop_times_df["departure_time"] - stop_times_df["first_arrival"]
        )

        stop_times_df["arrival_time"] = stop_times_df["arrival_time"].apply(
            lambda x: time.strftime("%H:%M:%S", time.gmtime(x)) if ~np.isnan(x) else x
        )
        stop_times_df["departure_time"] = stop_times_df["departure_time"].apply(
            lambda x: time.strftime("%H:%M:%S", time.gmtime(x)) if ~np.isnan(x) else x
        )

        stop_times_df.drop(["first_arrival"], axis=1, inplace=True)

        # add model node id to stops
        if "model_node_id" in self.roadway_network.nodes_df.columns:
            stop_df = pd.merge(
                stop_df,
                self.roadway_network.nodes_df[['shst_node_id', 'model_node_id']],
                how = 'left',
                on = 'shst_node_id'
            )

        # add model node id to shapes
        if "model_node_id" in self.roadway_network.nodes_df.columns:
            shape_point_df = pd.merge(
                shape_point_df,
                self.roadway_network.nodes_df[['shst_node_id', 'model_node_id']].rename(
                    columns = {
                        'shst_node_id' : 'shape_shst_node_id', 
                        'model_node_id' : 'shape_model_node_id'
                    }
                ),
                how = 'left',
                on = 'shape_shst_node_id'
            )

        if 'geometry' in shape_point_df.columns:
            shape_point_df.drop(['geometry'], axis = 1, inplace = True)

        route_df = self.feed.routes.copy()
        route_df = pd.merge(
            route_df,
            trip_df[["agency_raw_name", "route_id"]].drop_duplicates(),
            how="inner",
            on=["agency_raw_name", "route_id"],
        )

        route_df.to_csv(os.path.join(path, "routes.txt"), index=False, sep=",")

        shape_point_df.to_csv(os.path.join(path, "shapes.txt"), index=False, sep=",")

        trip_df.to_csv(os.path.join(path, "trips.txt"), index=False, sep=",")

        freq_df.to_csv(os.path.join(path, "frequencies.txt"), index=False, sep=",")

        stop_df.to_csv(os.path.join(path, "stops.txt"), index=False, sep=",")

        stop_times_df.to_csv(os.path.join(path, "stop_times.txt"), index=False, sep=",")

        self.feed.agency.to_csv(os.path.join(path, "agency.txt"), index=False, sep=",")

        self.feed.fare_attributes.to_csv(
            os.path.join(path, "fare_attributes.txt"), index=False, sep=","
        )

        self.feed.fare_rules.to_csv(
            os.path.join(path, "fare_rules.txt"), index=False, sep=","
        )

        if self.unique_rail_links_gdf is not None:
            self.write_rail_links_and_nodes(path)
    
    def write_rail_links_and_nodes(
        self,
        path: Optional[str] = None
    ):
        """
        write out rail links and nodes, instead of directly appending them to the radway_network
        """

        # add columns to unique_rail_nodes_gdf
        self.unique_rail_nodes_gdf['rail_only'] = 1

        # add columns to unique_rail_links_gdf
        self.unique_rail_links_gdf['rail_only'] = 1
        if 'fromIntersectionId' not in self.unique_rail_links_gdf.columns:
            self.unique_rail_links_gdf = pd.merge(
                self.unique_rail_links_gdf,
                self.unique_rail_nodes_gdf[['agency_raw_name', 'stop_id', 'shst_node_id']].rename(
                    columns = {'stop_id' : 'from_stop_id'}
                ),
                how = 'left',
                on = ['agency_raw_name', 'from_stop_id']
            )
            self.unique_rail_links_gdf.rename(
                columns = {'shst_node_id' : 'fromIntersectionId'},
                inplace = True
            )
        if 'toIntersectionId' not in self.unique_rail_links_gdf.columns:
            self.unique_rail_links_gdf = pd.merge(
                self.unique_rail_links_gdf,
                self.unique_rail_nodes_gdf[['agency_raw_name', 'stop_id', 'shst_node_id']].rename(
                    columns = {'stop_id' : 'to_stop_id'}
                ),
                how = 'left',
                on = ['agency_raw_name', 'to_stop_id']
            )
            self.unique_rail_links_gdf.rename(
                columns = {'shst_node_id' : 'toIntersectionId'},
                inplace = True
            )

        # write out
        self.unique_rail_links_gdf.to_file(
            os.path.join(path, 'rail_links.geojson')
        )

        self.unique_rail_nodes_gdf.to_file(
            os.path.join(path, 'rail_nodes.geojson')
        )

class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    Source: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
