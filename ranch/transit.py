from __future__ import annotations

import os
import re
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import partridge as ptg
import peartree as pt
from partridge.config import default_config
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import time

from .logger import RanchLogger
from .parameters import Parameters
from .roadway import Roadway
from .utils import geodesic_point_buffer, ox_graph, find_closest_node
from .sharedstreets import run_shst_match, read_shst_extraction

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
        feed: DotDict = None, 
        roadway_network: Roadway = None,
        parameters: Union[Parameters, dict] = {}
    ):
        """
        Constructor
        """
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
        path: str,
        roadway_network: Roadway,
        parameters: Dict
    ):
        """
        read from every GTFS folders in the path and combine then into one
        """

        gtfs_agencies_list = os.listdir(path)

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
                os.path.join(path, agency)
            )

            # gtfs cannot read fare tables for all agencies
            fare_attributes_df = pd.DataFrame()
            fare_rules_df = pd.DataFrame()

            if "fare_attributes.txt" in os.listdir(os.path.join(path, agency)):
        
                fare_attributes_df = pd.read_csv(os.path.join(path, agency, "fare_attributes.txt"),
                                         dtype = {"fare_id" : str})
    
            if "fare_rules.txt" in os.listdir(os.path.join(path, agency)):
        
                fare_rules_df = pd.read_csv(
                    os.path.join(path, agency, "fare_rules.txt"),
                    dtype = {"fare_id" : str, "route_id" : str, "origin_id" : str, "destination_id" : str,
                         " route_id" : str, " origin_id" : str, " destination_id" : str,})      

            feed.agency = pd.concat([feed.agency, agency_feed.agency], sort = False, ignore_index = True)
            feed.routes = pd.concat([feed.routes, agency_feed.routes], sort = False, ignore_index = True)
            feed.trips = pd.concat([feed.trips, agency_feed.trips], sort = False, ignore_index = True)
            feed.stops = pd.concat([feed.stops, agency_feed.stops], sort = False, ignore_index = True)
            feed.stop_times = pd.concat([feed.stop_times, agency_feed.stop_times], sort = False, ignore_index = True)
            feed.shapes = pd.concat([feed.shapes, agency_feed.shapes], sort = False, ignore_index = True)
            feed.fare_attributes = pd.concat([feed.fare_attributes, fare_attributes_df], sort = False, ignore_index = True)
            feed.fare_rules = pd.concat([feed.fare_rules, fare_rules_df], sort = False, ignore_index = True)
            feed.frequencies = pd.concat([feed.frequencies, agency_feed.frequencies], sort = False, ignore_index = True)

        transit_network = Transit(
            feed = feed,
            roadway_network = roadway_network,
            parameters=parameters
        )

        return transit_network


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
                os.path.join(feed_path, "calendar_orig.txt"),
                index = False,
                sep = ","
            )
    
            calendar_df["weekdays"] = calendar_df.apply(
                lambda x: x.monday + x.tuesday + x.wednesday + x.thursday + x.friday,
                axis = 1
            )
            calendar_df = calendar_df[calendar_df.weekdays > 0]
    
            calendar_df.drop("weekdays", axis = 1).to_csv(
                os.path.join(feed_path, "calendar.txt"),
                index = False,
                sep = ","
            )

        RanchLogger.info("Read and get representative transit feed from: {}".format(feed_path))
        
        feed = pt.get_representative_feed(feed_path)

        agency_gtfs_name = os.path.basename(feed_path)
        
        validate_feed = Transit.validate_feed(feed, agency_gtfs_name)

        return validate_feed

    def validate_feed(
        feed,
        agency_gtfs_name
    ):
        """
        add fields and validate feed 
        """

        feed.routes["agency_raw_name"] = agency_gtfs_name
    
        feed.stops["agency_raw_name"] = agency_gtfs_name
    
        feed.trips["agency_raw_name"] = agency_gtfs_name
    
        # when 'direction_id' is not present, fill in default
        if "direction_id" not in feed.trips.columns: # Marguerita
            feed.trips["direction_id"] = 0
    
        feed.trips["direction_id"].fillna(0, inplace = True)
   
        feed.shapes["agency_raw_name"] = agency_gtfs_name
    
        feed.stop_times["agency_raw_name"] = agency_gtfs_name
    
        feed.agency["agency_raw_name"] = agency_gtfs_name
        
        # add agency_id in routes.txt if missing
        if "agency_id" not in feed.routes.columns:
            if "agency_id" in feed.agency.columns:
                feed.agency["agency_id"] = feed.agency.agency_id.iloc[0]
    
        if len(feed.shapes) == 0: # ACE, CCTA, VINE
            
            RanchLogger("missing shapes.txt for {}".format(agency_gtfs_name))
            
            group_df = feed.trips.groupby(
                ["route_id", "direction_id"]
            )["trip_id"].first().reset_index().drop("trip_id", axis = 1)

            group_df["shape_id"] = range(1, len(group_df) + 1)

            if "shape_id" in feed.trips.columns:
                feed.trips.drop("shape_id", axis = 1, inplace = True)
                feed.trips = pd.merge(feed.trips, group_df, how = "left", on = ["route_id", "direction_id"])
        
        if len(feed.trips[feed.trips.shape_id.isnull()]) > 0:
            RanchLogger("partial complete shape_id for {}".format(agency_gtfs_name))
            
            trips_missing_shape_df = feed.trips[feed.trips.shape_id.isnull()].copy()
            
            group_df = trips_missing_shape_df.groupby(
                ["route_id", "direction_id"]
            )["trip_id"].first().reset_index().drop("trip_id", axis = 1)
            group_df["shape_id"] = range(1, len(group_df) + 1)
            group_df["shape_id"] = group_df["shape_id"].apply(lambda x: "psudo" + str(x))
            
            trips_missing_shape_df = pd.merge(
                trips_missing_shape_df.drop("shape_id", axis = 1), 
                group_df, 
                how = "left", 
                on = ["route_id", "direction_id"]
            )

            feed.trips = pd.concat(
                [
                    feed.trips[feed.trips.shape_id.notnull()], 
                    trips_missing_shape_df
                ],
                ignore_index = True,
                sort = False
            )

        return feed

    def get_representative_trip_for_route(
        self
    ):
        
        """
        get the representative trips for each route, by direction, tod
        
        """
        
        RanchLogger.info('Getting representative trip for rach route by time of day and direction...')
        
        # get the first stop of each trip to determine the time period for each trip
        # process time
        stop_times_df = self.feed.stop_times.copy()
        stop_times_df['arrival_h'] = pd.to_datetime(stop_times_df['arrival_time'], unit = 's').dt.hour
        stop_times_df['arrival_m'] = pd.to_datetime(stop_times_df['arrival_time'], unit = 's').dt.minute
        stop_times_df['departure_h'] = pd.to_datetime(stop_times_df['departure_time'], unit = 's').dt.hour
        stop_times_df['departure_m'] = pd.to_datetime(stop_times_df['departure_time'], unit = 's').dt.minute
        
        # according to the gtfs reference, the stop sequence does not have to be consecutive, but has to always increase
        # so we can get the fisrt stop by the smallest stop sequence on the trip
        stop_times_df.sort_values(
            by = ['agency_raw_name', "trip_id", "stop_sequence"], 
            ascending = True, 
            inplace = True
        )
        first_stop_df = stop_times_df.drop_duplicates(subset = ['agency_raw_name', "trip_id"])
        
        ## identify peak, offpeak trips, based on the arrival time of first stop
        trip_df = self.feed.trips.copy()
        trip_df = pd.merge(
            trip_df, 
            first_stop_df,
            how = 'left',
            on = ['agency_raw_name', 'trip_id']
        )
        
        model_time_period = self.parameters.model_time_period

        ## AM: 6-10am, MD: 10am-3pm, PM: 3-7pm, NT 7pm-3am, EA 3-6am
        trip_df['tod'] = np.where(
            (trip_df['arrival_h'] >= model_time_period.get('AM').get('start')) & (trip_df['arrival_h'] < model_time_period.get('AM').get('end')),
            'AM',
            np.where(
                (trip_df['arrival_h'] >= model_time_period.get('MD').get('start')) & (trip_df['arrival_h'] < model_time_period.get('MD').get('end')),
                'MD',
                np.where(
                    (trip_df['arrival_h'] >= model_time_period.get('PM').get('start')) & (trip_df['arrival_h'] < model_time_period.get('PM').get('end')),
                    'PM',
                    np.where(
                        (trip_df['arrival_h'] >= model_time_period.get('EA').get('start')) & (trip_df['arrival_h'] < model_time_period.get('EA').get('end')),
                        'EA',
                        'NT'
                    )
                )
            )
        )
    
        # calculate frequency for EA and NT period using 5-6am, and 7-10pm
        trip_EA_NT_df = trip_df.copy()
        trip_EA_NT_df["tod"] = np.where(
            (trip_df['arrival_h'] >= model_time_period.get('EA').get('frequency_start')) & (trip_df['arrival_h'] < model_time_period.get('EA').get('frequency_end')),
            "EA",
            np.where(
                (trip_df['arrival_h'] >= model_time_period.get('NT').get('frequency_start')) & (trip_df['arrival_h'] < model_time_period.get('NT').get('frequency_end')),
                "NT",
                "NA"
            )
        )
        
        # get the most frequent trip for each route, by direction, by time of day
        ## trips share the same shape_id is considered being the same
        ## first get the trip count for each shape_id
        trip_freq_df = trip_df.groupby(
            ['agency_raw_name', 'route_id', 'tod', 'direction_id', 'shape_id']
        )['trip_id'].count().to_frame()
        
        ## then choose the most frequent shape_id for each route
        # for frequency use the total number of trips
        def agg(x):
            m = x.shape_id.iloc[np.argmax(x.trip_id.values)]
            return pd.Series({'trip_num' : x.trip_id.sum(), 'shape_id' : m})
    
        trip_freq_df = trip_freq_df.reset_index().groupby(
            ['agency_raw_name', 'route_id', 'tod', 'direction_id']
        ).apply(agg)
        
        # retain the complete trip info of the represent trip only
        trip_df = pd.merge(
            trip_df, 
            trip_freq_df.reset_index(),
            how = 'inner',
            on = ['agency_raw_name', 'route_id', 'tod', 'direction_id', 'shape_id']
            ).drop_duplicates(
                ['agency_raw_name', 'route_id', 'direction_id', 'tod']
            )
            
        trip_EA_NT_df = pd.merge(
            trip_EA_NT_df, 
            trip_freq_df.reset_index(),
            how = 'inner',
            on = ['agency_raw_name', 'route_id', 'tod', 'direction_id', 'shape_id']
        )
        
        trip_EA_NT_df = trip_EA_NT_df[
            trip_EA_NT_df.tod.isin(["EA", "NT"])
            ].groupby(
            ['agency_raw_name', "route_id", "tod", "direction_id", "shape_id"]
            )["trip_id"].count().reset_index()
        
        trip_EA_NT_df.rename(columns = {"trip_id" : "trip_num"}, inplace = True)
        
        trip_df = pd.merge(
            trip_df,
            trip_EA_NT_df,
            how = "left",
            on = ['agency_raw_name', "route_id", "tod", "direction_id", "shape_id"]
        )
        
        trip_df["trip_num"] = np.where(
            trip_df.trip_num_y.isnull(),
            trip_df.trip_num_x,
            trip_df.trip_num_y
        )

        self.feed.trips = trip_df

    def snap_stop_to_node(
        self
    ):
        
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
        
        RanchLogger.info('Snapping gtfs stops to roadway node osmid...')
        
        # get rid of motorway nodes
        non_motorway_links_df = self.roadway_network.links_df[
            ~self.roadway_network.links_df.roadway.isin(["motorway", "motorway_link"])
        ].copy()

        node_candidates_for_stops_df = self.roadway_network.nodes_df[
            self.roadway_network.nodes_df.shst_node_id.isin(
                non_motorway_links_df.fromIntersectionId.tolist() + non_motorway_links_df.toIntersectionId.tolist()
                )
            ].copy()

        stop_df = self.feed.stops.copy()
        stop_df['geometry'] = [Point(xy) for xy in zip(stop_df['stop_lon'], stop_df['stop_lat'])]
        stop_df = gpd.GeoDataFrame(stop_df)
        stop_df.crs = {'init' : 'epsg:4326'}
        
        RanchLogger.info('Snapping gtfs stops to roadway node osmid...')
        stop_to_node_gdf = find_closest_node(
            stop_df, 
            node_candidates_for_stops_df,
            unique_id = ['agency_raw_name', 'stop_id']
        )
        
        stop_to_node_gdf.drop(['X','Y'], axis = 1, inplace = True)
        stop_df = pd.merge(
            stop_df,
            stop_to_node_gdf, 
            how = 'left', 
            on = ['agency_raw_name', 'stop_id']
        )
        
        column_list = self.feed.stops.columns.values.tolist() + ['osm_node_id', 'shst_node_id']
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
            good_links_buffer_radius = self.parameters.transit_routing_parameters.get('good_links_buffer_radius')

        if ft_penalty:
            ft_penalty = ft_penalty
        else:
            ft_penalty = self.parameters.transit_routing_parameters.get('ft_penalty')

        if non_good_links_penalty:
            non_good_links_penalty = non_good_links_penalty
        else:
            non_good_links_penalty = self.parameters.transit_routing_parameters.get('non_good_links_penalty')

        trip_df = self.feed.trips.copy()
        stop_df = self.feed.stops.copy()
        stop_time_df = self.feed.stop_times.copy()
        links_gdf = self.roadway_network.links_df.copy()
        nodes_gdf = self.roadway_network.nodes_df.copy()
        
        # append stop info to stop times table
        stop_time_df = pd.merge(
            stop_time_df,
            stop_df,
            how = 'left',
            on = ['agency_raw_name', 'stop_id']
        )

        # for each stop, get which trips are using them
        stop_trip_df = stop_time_df.drop_duplicates(
            subset = ['agency_raw_name', 'trip_id', 'stop_id'])
        
        RanchLogger.info('Routing bus on roadway network from start to end with osmnx...')
        
        # get route type for trips, get bus trips
        trip_df = pd.merge(trip_df, self.feed.routes, how = 'left', on = ['agency_raw_name', 'route_id'])
        bus_trip_df = trip_df[trip_df['route_type'] == 3]

        # for trips with same shape_id, keep the one with the most #stops
        # count # stops on each trip
        num_stops_on_trips_df = stop_time_df.groupby(
            ['agency_raw_name', "trip_id"]
        )["stop_id"].count().reset_index().rename(columns = {"stop_id": "num_stop"})
        
        bus_trip_df = pd.merge(
            bus_trip_df, 
            num_stops_on_trips_df[['agency_raw_name', "trip_id", "num_stop"]], 
            how = "left", 
            on = ['agency_raw_name', "trip_id"]
        )
        
        bus_trip_df.sort_values(
            by = ["num_stop"], 
            inplace = True, 
            ascending = False
        )

        # keep the trip with most stops
        bus_trip_df.drop_duplicates(
            subset = ['agency_raw_name', 'route_id', "shape_id"], 
            keep = "first", 
            inplace = True
        )

        # get stops that are on bus trips only
        stops_on_bus_trips_df = stop_trip_df[
            (stop_trip_df['agency_raw_name'].isin(bus_trip_df['agency_raw_name'].unique())) &
            (stop_trip_df['trip_id'].isin(bus_trip_df['trip_id'].unique()))
        ].copy()
        stops_on_bus_trips_df.drop_duplicates(
            subset=['agency_raw_name', 'stop_id'],
            inplace = True
        )

        RanchLogger.info("Setting good link dictionary")

        # set good link dictionary based on stops
        self.set_good_links(
            stops_on_bus_trips_df,
            good_links_buffer_radius
        )

        # output dataframe for osmnx success
        trip_osm_link_df = pd.DataFrame()
        
        # loop through bus trips
        for agency_raw_name in bus_trip_df.agency_raw_name.unique():
            trip_id_list = bus_trip_df[
                bus_trip_df['agency_raw_name'] == agency_raw_name
            ]['trip_id'].tolist()

            for trip_id in trip_id_list:
            
                RanchLogger.info("routing {} trip {}".format(agency_raw_name, trip_id))
                # get the stops on the trip
                trip_stops_df = stop_time_df[
                    (stop_time_df['trip_id'] == trip_id) &
                    (stop_time_df['agency_raw_name'] == agency_raw_name)
                ].copy()
            
                # get the good links from good links dictionary
                good_links_list = self.get_good_link_for_trip(
                    trip_stops_df
                )
            
                # update link weights
                links_gdf["length_weighted"] = np.where(
                    links_gdf.shstReferenceId.isin(good_links_list), 
                    links_gdf["length"], 
                    links_gdf["length"]*non_good_links_penalty
                )
            
                # apply ft penalty
                links_gdf["ft_penalty"] = links_gdf["roadway"].map(ft_penalty)
                links_gdf["ft_penalty"].fillna(ft_penalty["default"], inplace = True)
            
                links_gdf["length_weighted"] = links_gdf["length_weighted"] * links_gdf["ft_penalty"]
            
                # update graph
                
                G_trip = ox_graph(nodes_gdf, links_gdf)
            
                trip_stops_df.sort_values(by = ["stop_sequence"], inplace = True)
            
                # from first stop node OSM id
                closest_node_to_first_stop = int(trip_stops_df.osm_node_id.iloc[0])
                    
                # to last stop node OSM id
                closest_node_to_last_stop = int(trip_stops_df.osm_node_id.iloc[-1])
                
                RanchLogger.info("Routing trip {} from stop {} to stop {}".format(
                    trip_stops_df.trip_id.unique(),
                    trip_stops_df.stop_id.iloc[0],
                    trip_stops_df.stop_id.iloc[-1],
                ))

                path_osm_link_df = Transit.get_link_path_between_nodes(
                    G_trip,
                    closest_node_to_first_stop,
                    closest_node_to_last_stop,
                    weight_field = "length_weighted"
                )

                path_osm_link_df['trip_id'] = trip_id
                path_osm_link_df['agency_raw_name'] = agency_raw_name
                
                trip_osm_link_df = trip_osm_link_df.append(
                    path_osm_link_df, 
                    ignore_index = True, 
                    sort = False)    
    
        # after routing all trips, join with the links
        trip_osm_link_df = pd.merge(
            trip_osm_link_df, 
            trip_df[['agency_raw_name', 'trip_id', 'shape_id']], 
            how = 'left', 
            on = ['agency_raw_name', 'trip_id']
        )

        trip_osm_link_df = pd.merge(
            trip_osm_link_df,
            self.roadway_network.links_df[["u", "v", "wayId", "shstReferenceId", "shstGeometryId", 'geometry']].drop_duplicates(subset = ["u", "v"]),
            how = "left",
            on = ["u", "v"]
        )
        
        self.trip_osm_link_df = trip_osm_link_df

    def get_link_path_between_nodes(
        G,
        from_node,
        to_node,
        weight_field
    ):
        """
        return the complete links from start node to end node using networkx routing
        """

        # routing btw from and to nodes, return the list of nodes
        node_osmid_list = nx.shortest_path(
            G, 
            from_node, 
            to_node, 
            weight_field
        )

        # circular route
        if from_node == to_node:
            osm_link_df = pd.DataFrame(
                {
                    'u' : [from_node], 
                    'v' : [from_node],
                },
            )

            return osm_link_df
                    
        # get the links
        if len(node_osmid_list) > 1:
            osm_link_df = pd.DataFrame(
                {
                    'u' : node_osmid_list[:len(node_osmid_list)-1], 
                    'v' : node_osmid_list[1:len(node_osmid_list)],
                },
            )
                
            return osm_link_df
        else:
            return pd.DataFrame()

    def set_good_links(
        self,
        stops,
        good_links_buffer_radius
    ):

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
            #non_motorway_links_df, 
            drive_links_df,
            stops,
            buffer_radius = good_links_buffer_radius
        )

        good_link_dict = stop_good_link_df.groupby(
            ['agency_raw_name', 'stop_id']
        )['shstReferenceId'].apply(
            list
        ).to_dict()

        self.good_link_dict = good_link_dict

    def get_good_link_for_trip(
        self,
        trip_stops
    ):
        """
        for input stop IDs return a list of the good link IDs
        """

        link_shstReferenceId_list = []
        for agency_raw_name in trip_stops['agency_raw_name'].unique():
            stop_id_list = trip_stops[trip_stops['agency_raw_name'] == agency_raw_name]['stop_id'].unique()
            for stop_id in stop_id_list:
                #print(stop_id)
                if self.good_link_dict.get(
                    (agency_raw_name,
                    stop_id)
                ):
                    link_shstReferenceId_list += self.good_link_dict.get(
                        (agency_raw_name,
                        stop_id)
                    )

        return link_shstReferenceId_list

    def links_within_stop_buffer(
        drive_link_df, 
        stops, 
        buffer_radius
        ):
        """
        find the links that are within buffer of nodes
        """
        
        stop_buffer_df = stops.copy()
        stop_buffer_df["geometry"] = stop_buffer_df.apply(
            lambda x: geodesic_point_buffer(x.stop_lat, x.stop_lon, buffer_radius), 
            axis = 1
        )
        
        stop_buffer_df = gpd.GeoDataFrame(
            stop_buffer_df, 
            geometry = stop_buffer_df["geometry"], 
            crs = {'init' : 'epsg:4326'}
        )
        
        stop_buffer_link_df = gpd.sjoin(
            drive_link_df, 
            stop_buffer_df[["geometry", "agency_raw_name","stop_id"]], 
            how = "left", 
            op = "intersects"
        )
        
        stop_buffer_link_df = stop_buffer_link_df[
            stop_buffer_link_df.stop_id.notnull()
        ]
        
        return stop_buffer_link_df

    def set_bad_stops(
        self, 
        bad_stop_buffer_radius: Optional[float] = None
        ):
        """
        for each stop location, check if the routed route is within 50 meters
        """

        if bad_stop_buffer_radius:
            bad_stop_buffer_radius = bad_stop_buffer_radius
        else:
            bad_stop_buffer_radius = self.parameters.transit_routing_parameters.get('bad_stops_buffer_radius')

        trip_df = self.feed.trips.copy()
        stop_df = self.feed.stops.copy()
        stop_time_df = self.feed.stop_times.copy()

        trip_osm_link_gdf = gpd.GeoDataFrame(
            self.trip_osm_link_df, 
            geometry = self.trip_osm_link_df["geometry"],
            crs = self.roadway_network.links_df.crs)
        
        # get chained stops on a trip
        chained_stop_df = stop_time_df[
            stop_time_df['trip_id'].isin(trip_df.trip_id.tolist())
        ].copy()
        chained_stop_df = pd.merge(
            chained_stop_df, 
            stop_df,
            how = 'left',
            on = ['agency_raw_name', 'stop_id']
        )
        
        dict_stop_far_from_trip = {}
        
        # loop through agency and bus trips
        for agency_raw_name in trip_osm_link_gdf['agency_raw_name'].unique():
            
            trip_id_list = trip_osm_link_gdf[
                trip_osm_link_gdf['agency_raw_name'] == agency_raw_name
            ]['trip_id'].unique()
            
            for trip_id in trip_id_list:
            
                # get the stops on the trip
                trip_stop_df = chained_stop_df[
                    (chained_stop_df['trip_id'] == trip_id) & 
                    (chained_stop_df['agency_raw_name'] == agency_raw_name)
                ].copy()
            
                dict_stop_far_from_trip[(agency_raw_name,trip_id)] = []
            
                for i in range(len(trip_stop_df)):
                
                    single_stop_df = trip_stop_df.iloc[i: i+1].copy()
                
                    trip_links_gdf = trip_osm_link_gdf[
                        (trip_osm_link_gdf['trip_id'] == trip_id) &
                        (trip_osm_link_gdf['agency_raw_name'] == agency_raw_name)
                    ].copy()
                
                    # get the links that are within stop buffer
                    links_list = Transit.links_within_stop_buffer(
                        trip_links_gdf, 
                        single_stop_df, 
                        buffer_radius = bad_stop_buffer_radius
                    )
            
                    if len(links_list) == 0:
                        dict_stop_far_from_trip[(agency_raw_name,trip_id)] += single_stop_df['stop_id'].tolist()
        
        self.bad_stop_dict = dict_stop_far_from_trip
    
    def route_bus_link_osmnx_between_stops(
        self, 
        good_links_buffer_radius: Optional[float] = None, 
        ft_penalty: Optional[Dict] = None ,
        non_good_links_penalty: Optional[float] = None,
    ):
        """
        route bus trips between the bad stops
        """

        if good_links_buffer_radius:
            good_links_buffer_radius = good_links_buffer_radius
        else:
            good_links_buffer_radius = self.parameters.transit_routing_parameters.get('good_links_buffer_radius')

        if ft_penalty:
            ft_penalty = ft_penalty
        else:
            ft_penalty = self.parameters.transit_routing_parameters.get('ft_penalty')

        if non_good_links_penalty:
            non_good_links_penalty = non_good_links_penalty
        else:
            non_good_links_penalty = self.parameters.transit_routing_parameters.get('non_good_links_penalty')

        trip_df = self.feed.trips.copy()
        stop_df = self.feed.stops.copy()
        stop_time_df = self.feed.stop_times.copy()
        links_gdf = self.roadway_network.links_df.copy()
        nodes_gdf = self.roadway_network.nodes_df.copy()
        
        # append stop info to stop times table
        stop_time_df = pd.merge(
            stop_time_df,
            stop_df,
            how = 'left',
            on = ['agency_raw_name', 'stop_id']
        )
        
        RanchLogger.info('Routing bus on roadway network from start to end with osmnx...')
        
        # get route type for trips, get bus trips
        trip_df = pd.merge(trip_df, self.feed.routes, how = 'left', on = ['agency_raw_name', 'route_id'])
        bus_trip_df = trip_df[trip_df['route_type'] == 3]

        # for trips with same shape_id, keep the one with the most #stops
        num_stops_on_trips_df = stop_time_df.groupby(
            ['agency_raw_name', "trip_id"]
        )["stop_id"].count().reset_index().rename(columns = {"stop_id": "num_stop"})
        
        bus_trip_df = pd.merge(
            bus_trip_df, 
            num_stops_on_trips_df[['agency_raw_name', "trip_id", "num_stop"]], 
            how = "left", 
            on = ['agency_raw_name', "trip_id"]
        )
        
        bus_trip_df.sort_values(
            by = ["num_stop"], 
            inplace = True, 
            ascending = False
        )

        # keep the trip with most stops
        bus_trip_df.drop_duplicates(
            subset = ['agency_raw_name', 'route_id', "shape_id"], 
            keep = "first", 
            inplace = True
        )
        
        # output dataframe for osmnx success
        trip_osm_link_df = pd.DataFrame()
        
        # loop through for bus trips
        for agency_raw_name in bus_trip_df.agency_raw_name.unique():
            trip_id_list = bus_trip_df[
                bus_trip_df['agency_raw_name'] == agency_raw_name
            ]['trip_id'].tolist()

            for trip_id in trip_id_list:
            
                shape_id = bus_trip_df[
                    (bus_trip_df['agency_raw_name'] == agency_raw_name) &
                    (bus_trip_df['trip_id'] == trip_id)
                ]['shape_id'].iloc[0]

                # get the stops on the trip
                trip_stops_df = stop_time_df[
                    (stop_time_df['trip_id'] == trip_id) &
                    (stop_time_df['agency_raw_name'] == agency_raw_name)
                ].copy()
            
                # get the links that are within stop buffer
                good_links_list = self.get_good_link_for_trip(
                    trip_stops_df
                )

                # update link weights
                links_gdf["length_weighted"] = np.where(
                    links_gdf.shstReferenceId.isin(good_links_list), 
                    links_gdf["length"], 
                    links_gdf["length"]*non_good_links_penalty
                )
            
                # apply ft penalty
                links_gdf["ft_penalty"] = links_gdf["roadway"].map(ft_penalty)
                links_gdf["ft_penalty"].fillna(ft_penalty["default"], inplace = True)
            
                links_gdf["length_weighted"] = links_gdf["length_weighted"] * links_gdf["ft_penalty"]
            
                # update graph

                G_trip = ox_graph(nodes_gdf, links_gdf)
            
                trip_stops_df.sort_values(by = ["stop_sequence"], inplace = True)
                
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
                
                for s in range(len(route_by_stop_df)-1):
                    # from stop node OSM id
                    closest_node_to_first_stop = int(route_by_stop_df.osm_node_id.iloc[s])
                    
                    # to stop node OSM id
                    closest_node_to_last_stop = int(route_by_stop_df.osm_node_id.iloc[s+1])
                    
                    # osmnx routing btw from and to stops, return the list of nodes
                    RanchLogger.info("Routing trip {} from stop {} node {}, to stop {} node {}".format(
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
                        weight_field = "length_weighted"
                    )
                    
                    path_osm_link_df['trip_id'] = trip_id
                    path_osm_link_df['agency_raw_name'] = agency_raw_name
                    path_osm_link_df['shape_id'] = shape_id
                
                    trip_osm_link_df = trip_osm_link_df.append(
                        path_osm_link_df, 
                        ignore_index = True, 
                        sort = False)   

        # after routing all trips, join with the links
        trip_osm_link_df = pd.merge(
            trip_osm_link_df.drop('trip_id', axis = 1), 
            trip_df[['agency_raw_name', 'trip_id', 'shape_id']], 
            how = 'left', 
            on = ['agency_raw_name', 'shape_id']
        )

        trip_osm_link_df = pd.merge(
            trip_osm_link_df,
            self.roadway_network.links_df[["u", "v", "wayId", "shstReferenceId", "shstGeometryId", 'geometry']].drop_duplicates(subset = ["u", "v"]),
            how = "left",
            on = ["u", "v"]
        )
        
        self.trip_osm_link_df = trip_osm_link_df

    def route_gtfs_using_shortest_path(
        self
    ):
        """
        method that calls methods for routing using shortest path
        """
        RanchLogger.info("Route bus trips using shortest path")
        self.route_bus_link_osmnx_from_start_to_end()
        self.set_bad_stops()
        self.route_bus_link_osmnx_between_stops()

    def match_gtfs_shapes_to_shst(
        self,
        path: Optional[str] = None
    ):
        """
        1. call the method that matches gtfs shapes to shst,
        2. clean up the match result
        """
        if path:
            path = path
        else:
            path = self.parameters.data_interim_dir

        RanchLogger.info("Route bus trips using shst match")
        
        self._match_gtfs_shapes_to_shst(path)

        self.trip_shst_link_df = pd.merge(
            self.trip_shst_link_df.drop(['geometry'], axis = 1), 
            self.roadway_network.links_df[
                ['shstReferenceId','wayId','u','v', "fromIntersectionId", "toIntersectionId", 'geometry']
            ],
            how = 'left',
            on = 'shstReferenceId'
        )

        self.trip_shst_link_df["u"] = self.trip_shst_link_df["u"].fillna(0).astype(np.int64)
        self.trip_shst_link_df["v"] = self.trip_shst_link_df["v"].fillna(0).astype(np.int64)

        trip_shst_link_df = self.trip_shst_link_df.copy()

        trip_shst_link_df['next_agency_raw_name'] = trip_shst_link_df['agency_raw_name'].iloc[1:].append(
            pd.Series(trip_shst_link_df['agency_raw_name'].iloc[-1])).reset_index(drop=True)

        trip_shst_link_df['next_shape_id'] = trip_shst_link_df['shape_id'].iloc[1:].append(
            pd.Series(trip_shst_link_df['shape_id'].iloc[-1])).reset_index(drop=True)
    
        trip_shst_link_df['next_u'] = trip_shst_link_df['u'].iloc[1:].append(
            pd.Series(trip_shst_link_df['v'].iloc[-1])).reset_index(drop=True)
    
        incomplete_trip_shst_link_df = trip_shst_link_df[
            (trip_shst_link_df.agency_raw_name==trip_shst_link_df.next_agency_raw_name) &
            (trip_shst_link_df.shape_id==trip_shst_link_df.next_shape_id) &
            (trip_shst_link_df.v!=trip_shst_link_df.next_u)
        ].copy()

        incomplete_trip_shst_link_df['agency_shape_id'] = incomplete_trip_shst_link_df['agency_raw_name'] + "_" + incomplete_trip_shst_link_df['shape_id'].astype(str)
    
        self.trip_shst_link_df['agency_shape_id'] = self.trip_shst_link_df['agency_raw_name'] + "_" + self.trip_shst_link_df['shape_id'].astype(str)

        self.trip_shst_link_df = self.trip_shst_link_df[
            ~(
                self.trip_shst_link_df.agency_shape_id.isin(incomplete_trip_shst_link_df.agency_shape_id.unique())
            )
        ]

        self.trip_shst_link_df = pd.merge(
            self.trip_shst_link_df,
            self.feed.trips[['agency_raw_name', 'trip_id', 'shape_id']], 
            how = 'left', 
            on = ['agency_raw_name', 'shape_id']
        )

    def _match_gtfs_shapes_to_shst(
        self,
        path: str
    ):
        """
        1. write out geojson from gtfs shapes for shst match,
        2. run the actual match method,
        3. read the match result
        """
        
        shapes_df = self.feed.shapes.copy()
        shapes_df = shapes_df[shapes_df.shape_id.isin(self.feed.trips.shape_id.unique())]
        
        shapes_df = gpd.GeoDataFrame(
            shapes_df, 
            geometry = gpd.points_from_xy(shapes_df['shape_pt_lon'], shapes_df['shape_pt_lat']),
            crs = self.roadway_network.links_df.crs
        )
        
        lines_from_shapes_df = shapes_df.groupby(
            ['agency_raw_name', 'shape_id']
            )['geometry'].apply(lambda x:LineString(x.tolist())).reset_index()
        lines_from_shapes_df = gpd.GeoDataFrame(
            lines_from_shapes_df, 
            geometry = 'geometry'
            )
    
        lines_from_shapes_df.to_file(
            os.path.join(path, "lines_from_shapes.geojson"), 
            driver='GeoJSON')
        
        run_shst_match(
            input_network_file = os.path.join(path, "lines_from_shapes.geojson"),
            input_unqiue_id=['agency_raw_name', 'shape_id'],
            output_dir = path,
            custom_match_option = '--follow-line-direction --tile-hierarchy=8'
        )

        trip_shst_link_df = read_shst_extraction(path, "*.matched.geojson")

        trip_shst_link_df.rename(
            columns = {
                'pp_agency_raw_name' : 'agency_raw_name',
                'pp_shape_id' : 'shape_id',
            },
            inplace = True
        )

        self.trip_shst_link_df = trip_shst_link_df

    def route_bus_trip(
        self,
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
                self.trip_osm_link_df, 
                crs = self.roadway_network.links_df.crs)

            trip_osm_link_gdf.drop('wayId',axis=1).to_file(
                os.path.join(self.parameters.scratch_location, 'test_routing.geojson'),
                driver = 'GeoJSON'
            )
           
        # route using shts match
        if self.trip_shst_link_df is None:
            self.match_gtfs_shapes_to_shst()

        # get route type for trips, get bus trips
        trip_df = pd.merge(
            self.feed.trips, 
            self.feed.routes, 
            how = 'left', on = ['agency_raw_name', 'route_id'])
        bus_trip_df = trip_df[trip_df['route_type'] == 3].copy()

        bus_trip_df['agency_shape_id'] = bus_trip_df['agency_raw_name'] + "_" + bus_trip_df['shape_id'].astype(str)

        RanchLogger.info("representative trips include {} bus shapes, {} bus trips".format(
            bus_trip_df.agency_shape_id.nunique(),
            len(bus_trip_df)
        ))

        trip_shst_link_df = self.trip_shst_link_df.copy()

        # keep bus shapes from shst match
        trip_shst_link_df = trip_shst_link_df[
            (trip_shst_link_df.agency_shape_id.isin(bus_trip_df.agency_shape_id.unique()))
        ]

        trip_shst_link_df['method'] = 'shst match'

        RanchLogger.info("shst matched {} bus shapes, {} bus trips".format(
            trip_shst_link_df.agency_shape_id.nunique(),
            len(trip_shst_link_df.groupby(['agency_raw_name', 'trip_id']).count()))
        )

        # keep bus shapes in shortest path routing that are not in shst match
        trip_osm_link_df = self.trip_osm_link_df.copy()
        trip_osm_link_df['agency_shape_id'] = trip_osm_link_df['agency_raw_name'] + "_" + trip_osm_link_df['shape_id'].astype(str)

        trip_osm_link_df = trip_osm_link_df[
            ~(trip_osm_link_df.agency_shape_id.isin(trip_shst_link_df.agency_shape_id.unique()))
        ]

        trip_osm_link_df['method'] = 'shortest path'

        RanchLogger.info("shortest path method matched {} bus shapes, {} bus trips".format(
            trip_osm_link_df.agency_shape_id.nunique(),
            len(trip_osm_link_df.groupby(['agency_raw_name', 'trip_id']).count()))
        )

        bus_trip_link_df = pd.concat(
            [
                trip_osm_link_df, 
                trip_shst_link_df[trip_osm_link_df.columns]
            ],
            sort = False,
            ignore_index = True
        )

        self.bus_trip_link_df = bus_trip_link_df

    def update_bus_stop_node(
        self
    ):
        """
        after routing buses, update bus stop nodes
        match stops to the nodes that are on the bus links
        """

        # first for each stop, get what trips use them
        stop_time_df = self.feed.stop_times.copy()
        stop_df = self.feed.stops.copy()

        stop_df['geometry'] = [Point(xy) for xy in zip(stop_df['stop_lon'], stop_df['stop_lat'])]
        stop_df = gpd.GeoDataFrame(stop_df)
        stop_df.crs = {'init' : 'epsg:4326'}

        ## append stop info to stop times table
        stop_time_df = pd.merge(
            stop_time_df,
            stop_df,
            how = 'left',
            on = ['agency_raw_name', 'stop_id']
        )

        stop_time_df = gpd.GeoDataFrame(stop_time_df, crs = stop_df.crs)

        # get route type for trips, get bus trips
        trip_df = pd.merge(
            self.feed.trips, 
            self.feed.routes, 
            how = 'left', on = ['agency_raw_name', 'route_id'])
        bus_trip_df = trip_df[trip_df['route_type'] == 3]

        stop_time_df = pd.merge(
            stop_time_df,
            bus_trip_df[['agency_raw_name', 'trip_id']],
            how = 'inner',
            on = ['agency_raw_name', 'trip_id']
        )

        stop_to_node_df = pd.DataFrame()

        for agency_raw_name in self.bus_trip_link_df.agency_raw_name.unique():
            agency_trip_link_df = self.bus_trip_link_df[
                self.bus_trip_link_df.agency_raw_name == agency_raw_name
            ].copy()
            
            for trip_id in agency_trip_link_df.trip_id.unique():
                
                trip_stop_df = stop_time_df[
                    (stop_time_df.trip_id == trip_id) &
                    (stop_time_df.agency_raw_name == agency_raw_name)
                ].copy()

                related_bus_trip_link_df = agency_trip_link_df[
                    (agency_trip_link_df.trip_id == trip_id)
                ].copy()

                trip_node_df = self.roadway_network.nodes_df[
                    self.roadway_network.nodes_df.osm_node_id.isin(
                        related_bus_trip_link_df.u.tolist() +
                        related_bus_trip_link_df.v.tolist()
                    )
                ].copy()

                trip_stop_df = find_closest_node(
                    trip_stop_df, 
                    trip_node_df,
                    unique_id = ['agency_raw_name', 'stop_id', 'trip_id']
                )
        
                stop_to_node_df = stop_to_node_df.append(
                    trip_stop_df,
                    sort = False,
                    ignore_index = True
                )
        """
        ## for each stop, get which trips are using them
        stop_trip_df = stop_time_df.groupby(
            ['agency_raw_name', 'stop_id']
            )['trip_id'].unique().to_frame().reset_index()

        stop_to_node_df = pd.DataFrame()

        # then get the common nodes on those trips / shapes
        # comments: common nodes are rare, as the routing is not perfect
        # will make stop nodes unique to trips, duplicate the stop record when necessary
        for index, row in stop_trip_df.iterrows():
            agency = row.agency_raw_name
            stop_id = row.stop_id
            node_list = []

            for trip_id in row.trip_id:
                related_bus_trip_link_df = self.bus_trip_link_df[
                    (self.bus_trip_link_df.agency_raw_name == agency) & 
                    (self.bus_trip_link_df.trip_id == trip_id)
                ].copy()
                if len(related_bus_trip_link_df) > 0:
                    node_list.append(list(set(
                        related_bus_trip_link_df.u.tolist() + 
                        related_bus_trip_link_df.v.tolist()
                    )))

            common_node_list = list(set.intersection(*map(set, node_list)))

            common_node_df = self.roadway_network.nodes_df[
                (self.roadway_network.nodes_df.osm_node_id.isin(
                    common_node_list
                ))
            ].copy()

            single_stop_df = stop_df[
                (stop_df.agency_raw_name == agency) &
                (stop_df.stop_id == stop_id)
            ].copy()

            print(common_node_df)
            # then match stops to close nodes
            single_stop_df = find_closest_node(
                single_stop_df, 
                common_node_df,
                unique_id = ['agency_raw_name', 'stop_id']
            )
        
            stop_to_node_df = stop_to_node_df.append(
                single_stop_df,
                sort = False,
                ignore_index = True
            )
        """
        stop_to_node_df.drop(['X','Y'], axis = 1, inplace = True)

        if 'osm_node_id' in stop_df.columns:
            stop_df.drop(['osm_node_id', 'shst_node_id'], axis=1, inplace=True)
        
        stop_df = pd.merge(
            stop_df,
            stop_to_node_df, 
            how = 'left',
            on = ['agency_raw_name', 'stop_id']
        )

        column_list = self.feed.stops.columns.values.tolist() + ['trip_id']
        self.bus_stops = stop_df[column_list]

    def create_freq_table(
        self
    ):
        
        """
        create frequency table for trips
        """
        
        RanchLogger.info('creating frequency reference...')
        
        tod_numhours_dict = {}
        model_time_period = self.parameters.model_time_period

        for key in model_time_period.keys():
            if model_time_period.get(key).get('frequency_start') is None:
                tod_numhours_dict[key] = model_time_period.get(key).get('end') - model_time_period.get(key).get('start')
            else:
                tod_numhours_dict[key] = model_time_period.get(key).get('frequency_end') - model_time_period.get(key).get('frequency_start')     

        freq_df = self.feed.trips[['agency_raw_name', 'trip_id', 'tod', 'direction_id', 'trip_num']].copy()
        freq_df['headway_secs'] = freq_df.tod.map(tod_numhours_dict)
        freq_df['headway_secs'] = freq_df.apply(
            lambda x: int(x.headway_secs * 60 * 60 / x.trip_num),
            axis = 1)
        
        model_time_enum_list = self.parameters.model_time_enum_list
        
        freq_df['start_time'] = freq_df.tod.map(model_time_enum_list.get("start_time"))
        freq_df['end_time'] = freq_df.tod.map(model_time_enum_list.get("end_time"))
    
        self.feed.frequencies = freq_df 

    def create_shape_node_table(
        self
    ):
        """
        create complete node lists each transit traverses to replace the gtfs shape.txt
        """
        bus_trip_link_df = self.bus_trip_link_df.copy()
        bus_trip_link_with_unique_shape_id = bus_trip_link_df.drop_duplicates(
            subset = ["shape_id"]
        ).trip_id.tolist()
        
        bus_trip_link_df = bus_trip_link_df[
            bus_trip_link_df.trip_id.isin(bus_trip_link_with_unique_shape_id)
        ].copy()
        
        if self.rail_trip_link_df is not None:
            shape_link_df = pd.concat(
                [   
                    bus_trip_link_df[["u", "v", 'shape_id']],
                    self.rail_trip_link_df[['u', 'v', 'shape_id']]
                ],
                sort = False,
                ignore_index = True)
        
        if self.rail_trip_link_df is None:
            shape_link_df = bus_trip_link_df.copy()
        
        shape_link_df.u = shape_link_df.u.fillna(0).astype(np.int64)
        shape_link_df.v = shape_link_df.v.fillna(0).astype(np.int64)

        shape_point_df = gpd.GeoDataFrame()
        
        for shape_id in shape_link_df.shape_id.unique():
            shape_df = shape_link_df[shape_link_df.shape_id == shape_id]
            point_df = pd.DataFrame(
                data = {
                    "shape_id" : shape_id,
                    "shape_osm_node_id" : shape_df.u.tolist() + [shape_df.v.iloc[-1]],
                    #"shape_model_node_id" : shape_df.A.tolist() + [shape_df.B.iloc[-1]],
                    "shape_pt_sequence" : range(1, 1+len(shape_df)+1)}
            )
    
            shape_point_df = pd.concat(
                [shape_point_df,point_df],
                sort = False,
                ignore_index = True)

        shape_point_df = pd.merge(
            shape_point_df,
            self.roadway_network.nodes_df[["osm_node_id", "shst_node_id", "geometry"]],
            how = "left",
            left_on = "shape_osm_node_id",
            right_on = "osm_node_id")
        
        shape_point_df.crs = {'init' : 'epsg:4326'}
        #shape_point_df = shape_point_df.to_crs(epsg = 4326)
        
        RanchLogger.info(shape_point_df[shape_point_df.geometry.isnull()])
        
        shape_point_df["shape_pt_lat"] = shape_point_df.geometry.map(lambda g:g.y)
        shape_point_df["shape_pt_lon"] = shape_point_df.geometry.map(lambda g:g.x)
        
        shape_point_df["shape_id"] = shape_point_df["shape_id"].astype(int)
        
        shape_point_df.rename(columns = {"shst_node_id":"shape_shst_node_id"}, inplace = True)
            
        self.shape_point_df = shape_point_df

    def write_standard_transit(
        self,
        path: Optional[str] = None
    ):
        """
        write out transit network in standard format
        """

        if path is None:
            path = self.parameters.scratch_location

        else:
            path = path
        
        shape_point_df = self.shape_point_df.copy()
        trip_df = self.feed.trips.copy()
        
        trip_df["shape_id"] = trip_df["shape_id"].astype(int)
        
        trip_df = trip_df[
            trip_df.shape_id.isin(shape_point_df.shape_id.unique().tolist())
        ]
        
        freq_df = self.feed.frequencies.copy()
        freq_df = pd.merge(
            freq_df,
            trip_df[['agency_raw_name', 'trip_id']],
            how = 'inner',
            on = ['agency_raw_name', 'trip_id']
        )
        
        if self.rail_stops is None:
            stop_df = self.bus_stops.copy()
        else:
            stop_df = pd.concat(
                [self.bus_stops, self.rail_stops],
                sort = False,
                ignore_index= True
            )

        stop_times_df = self.feed.stop_times.copy()
        stop_times_df = pd.merge(
            stop_times_df,
            trip_df[['agency_raw_name', 'trip_id']],
            how = 'inner',
            on = ['agency_raw_name', 'trip_id']
        )
        
        # update time to relative time for frequency based transit system
        stop_times_df['first_arrival'] = stop_times_df.groupby(['agency_raw_name', 'trip_id'])['arrival_time'].transform(min)
        stop_times_df['arrival_time'] = stop_times_df['arrival_time'] - stop_times_df['first_arrival']
        stop_times_df['departure_time'] = stop_times_df['departure_time'] - stop_times_df['first_arrival']
        
        stop_times_df['arrival_time'] = stop_times_df['arrival_time'].apply(
            lambda x : time.strftime('%H:%M:%S', time.gmtime(x)) if ~np.isnan(x) else x)
        stop_times_df['departure_time'] = stop_times_df['departure_time'].apply(
            lambda x : time.strftime('%H:%M:%S', time.gmtime(x)) if ~np.isnan(x) else x)

        
        stop_times_df.drop(['first_arrival'], axis = 1, inplace = True)
        
        route_df = self.feed.routes.copy()
        route_df = pd.merge(
            route_df,
            trip_df[['agency_raw_name', 'route_id']],
            how = 'inner',
            on = ['agency_raw_name', 'route_id']
        )
        
        route_df.to_csv(os.path.join(path, "routes.txt"), 
                        index = False, 
                        sep = ',')
    
        shape_point_df.to_csv(os.path.join(path, "shapes.txt"), 
                            index = False, 
                            sep = ',')
    
        trip_df.to_csv(os.path.join(path, "trips.txt"), 
                    index = False, 
                    sep = ',')
    
        freq_df.to_csv(os.path.join(path, "frequencies.txt"), 
                        index = False, 
                        sep = ',')
        
        stop_df.to_csv(os.path.join(path, "stops.txt"), 
                    index = False, 
                    sep = ',')
    
        stop_times_df.to_csv(os.path.join(path, "stop_times.txt"), 
                            index = False, 
                            sep = ',')


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
