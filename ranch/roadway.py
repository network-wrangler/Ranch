import copy
import itertools
from ntpath import join
import os
from typing import Dict, Optional, Union

import geopandas as gpd
from geopandas import GeoDataFrame
from pandas import DataFrame
import pandas as pd
import numpy as np
from shapely.geometry import Point

from .logger import RanchLogger
from .sharedstreets import read_shst_extraction, extract_osm_link_from_shst_extraction
from .osm import add_two_way_osm, highway_attribute_list_to_value
from .utils import create_unique_node_id, fill_na, identify_dead_end_nodes, buffer1, get_non_near_connectors, haversine_distance
from .utils import generate_centroid_connectors_link, generate_centroid_connectors_shape, create_unique_shape_id, create_unique_link_id
from .parameters import Parameters

class Roadway(object):
    """
    Roadway Network Object
    """

    def __init__(
        self,
        nodes: GeoDataFrame,
        links: GeoDataFrame,
        shapes: GeoDataFrame,
        parameters: Union[Parameters, dict] = {},
    ):
        """
        Constructor

        Args:
            nodes: geodataframe of nodes
            links: dataframe of links
            shapes: geodataframe of shapes
            parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not specified, will use default parameters.

        """
        self.nodes_df = nodes
        self.links_df = links
        self.shapes_df = shapes

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

    def create_roadway_network_from_extracts(
        shst_extract_dir: str,
        osm_extract_dir: str,
        parameters: Dict,
    ):
        """
        creates roadway network from shst and osm extracts
        """

        if not shst_extract_dir:
            msg = "Please specify directory for sharedstreet extraction files."
            RanchLogger.error(msg)
            raise ValueError(msg)

        if not osm_extract_dir:
            msg = "Please specify directory for osmnx extraction files."
            RanchLogger.error(msg)
            raise ValueError(msg)

        if shst_extract_dir:
            RanchLogger.info("Reading sharedstreets data")
            shst_link_gdf = read_shst_extraction(shst_extract_dir, "*.out.geojson")

            # shst geometry file might have duplicates, if multiple geometries has overlapping tiles
            # drop duplicates

            RanchLogger.info("Removing duplicates in shst extraction data")
            RanchLogger.info("...before removing duplicates, shst extraction has {} geometries.".format(shst_link_gdf.shape[0]))

            shst_link_non_dup_gdf = shst_link_gdf.drop_duplicates(
                subset = ['id', 'fromIntersectionId', 'toIntersectionId', 'forwardReferenceId', 'backReferenceId']
            )

            RanchLogger.info("...after removing duplicates, shst extraction has {} geometries.".format(shst_link_non_dup_gdf.shape[0]))
        
        if osm_extract_dir:
            RanchLogger.info("Reading osmnx data")
            osmnx_link_gdf = gpd.read_file(os.path.join(osm_extract_dir, "link.geojson"))
            osmnx_node_gdf = gpd.read_file(os.path.join(osm_extract_dir, "node.geojson"))

        RanchLogger.info("Extracting corresponding osm ways for every shst geometry")
        osm_from_shst_link_df = extract_osm_link_from_shst_extraction(
            shst_link_non_dup_gdf
        )

        # add two-way osm links
        osm_from_shst_link_df = add_two_way_osm(osm_from_shst_link_df, osmnx_link_gdf)

        # fill na
        osm_from_shst_link_df = fill_na(osm_from_shst_link_df)

        # aggregate osm data back to shst geometry based links
        link_gdf = Roadway.consolidate_osm_way_to_shst_link(osm_from_shst_link_df)

        # calculate roadway property
        highway_to_roadway_df = pd.read_csv(parameters.highway_to_roadway_crosswalk_file).fillna("")

        highway_to_roadway_dict = pd.Series(highway_to_roadway_df.roadway.values, 
                                    index = highway_to_roadway_df.highway).to_dict()

        roadway_hierarchy_dict = pd.Series(highway_to_roadway_df.hierarchy.values, 
                                   index = highway_to_roadway_df.roadway).to_dict()
    
        link_gdf["roadway"] = link_gdf.apply(lambda x: highway_attribute_list_to_value(x, 
                                                                              highway_to_roadway_dict,
                                                                              roadway_hierarchy_dict),
                                    axis = 1)
        
        # there are links with different shstgeomid, but same shstrefid, to/from nodes
        # drop one of the links that have two shstGeomId

        link_gdf.drop_duplicates(subset = ["shstReferenceId"],
                        inplace = True)

        # add network type variables

        network_type_df = pd.read_csv(parameters.network_type_file)

        link_gdf = pd.merge(link_gdf,
                    network_type_df,
                    how = "left",
                    on = "roadway")

        # create node gdf

        node_gdf = Roadway.create_node_gdf(link_gdf)
        node_gdf = Roadway.add_network_type_for_nodes(link_gdf, node_gdf)

        # create shape gdf
        shape_gdf = shst_link_non_dup_gdf[shst_link_non_dup_gdf.id.isin(link_gdf.shstGeometryId.tolist())].copy()

        roadway_network = Roadway(
            nodes = node_gdf,
            links = link_gdf,
            shapes = shape_gdf,
            parameters = parameters
        )

        return roadway_network

    def consolidate_osm_way_to_shst_link(osm_link):
        """
        if a shst link has more than one osm ways, aggregate info into one, e.g. series([1,2,3]) to cell value [1,2,3]
        
        Parameters
        ----------
        osm link with shst info
        
        return
        ----------
        shst link with osm info
        
        """
        osm_link_gdf = osm_link.copy()

        agg_dict = {"geometry" : lambda x: x.iloc[0],
                    "u" : lambda x: x.iloc[0],
                    "v" : lambda x: x.iloc[-1]}
        
        for c in osm_link_gdf.columns:
            if c in ['link', 'nodeIds', 'oneWay', 'roadClass', 'roundabout', 'wayId', 'access', 'area', 'bridge',
                'est_width', 'highway', 'junction', 'key', 'landuse', 'lanes', 'maxspeed', 'name', 'oneway', 'ref', 'service', 
                'tunnel', 'width']:
                agg_dict.update({c : lambda x: list(x) if len(list(x)) > 1 else list(x)[0]})
        
        RanchLogger.info("Start aggregating osm segments to one shst link for forward links")
        forward_link_gdf = osm_link_gdf[osm_link_gdf.reverse_out == 0].copy()
        
        if len(forward_link_gdf) > 0:
            forward_link_gdf = forward_link_gdf.groupby(
                                            ["shstReferenceId", "id", "shstGeometryId", "fromIntersectionId", "toIntersectionId"]
                                            ).agg(agg_dict).reset_index()
            forward_link_gdf["forward"] = 1
        else:
            forward_link_gdf = None
        
        RanchLogger.info("Start aggregating osm segments to one shst link for backward links")
        
        backward_link_gdf = osm_link_gdf[osm_link_gdf.reverse_out==1].copy()
        
        if len(backward_link_gdf) > 0:
            agg_dict.update({"u" : lambda x: x.iloc[-1],
                        "v" : lambda x: x.iloc[0]})    

            backward_link_gdf = backward_link_gdf.groupby(
                                            ["shstReferenceId", "id", "shstGeometryId", "fromIntersectionId", "toIntersectionId"]
                                            ).agg(agg_dict).reset_index()
        else:
            backward_link_gdf = None
        
        shst_link_gdf = None
        
        if (forward_link_gdf is None):
            RanchLogger.info("back")
            shst_link_gdf = backward_link_gdf
            
        if (backward_link_gdf is None):
            RanchLogger.info("for")
            shst_link_gdf = forward_link_gdf
            
        if (forward_link_gdf is not None) and (backward_link_gdf is not None):
            RanchLogger.info("all")
            shst_link_gdf = pd.concat([forward_link_gdf, backward_link_gdf],
                                    sort = False,
                                    ignore_index = True)
            
        shst_link_gdf = GeoDataFrame(shst_link_gdf,
                                        crs = {'init': 'epsg:4326'})
        
        return shst_link_gdf

    def create_node_gdf(link_gdf):
        """
        create shst node gdf from shst geometry
        
        Paramters
        ---------
        link_gdf:  shst links with osm info
        
        return
        ---------
        shst nodes with osm info
        
        """
        RanchLogger.info("Start creating shst nodes")
        # geometry only matches for forward direction
        forward_link_gdf = link_gdf[link_gdf.forward == 1].copy()
        
        # create point geometry from shst linestring
        forward_link_gdf["u_point"] = forward_link_gdf.apply(lambda x: Point(list(x.geometry.coords)[0]), axis = 1)
        forward_link_gdf["v_point"] = forward_link_gdf.apply(lambda x: Point(list(x.geometry.coords)[-1]), axis = 1)
        
        # get from points
        point_gdf = forward_link_gdf[["u", "fromIntersectionId", "u_point"]].copy()
        
        point_gdf.rename(columns = {"u" : "osm_node_id",
                        "fromIntersectionId" : "shst_node_id",
                        "u_point" : "geometry"},
                        inplace = True)
        
        # append to points
        point_gdf = pd.concat([point_gdf, forward_link_gdf[["v", "toIntersectionId", "v_point"]].rename(columns = 
                        {"v" : "osm_node_id",
                        "toIntersectionId" : "shst_node_id",
                        "v_point" : "geometry"})],
                        sort = False,
                        ignore_index = True)
        
        # drop duplicates
        point_gdf.drop_duplicates(subset = ["osm_node_id", "shst_node_id"], inplace = True)
        
        point_gdf = GeoDataFrame(point_gdf,
                                    crs = {'init': 'epsg:4326'})
        
        return point_gdf

    def add_network_type_for_nodes(links, nodes):
        """
        add network type variable for node
        """
        A_B_df = pd.concat([links[["u", "drive_access", "walk_access", "bike_access"]].rename(columns = {"u":"osm_node_id"}),
                        links[["v", "drive_access", "walk_access", "bike_access"]].rename(columns = {"v":"osm_node_id"})],
                        sort = False,
                        ignore_index = True)

        A_B_df.drop_duplicates(inplace = True)

        A_B_df = A_B_df.groupby("osm_node_id").max().reset_index()

        node_gdf = pd.merge(nodes,
                            A_B_df,
                            how = "left",
                            on = "osm_node_id")

        return node_gdf

    # step 5 tidy roadway
    def tidy_roadway(
        self,
        county_boundary_file: str,
        county_variable_name: str,
    ):
        """
        step 5: clean up roadway object
        """

        if not county_boundary_file:
            msg = "Missing polygon file for county boundary."
            RanchLogger.error(msg)
            raise ValueError(msg)

        if county_boundary_file:
            filename, file_extension = os.path.splitext(county_boundary_file)
            if file_extension in [".shp", ".geojson"]:
                county_gdf = gpd.read_file(county_boundary_file)
                self.county_gdf = county_gdf
                self.county_variable_name = county_variable_name
        
            else:
                msg = "Invalid boundary file, should be .shp or .geojson"
                RanchLogger.error(msg)
                raise ValueError(msg)

        RanchLogger.info("Starting Step 5 Tidy Roadway")

        ## 5.0 join county name to shapes and nodes
        self._calculate_county(
            county_gdf = county_gdf,
            county_variable_name = county_variable_name
        )

        ## 5.1 keep links within county boundary, keep nodes and shapes accordingly
        self._keep_links_nodes_within_county()

        ## 5.2 drop circular links
        self._drop_circular_links()

        ## 5.3 flag dead end 
        self._make_dead_end_non_drive()

        ## 5.4 drop duplicate links between same AB node pair
        self._drop_alternative_links_between_same_AB_nodes()

        ## 5.5 link and node numbering
        self._link_node_numbering()

    def _calculate_county(
        self,
        county_gdf: GeoDataFrame,
        county_variable_name: str,
    ):
        links_df = self.links_df.copy()
        nodes_df = self.nodes_df.copy()

        # links_centroid_df['geometry'] = links_centroid_df["geometry"].centroid
        RanchLogger.info(
            "Joining network with county boundary file for {} county".format(county_gdf[county_variable_name].unique())
        )

        joined_links_gdf = gpd.sjoin(
            links_df, 
            county_gdf, 
            how="left", 
            op="intersects"
        )

        # for links that cross county boudaries and potentially sjoin-ed to two counties
        # drop duplciates, keep one county match
        joined_links_gdf.drop_duplicates(subset = ['shstReferenceId'], inplace = True)
        joined_links_gdf.rename(
            columns = {county_variable_name : 'county'},
            inplace = True
        )

        # links_centroid_df['geometry'] = links_centroid_df["geometry"].centroid
        joined_nodes_gdf = gpd.sjoin(
            nodes_df, 
            county_gdf, 
            how="left", 
            op="intersects"
        )

        # for links that cross county boudaries and potentially sjoin-ed to two counties
        # drop duplciates, keep one county match
        joined_nodes_gdf.drop_duplicates(
            subset = ['osm_node_id','shst_node_id'], 
            inplace = True
        )

        joined_nodes_gdf.rename(
            columns = {county_variable_name : 'county'},
            inplace = True
        )
        
        joined_nodes_gdf['county'].fillna('external', inplace = True)

        # join back to roadway object
        self.links_df = pd.merge(
            self.links_df,
            joined_links_gdf[['shstReferenceId', 'county']],
            how = 'left',
            on = ['shstReferenceId']
        )

        self.nodes_df = pd.merge(
            self.nodes_df,
            joined_nodes_gdf[['osm_node_id', 'shst_node_id', 'county']],
            how = 'left',
            on = ['osm_node_id', 'shst_node_id']
        )

    def _keep_links_nodes_within_county(
        self,
    ):
        """
        drop links and nodes that are outside of the region
        """
        RanchLogger.info(
            "Dropping links and nodes that are outside of {} county".format(self.links_df.county.dropna().unique())
        )

        self.links_df = self.links_df[
            self.links_df.county.notnull()
        ]
        self.nodes_df = self.nodes_df[
            self.nodes_df.shst_node_id.isin(
                self.links_df.fromIntersectionId.tolist() + 
                self.links_df.toIntersectionId.tolist()
                )
        ]
        self.shapes_df = self.shapes_df[
            self.shapes_df.id.isin(
                self.links_df.shstGeometryId.tolist()
            )
        ]

    def _make_dead_end_non_drive(
        self,
    ):
        """
        iterative process to identify dead end nodes
        make dead end links and nodes drive_access = 0
        """

        RanchLogger.info("Flagging dead-end streets for drive network")

        non_dead_end_link_handle_df = self.links_df[
            (self.links_df.drive_access == 1)
        ][["u", "v"]]

        dead_end_node_list = identify_dead_end_nodes(non_dead_end_link_handle_df)

        cumulative_dead_end_node_list = []

        while len(dead_end_node_list) > 0:
            
            cumulative_dead_end_node_list = cumulative_dead_end_node_list + dead_end_node_list
    
            non_dead_end_link_handle_df = non_dead_end_link_handle_df[
                ~(non_dead_end_link_handle_df.u.isin(dead_end_node_list)) & 
                ~(non_dead_end_link_handle_df.v.isin(dead_end_node_list))
            ].copy()
    
            dead_end_node_list = identify_dead_end_nodes(non_dead_end_link_handle_df)
        
        # update node and link drive access
        # if u/v in dead end node list, then drive access = 0
        # if osm_node_id in dead end node list, then drive access = 0
        RanchLogger.info("Making drive-end streets drive_access = 1")

        self.links_df['drive_access'] = np.where(
            (
                (self.links_df.u.isin(cumulative_dead_end_node_list)) | 
                (self.links_df.v.isin(cumulative_dead_end_node_list))
            ) &
            ~(self.links_df.roadway.isin(
                ['primary', 'secondary', 'motorway', 'primary_link','motorway_link', 'trunk_link', 'trunk', 'secondary_link','tertiary_link']
                )
            ),
            0,
            self.links_df.drive_access)
        
        # update network type variable for node

        A_B_df = pd.concat(
            [
                self.links_df[["u", "fromIntersectionId", "drive_access", "walk_access", "bike_access"]].rename(
                    columns = {"u":"osm_node_id", "fromIntersectionId" : "shst_node_id"}),
                self.links_df[["v", "toIntersectionId", "drive_access", "walk_access", "bike_access"]].rename(
                    columns = {"v":"osm_node_id", "toIntersectionId" : "shst_node_id"})
            ],
            sort = False,
            ignore_index = True)

        A_B_df.drop_duplicates(inplace = True)

        A_B_df = A_B_df.groupby(["osm_node_id", "shst_node_id"]).max().reset_index()

        self.nodes_df = pd.merge(
            self.nodes_df.drop(["drive_access", "walk_access", "bike_access"], axis = 1),
            A_B_df,
            how = "left",
            on = ["osm_node_id", "shst_node_id"]
        )

    def _drop_circular_links(
        self,
    ):
        """
        drop circular links
        """

        RanchLogger.info("Droppping circular links")
        circular_link_gdf = self.links_df[
            self.links_df.u == self.links_df.v
        ].copy()
        
        self.links_df = self.links_df[
            ~self.links_df.shstReferenceId.isin(circular_link_gdf.shstReferenceId.tolist())
        ]
        
        self.shapes_df = self.shapes_df[self.shapes_df.id.isin(self.links_df.id)]
        self.nodes_df = self.nodes_df[
            (self.nodes_df.osm_node_id.isin(self.links_df.u.tolist())) | 
            (self.nodes_df.osm_node_id.isin(self.links_df.v.tolist()))
        ]

    def _drop_alternative_links_between_same_AB_nodes(
        self,
    ):
        """
        drop duplicate links between same AB node pair
        those are not allowed in model networks

        keep links that are higher hierarchy, drive, bike, walk, longer length
        """

        # add length in meters

        geom_length = self.links_df[['geometry']].copy()
        geom_length = geom_length.to_crs(epsg = 26915)
        geom_length["length"] = geom_length.length

        self.links_df["length"] = geom_length["length"]

        RanchLogger.info("Dropping alternative links between same AB nodes")

        non_unique_AB_links_df = self.links_df.groupby(["u", "v"]).shstReferenceId.count().sort_values().reset_index()
        non_unique_AB_links_df = non_unique_AB_links_df[non_unique_AB_links_df.shstReferenceId > 1]

        non_unique_AB_links_df = pd.merge(
            non_unique_AB_links_df[["u", "v"]],
            self.links_df[["u", "v", "highway", "roadway", "drive_access", "bike_access", "walk_access", "length","wayId", "shstGeometryId", "shstReferenceId", "geometry"]],
            how = "left",
            on = ["u", "v"]
        )

        roadway_hierarchy_df = pd.read_csv(
            os.path.join(self.parameters.highway_to_roadway_crosswalk_file)
        )

        roadway_hierarchy_df = roadway_hierarchy_df.drop_duplicates(subset = "roadway")

        non_unique_AB_links_df = pd.merge(
            non_unique_AB_links_df,
            roadway_hierarchy_df[["roadway", "hierarchy"]],
            how = "left",
            on = "roadway"
        )

        # sort on hierarchy (ascending), 
        # drive_access(descending), 
        # bike_access(descending), 
        # walk_access(descending), 
        # length(ascending)

        non_unique_AB_links_sorted_df =  non_unique_AB_links_df.sort_values(
            by = ["hierarchy", "drive_access", "bike_access", "walk_access", "length"],
            ascending = [True, False, False, False, True]
        )

        unique_AB_links_df = non_unique_AB_links_sorted_df.drop_duplicates(
            subset = ["u", "v"], 
            keep = "first"
        )

        from_list = non_unique_AB_links_df.shstReferenceId.tolist()
        to_list = unique_AB_links_df.shstReferenceId.tolist()

        drop_link_model_link_id_list = [c for c in from_list if c not in to_list]

        self.links_df = self.links_df[
            ~self.links_df.shstReferenceId.isin(drop_link_model_link_id_list)
        ]

        self.shapes_df = self.shapes_df[self.shapes_df.id.isin(self.links_df.id)].copy()

    def _link_node_numbering(
        self,
        link_numbering_dictionary: Optional[dict] = {},
        node_numbering_dictionary: Optional[dict] = {},
    ):
        """
        numbering links and nodes according to county rules
        """

        RanchLogger.info("Numbering links and nodes using county based rules")

        if link_numbering_dictionary:
            county_link_range = link_numbering_dictionary
        else:
            county_link_range = self.parameters.county_link_range
        
        if node_numbering_dictionary:
            county_node_range = node_numbering_dictionary
        else:
            county_node_range = self.parameters.county_node_range

        self.nodes_df["model_node_id"] = self.nodes_df.groupby(["county"]).cumcount()

        self.nodes_df["county_numbering_start"] = self.nodes_df["county"].apply(
            lambda x: county_node_range[x]['start']
        )

        self.nodes_df["model_node_id"] = self.nodes_df["model_node_id"] + self.nodes_df["county_numbering_start"]

        node_osm_model_id_dict = dict(zip(self.nodes_df.osm_node_id, self.nodes_df.model_node_id))

        self.links_df["model_link_id"] = self.links_df.groupby(["county"]).cumcount()

        self.links_df["county_numbering_start"] = self.links_df["county"].apply(
            lambda x: county_link_range[x]['start']
        )

        self.links_df["model_link_id"] = self.links_df["model_link_id"] + self.links_df["county_numbering_start"]

        self.links_df['A'] = self.links_df['u'].map(node_osm_model_id_dict)

        self.links_df['B'] = self.links_df['v'].map(node_osm_model_id_dict)

    def build_centroid_connectors(
        self,
        build_taz_drive: bool = True,
        build_taz_active_modes: bool = True,
        build_maz_drive: bool = False,
        build_maz_active_modes: bool = False,
        input_taz_polygon_file: str = None,
        input_taz_node_file: str = None,
        input_maz_polygon_file: str = None,
        input_maz_node_file: str = None,
        taz_unique_id: str = None,
        maz_unique_id: str = None,
    ):

        """
        build centroid connectors

        Inputs:
            RoadwayNetwork
            build_taz_drive: Boolean, true when building TAZ drive connectors
            build_taz_active_modes: Boolean, true when building TAZ bike and walk connectors
            build_maz_drive: Boolean, true when building MAZ drive connectors
            build_maz_drive: Boolean, true when building MAZ bike and walk connectors
            input_taz_polygon_file: polygon shapefile or geojson for TAZ boudnaries
            input_taz_node_file: node shapefile or geojson for TAZ centroid
            input_maz_polygon_file: polygon shapefile or geojson for MAZ boudnaries
            input_maz_node_file: node shapefile or geojson for MAZ centroid     
        """

        if (build_taz_drive) | (build_taz_active_modes):
            if input_taz_polygon_file is None:
                msg = "Missing taz polygon file, specify using input_taz_polygon_file"
                RanchLogger.error(msg)
                raise ValueError(msg)
            else:
                filename, file_extension = os.path.splitext(input_taz_polygon_file)
                if file_extension in [".shp", ".geojson"]:
                    taz_polygon_gdf = gpd.read_file(input_taz_polygon_file)
                    if taz_polygon_gdf.crs is None:
                        msg = "Input file {} missing CRS".format(input_taz_polygon_file)
                        RanchLogger.error(msg)
                        raise ValueError(msg)
                    else:
                        RanchLogger.info("input file {} has crs : {}".format(input_taz_polygon_file, taz_polygon_gdf.crs))
                    if taz_unique_id is None:
                        taz_polygon_gdf['taz_id'] = range(1, 1+len(taz_polygon_gdf))
                    elif taz_unique_id not in taz_polygon_gdf.columns:
                        msg = "Input file {} does not have unique ID {}".format(input_taz_polygon_file, taz_unique_id)
                        RanchLogger.error(msg)
                        raise ValueError(msg)
                    else:
                        None
                else:
                    msg = "Invalid network file {}, should be .shp or .geojson".format(input_taz_polygon_file)
                    RanchLogger.error(msg)
                    raise ValueError(msg)
        
            if input_taz_node_file is not None:
                filename, file_extension = os.path.splitext(input_taz_node_file)
                if file_extension in [".shp", ".geojson"]:
                    taz_node_gdf = gpd.read_file(input_taz_node_file)
                    if taz_node_gdf.crs is None:
                        msg = "Input file {} missing CRS".format(input_taz_node_file)
                        RanchLogger.error(msg)
                        raise ValueError(msg)
                    else:
                        RanchLogger.info("input file {} has crs : {}".format(input_taz_node_file, taz_node_gdf.crs))
                    if taz_unique_id not in taz_node_gdf.columns:
                        msg = "Input file {} does not have unique ID {}".format(input_taz_node_file, taz_unique_id)
                        RanchLogger.error(msg)
                        raise ValueError(msg)
                else:
                    msg = "Invalid network file {}, should be .shp or .geojson".format(input_taz_node_file)
                    RanchLogger.error(msg)
                    raise ValueError(msg)

            else:
                RanchLogger.info("Missing taz node file, will use input taz polygon centroid")

                taz_node_gdf = taz_polygon_gdf.copy()
                taz_node_gdf['geometry'] = taz_node_gdf['geometry'].representative_point()

            # convert to lat-long
            taz_polygon_gdf = taz_polygon_gdf.to_crs(epsg = 4269)
            taz_node_gdf = taz_node_gdf.to_crs(epsg = 4269)
            
            if 'model_node_id' not in taz_node_gdf.columns:
                self.assign_model_node_id_to_taz(taz_node_gdf)

            if build_taz_drive:
                self.build_taz_drive_connector(
                    taz_polygon_gdf
                    )
            if build_maz_active_modes:
                self.build_taz_active_modes_connector(
                    taz_polygon_gdf
                )

        if (build_maz_drive) | (build_maz_active_modes):
            if input_maz_polygon_file is None:
                msg = "Missing maz polygon file, specify using input_maz_polygon_file"
                RanchLogger.error(msg)
                raise ValueError(msg)
            else:
                filename, file_extension = os.path.splitext(input_maz_polygon_file)
                if file_extension in [".shp", ".geojson"]:
                    maz_polygon_gdf = gpd.read_file(input_maz_polygon_file)
                    if maz_polygon_gdf.crs is None:
                        msg = "Input file {} missing CRS".format(input_maz_polygon_file)
                        RanchLogger.error(msg)
                        raise ValueError(msg)
                    else:
                        RanchLogger.info("input file {} has crs : {}".format(input_maz_polygon_file, maz_polygon_gdf.crs))
                else:
                    msg = "Invalid network file {}, should be .shp or .geojson".format(input_maz_polygon_file)
                    RanchLogger.error(msg)
                    raise ValueError(msg)
        
            if input_maz_node_file is not None:
                filename, file_extension = os.path.splitext(input_maz_node_file)
                if file_extension in [".shp", ".geojson"]:
                    maz_node_gdf = gpd.read_file(input_maz_node_file)
                    if maz_node_gdf.crs is None:
                        msg = "Input file {} missing CRS".format(input_maz_node_file)
                        RanchLogger.error(msg)
                        raise ValueError(msg)
                    else:
                        RanchLogger.info("input file {} has crs : {}".format(input_maz_node_file, maz_node_gdf.crs))
                else:
                    msg = "Invalid network file {}, should be .shp or .geojson".format(input_maz_node_file)
                    RanchLogger.error(msg)
                    raise ValueError(msg)

            else:
                RanchLogger.info("Missing maz node file, will use input maz polygon centroid")

                maz_node_gdf = maz_polygon_gdf['geometry'].representative_centroids()

            # convert to lat-long
            maz_polygon_gdf = maz_polygon_gdf.to_crs(self.parameters.standard_crs)
            maz_node_gdf = maz_node_gdf.to_crs(self.parameters.standard_crs)

            if build_taz_drive:
                self.build_taz_drive_connector(
                    taz_node_gdf, 
                    taz_polygon_gdf
                    )
            if build_maz_active_modes:
                self.build_taz_active_modes_connector(
                    taz_node_gdf, 
                    taz_polygon_gdf
                )

    def assign_model_node_id_to_taz(
        self,
        taz_node_gdf
    ):
        """
        attribute taz node with county, model_node_id, drive_access, walk_access, bike_access
        """
        
        # add county
        taz_node_gdf = gpd.sjoin(
            taz_node_gdf,
            self.county_gdf[['geometry', self.county_variable_name]],
            how = 'left',
            op = 'within'
        )

        taz_node_gdf.rename(columns = {self.county_variable_name : 'county'}, inplace = True)
        taz_node_gdf['county'].fillna('external', inplace = True)

        # assign model node id based on county taz rules
        taz_node_gdf["model_node_id"] = taz_node_gdf.groupby(["county"]).cumcount()

        taz_node_gdf["county_numbering_start"] = taz_node_gdf["county"].apply(
            lambda x: self.parameters.county_taz_range[x]['start']
        )

        taz_node_gdf["model_node_id"] = taz_node_gdf["model_node_id"] + taz_node_gdf["county_numbering_start"]

        taz_node_gdf['drive_access'] = 1
        taz_node_gdf['walk_access'] = 1
        taz_node_gdf['bike_access'] = 1

        self.taz_node_gdf = taz_node_gdf

    def build_taz_drive_connector(
        self,
        taz_polygon_df: GeoDataFrame,
        num_connectors_per_centroid: int = 4
    ):
        """
        build taz drive centroid connectors
        """

        # step 1
        # find nodes that have only two assignable/drive 
        # geometries (not reference)

        node_two_geometry_df = self.get_non_intersection_drive_nodes()

        # step 2
        # for each zone, find nodes that have only two assignable/drive 
        # geometries (not reference) - good intersections

        taz_good_intersection_df = Roadway.get_nodes_in_zones(
            node_two_geometry_df,
            taz_polygon_df
        )

        # step 3
        # for zones with fewer than 4 good intersections
        # use drive nodes
        exclude_links_df = self.links_df[
            self.links_df.roadway.isin(
                ["motorway_link", "motorway", "trunk", "trunk_link"]
            )
        ].copy()

        drive_node_gdf = self.nodes_df[
            (self.nodes_df.drive_access == 1) & 
            ~(self.nodes_df.osm_node_id.isin(
                exclude_links_df.u.tolist() + exclude_links_df.v.tolist()))
        ].copy()

        taz_drive_node_df = Roadway.get_nodes_in_zones(
            drive_node_gdf,
            taz_polygon_df
        )

        # step 4
        # choose nodes for taz
        taz_centroid_gdf = self.taz_node_gdf.copy()
        taz_loading_node_df = Roadway.get_taz_loading_nodes(
            taz_centroid_gdf,
            taz_good_intersection_df,
            taz_drive_node_df,
            num_connectors_per_centroid
        )
        
        # step 5
        # create links and shapes between taz and chosen nodes
        taz_cc_shape_gdf = generate_centroid_connectors_shape(
            taz_loading_node_df
        )
        #taz_cc_shape_gdf = taz_cc_shape_gdf.drop(['ld_point', 'c_point'], axis = 1)
        
        #taz_cc_shape_gdf = gpd.GeoDataFrame(
        #    taz_cc_shape_gdf,
        #    crs = self.parameters.standard_crs
        #)

        join_gdf = taz_cc_shape_gdf.copy()
        join_gdf['geometry'] = join_gdf['geometry'].centroid

        join_gdf = gpd.sjoin(
            join_gdf,
            self.county_gdf[['geometry', self.county_variable_name]],
            how = 'left',
            op = 'within'
        )

        taz_cc_shape_gdf['county'] = join_gdf[self.county_variable_name]

        taz_cc_link_gdf = generate_centroid_connectors_link(
            taz_cc_shape_gdf
        )

        taz_cc_link_gdf['roadway'] = 'taz'
        taz_cc_link_gdf['drive_access'] = 1
        taz_cc_link_gdf['bike_access'] = 1
        taz_cc_link_gdf['walk_access'] = 1

        cc_link_columns_list = ["A", "B", "drive_access", "walk_access", "bike_access", 
                            "shstGeometryId", "id", "u", "v", "fromIntersectionId", "toIntersectionId", 'county', 'roadway', 'geometry']
        taz_cc_link_gdf = taz_cc_link_gdf[cc_link_columns_list].copy()
    
        cc_shape_columns_list = ["id", "geometry", "fromIntersectionId", 'county'] # no 'noIntersectionId' because shape is only in the inbound direction
        taz_cc_shape_gdf = taz_cc_shape_gdf[cc_shape_columns_list].drop_duplicates(subset = ["id"]).copy()

        self.taz_cc_shape_gdf = taz_cc_shape_gdf
        self.taz_cc_link_gdf = taz_cc_link_gdf

    def get_non_intersection_drive_nodes(
        self
    ):

        """
        return nodes that have only two drivable geometries
        """
        drive_links_gdf = self.links_df[
            ~(self.links_df.roadway.isin(
                ["motorway_link", "motorway", "trunk", "trunk_link", "service"]
                )
            ) &
            (self.links_df.drive_access == 1)
        ].copy()

        a_geometry_count_df = drive_links_gdf.groupby(
            ["u", "shstGeometryId"]
        )["shstReferenceId"].count().reset_index().rename(
            columns = {"u" : "osm_node_id"}
        )

        b_geometry_count_df = drive_links_gdf.groupby(
            ["v", "shstGeometryId"]
        )["shstReferenceId"].count().reset_index().rename(
            columns = {"v" : "osm_node_id"}
        )

        node_geometry_count_df = pd.concat(
            [a_geometry_count_df, b_geometry_count_df], 
            ignore_index = True, 
            sort = False
            )

        node_geometry_count_df = node_geometry_count_df.groupby(
            ["osm_node_id", "shstGeometryId"]
        ).count().reset_index().groupby(
            ["osm_node_id"]
        )["shstGeometryId"].count().reset_index()

        node_two_geometry_df = node_geometry_count_df[
            node_geometry_count_df.shstGeometryId == 2
        ].copy()
        
        node_two_geometry_df = self.nodes_df[
            self.nodes_df.osm_node_id.isin(
                node_two_geometry_df.osm_node_id.tolist())
        ].copy()

        return node_two_geometry_df

    def get_nodes_in_zones(
        nodes_gdf,
        zones_gdf
    ):
        """
        return nodes and the zones they are in

        Input:
            nodes_gdf: nodes geo data frame, points
            zones_gdf: zones geo data frame, polygons
        """
        polygon_buffer_gdf = zones_gdf.copy()

        polygon_buffer_gdf["geometry_buffer"] = polygon_buffer_gdf["geometry"].apply(
            lambda x: buffer1(x)
        )
        polygon_buffer_gdf.rename(
            columns = {"geometry" : "geometry_orig", 
            "geometry_buffer" : "geometry"}, 
            inplace = True
        )
        nodes_in_zones_gdf = gpd.sjoin(
            nodes_gdf, 
            polygon_buffer_gdf[["geometry", "taz_id"]], 
            how = "left", 
            op = "intersects"
        )

        return nodes_in_zones_gdf

    def get_taz_loading_nodes(
        taz_centroid_df,
        good_intersection_df,
        drive_nodes_df,
        num_connectors_per_centroid
    ):
        """
        for each zone, return chosen loading point
        """

        good_intersection_df['preferred'] = 1
        drive_nodes_df['preferred'] = 0

        all_load_nodes_df = pd.concat(
            [good_intersection_df, drive_nodes_df],
            sort = False,
            ignore_index=True
        )
        
        all_load_nodes_df = all_load_nodes_df[
            all_load_nodes_df['taz_id'].notnull()
        ]

        all_load_nodes_df["ld_point"] = all_load_nodes_df["geometry"].apply(
            lambda x: list(x.coords)[0]
        )
        taz_centroid_df["c_point"] = taz_centroid_df["geometry"].apply(
            lambda x: list(x.coords)[0]
        )

        all_load_nodes_df = pd.merge(
            all_load_nodes_df.drop('geometry', axis = 1),
            taz_centroid_df.drop('geometry', axis = 1),
            how = 'left',
            on = ['taz_id']
        )

        all_load_nodes_df['distance'] = all_load_nodes_df.apply(
            lambda x: haversine_distance(
                list(x.ld_point), list(x.c_point)
            ),
            axis = 1
        )
        
        # sort on preferred, distance
        all_load_nodes_df.sort_values(
            by = ['preferred', 'distance'],
            ascending = [False, True],
            inplace=True
        )

        all_load_nodes_df.drop_duplicates(
            subset = ['osm_node_id', 'taz_id'],
            inplace = True
        )

        all_load_nodes_df = get_non_near_connectors(
            all_load_nodes_df, 
            num_connectors_per_centroid,
            'taz_id')

        return all_load_nodes_df

    def standard_format(
        self,
        county_boundary_file: str,
        county_variable_name: str
    ):
        """
        pipeline step 8 create and write out roadway in standard format

        1. link and node numbering
        2. add locationReferences
        """

        # calculate county in case new links and nodes are generated

        if not county_boundary_file:
            msg = "Missing polygon file for county boundary."
            RanchLogger.error(msg)
            raise ValueError(msg)

        if county_boundary_file:
            filename, file_extension = os.path.splitext(county_boundary_file)
            if file_extension in [".shp", ".geojson"]:
                county_gdf = gpd.read_file(county_boundary_file)
        
            else:
                msg = "Invalid boundary file, should be .shp or .geojson"
                RanchLogger.error(msg)
                raise ValueError(msg)

        RanchLogger.info("Starting Step 8 creating roadway standard format")

        ## 1. join county name to shapes and nodes
        #self._calculate_county(
        #    county_gdf = county_gdf,
        #    county_variable_name = county_variable_name
        #)
        """
        ## add unique ranch link / node ID, based on coordinates
        self.nodes_df['ranch_node_id'] = np.where(
            self.nodes_df['shst_node_id'].notnull(),
            self.nodes_df['shst_node_id'],
            self.nodes_df['geometry'].apply(
                lambda x: create_unique_node_id(x)
                )
        )

        RanchLogger.info('{} nodes has {} unique osm ids, {} unique ranch ids'.format(
            len(self.nodes_df), 
            self.nodes_df.osm_node_id.nunique(),
            self.nodes_df.ranch_node_id.nunique()
            )
        )

        node_osm_ranch_dict = dict(zip(self.nodes_df.osm_node_id, self.nodes_df.ranch_node_id))

        self.shapes_df['ranch_shape_id'] = np.where(
            self.shapes_df['id'].notnull(),
            self.shapes_df['id'],
            self.shapes_df['geometry'].apply(
                lambda x: create_unique_shape_id(x)
            )
        )

        RanchLogger.info('{} shapes has {} unique shst id {} unique ranch ids'.format(
            len(self.shapes_df), 
            self.shapes_df.id.nunique(),
            self.shapes_df.ranch_shape_id.nunique()
            )
        )

        RanchLogger.info('{} links has {} unique shst ids'.format(
            len(self.links_df), self.links_df.shstReferenceId.nunique()
            )
        )

        self.links_df['ranch_link_id'] = np.where(
            self.links_df['shstReferenceId'].notnull(),
            self.links_df['shstReferenceId'],
            self.links_df.apply(
                lambda x: create_unique_link_id(
                    self.nodes_df[
                        self.nodes_df['osm_node_id'] == x.u
                    ]['geometry'].iloc[0],
                    self.nodes_df[
                        self.nodes_df['osm_node_id'] == x.v
                    ]['geometry'].iloc[0],
                ),
                axis =1
            )
        )

        RanchLogger.info('{} links has {} unique ranch ids'.format(
            len(self.links_df), self.links_df.ranch_link_id.nunique()
            )
        )

        self.links_df['p'] = self.links_df['u'].map(node_osm_ranch_dict)
        self.links_df['q'] = self.links_df['v'].map(node_osm_ranch_dict)
        """
        ## 2. add length in feet

        self.links_df = Roadway._add_length_in_feet(self.links_df)

        ## 3. add locationReferences to links

        self.links_df= Roadway._add_location_references(self.nodes_df,self.links_df)

        ## 4. add model link / node ID following county rules

        #self._link_node_numbering()

    def _add_length_in_feet(
        linestring_gdf: GeoDataFrame,
        variable: str = 'length'
    ):
        """
        add length in feet to line geodataframe
        """

        geom_length = linestring_gdf[['geometry']].copy()
        geom_length = geom_length.to_crs(epsg = 26915)
        geom_length[variable] = geom_length.length

        linestring_gdf[variable] = geom_length[variable]

        return linestring_gdf

    def _add_location_references(
        nodes: GeoDataFrame,
        links: GeoDataFrame,
        variable: str = 'locationReferences'
    ):
        """
        add location reference to links
        """
        node_gdf = nodes.copy()
        link_gdf = links.copy()

        node_gdf['X'] = node_gdf['geometry'].apply(lambda p: p.x)
        node_gdf['Y'] = node_gdf['geometry'].apply(lambda p: p.y)
        node_gdf['point'] = [list(xy) for xy in zip(node_gdf.X, node_gdf.Y)]
        node_dict = dict(zip(node_gdf.model_node_id, node_gdf.point))
    
        link_gdf['A_point'] = link_gdf['A'].map(node_dict)
        link_gdf['B_point'] = link_gdf['B'].map(node_dict)

        link_gdf[variable] = link_gdf.apply(
            lambda x: [
                {
                    'sequence':1, 
                    'point': x['A_point'],
                    'distanceToNextRef':x['length'],
                    'intersectionId':x['fromIntersectionId']
                },
                {
                    'sequence':2,
                    'point': x['B_point'],
                    'intersectionId':x['toIntersectionId']
                }
            ],
            axis = 1
        )

        links[variable] = link_gdf[variable]

        return links