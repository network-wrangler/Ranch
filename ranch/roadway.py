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
from .utils import fill_na, identify_dead_end_nodes
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
        # self._link_node_numbering()

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

        self.links_df["model_link_id"] = self.links_df.groupby(["county"]).cumcount()

        self.links_df["county_numbering_start"] = self.links_df["county"].apply(
            lambda x: county_link_range[x]['start']
        )

        self.links_df["model_link_id"] = self.links_df["model_link_id"] + self.links_df["county_numbering_start"]