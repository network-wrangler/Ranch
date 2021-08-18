from .logger import RanchLogger

import os

import geopandas as gpd
import json
import networkx as nx
import pandas as pd
import osmnx as ox
from pyproj import CRS

from .utils import link_df_to_geojson, point_df_to_geojson

__all__ = ["run_osmnx_extraction"]

def run_osmnx_extraction(
    input_polygon_file: str,
    output_dir: str
):

    """
    run osmnx extraction with input polygon

    Args:
        input_area_polygon_file: input polygon file that defines the extract region
        output_extraction_file: output file that stores the extraction
    """

    if not input_polygon_file:
        msg = "Missing polygon file for sharedstreet extraction."
        RanchLogger.error(msg)
        raise ValueError(msg)

    if not output_dir:
        msg = "Please specify output filename for extraction result."
        RanchLogger.error(msg)
        raise ValueError(msg)

    if input_polygon_file:
        filename, file_extension = os.path.splitext(input_polygon_file)
        if file_extension in [".shp", ".geojson"]:
            polygon_gdf = gpd.read_file(input_polygon_file)
        
        else:
            msg = "Invalid boundary file, should be .shp or .geojson"
            RanchLogger.error(msg)
            raise ValueError(msg)

    # convert to lat-long
    polygon_gdf = polygon_gdf.to_crs(CRS("EPSG:4269"))

    boundary = polygon_gdf.geometry.unary_union

    # OSM extraction
    G_drive = ox.graph_from_polygon(boundary, network_type='all', simplify=False)

    link_gdf = ox.graph_to_gdfs(G_drive, nodes = False, edges = True)
    node_gdf = ox.graph_to_gdfs(G_drive, nodes = True, edges = False)

    # write out osm extraction
    link_prop = link_gdf.drop("geometry", axis = 1).columns.tolist()
    link_geojson = link_df_to_geojson(link_gdf, link_prop)

    with open(os.path.join(output_dir, "link.geojson"), "w") as f:
        json.dump(link_geojson, f)

    node_prop = node_gdf.drop("geometry", axis = 1).columns.tolist()
    node_geojson = point_df_to_geojson(node_gdf, node_prop)

    with open(os.path.join(output_dir, "node.geojson"), "w") as f:
        json.dump(node_geojson, f)
