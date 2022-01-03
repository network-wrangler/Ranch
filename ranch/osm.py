import json
import os

from typing import Union

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from pyproj import CRS

from .logger import RanchLogger
from .parameters import standard_crs, alt_standard_crs
from .utils import link_df_to_geojson, point_df_to_geojson

__all__ = ["run_osmnx_extraction"]


def run_osmnx_extraction(
    input_polygon_file: Union[GeoDataFrame, str],
    output_dir: str
):
    """
    run osmnx extraction with input polygon

    Args:
        input_area_polygon_file: input polygon file that defines the extract region
        output_extraction_file: output file that stores the extraction
    """

#    if not input_polygon_file:
#        msg = "Missing polygon file for sharedstreet extraction."
#        RanchLogger.error(msg)
#        raise ValueError(msg)

    if not isinstance(input_polygon_file, (str, GeoDataFrame)):
        msg = "Polygon input must be a file path or a GeoDataFrame"
        RanchLogger.error(msg)
        raise ValueError(msg)


    if not output_dir:
        msg = "Please specify output filename for extraction result."
        RanchLogger.error(msg)
        raise ValueError(msg)

    if isinstance(input_polygon_file, str):
        filename, file_extension = os.path.splitext(input_polygon_file)
        if file_extension in [".shp", ".geojson"]:
            polygon_gdf = gpd.read_file(input_polygon_file)

        else:
            msg = "Invalid boundary file, should be .shp or .geojson"
            RanchLogger.error(msg)
            raise ValueError(msg)
    elif isinstance(input_polygon_file, GeoDataFrame):
        # No need to create a copy, since the to_crs line in the
        # below will return a copy.
        polygon_gdf = input_polygon_file


    # avoid conversion between WGS lat-long and NAD lat-long
    if polygon_gdf.crs == alt_standard_crs:
        polygon_gdf.crs = standard_crs

    # convert to lat-long
    polygon_gdf = polygon_gdf.to_crs(standard_crs)

    boundary = polygon_gdf.geometry.unary_union

    # OSM extraction
    G_drive = ox.graph_from_polygon(boundary, network_type="all", simplify=False)

    link_gdf = ox.graph_to_gdfs(G_drive, nodes=False, edges=True)
    node_gdf = ox.graph_to_gdfs(G_drive, nodes=True, edges=False)

    # write out osm extraction
    link_prop = link_gdf.drop("geometry", axis=1).columns.tolist()
    link_geojson = link_df_to_geojson(link_gdf, link_prop)

    with open(os.path.join(output_dir, "link.geojson"), "w") as f:
        json.dump(link_geojson, f)

    node_prop = node_gdf.drop("geometry", axis=1).columns.tolist()
    node_geojson = point_df_to_geojson(node_gdf, node_prop)

    with open(os.path.join(output_dir, "node.geojson"), "w") as f:
        json.dump(node_geojson, f)

    return link_geojson, node_geojson


def add_two_way_osm(link_gdf, osmnx_link):
    """
    for osm with oneway = False, add the reverse direction to complete

    Parameters
    ------------
    osm link from shst extraction, plus shst info

    return
    ------------
    complete osm link
    """
    osm_link_gdf = link_gdf.copy()
    osm_link_gdf["wayId"] = osm_link_gdf["wayId"].astype(int)
    osm_link_gdf.drop("name", axis=1, inplace=True)

    osmnx_link_gdf = osmnx_link.copy()

    osmnx_link_gdf.drop_duplicates(subset=["osmid"], inplace=True)
    osmnx_link_gdf.drop(["length", "u", "v", "geometry"], axis=1, inplace=True)

    RanchLogger.info(
        "shst extraction has {} geometries".format(osm_link_gdf.id.nunique())
    )

    RanchLogger.info("shst extraction has {} osm links".format(osm_link_gdf.shape[0]))

    osm_link_gdf["u"] = osm_link_gdf.nodeIds.apply(lambda x: int(x[0]))
    osm_link_gdf["v"] = osm_link_gdf.nodeIds.apply(lambda x: int(x[-1]))

    RanchLogger.info("---joining osm shst with osmnx data---")
    osm_link_gdf = pd.merge(
        osm_link_gdf, osmnx_link_gdf, left_on=["wayId"], right_on=["osmid"], how="left"
    )

    reverse_osm_link_gdf = osm_link_gdf[
        (osm_link_gdf.oneWay == False)
        & (osm_link_gdf.forwardReferenceId != osm_link_gdf.backReferenceId)
        & (osm_link_gdf.u != osm_link_gdf.v)
    ].copy()

    RanchLogger.info(
        "shst extraction has {} two-way osm links".format(reverse_osm_link_gdf.shape[0])
    )
    RanchLogger.info(
        "and they are {} geometrys".format(reverse_osm_link_gdf.id.nunique())
    )

    reverse_osm_link_gdf.rename(
        columns={
            "u": "v",
            "v": "u",
            "forwardReferenceId": "backReferenceId",
            "backReferenceId": "forwardReferenceId",
            "fromIntersectionId": "toIntersectionId",
            "toIntersectionId": "fromIntersectionId",
        },
        inplace=True,
    )

    reverse_osm_link_gdf["reverse_out"] = 1

    osm_link_gdf = pd.concat(
        [osm_link_gdf, reverse_osm_link_gdf], sort=False, ignore_index=True
    )

    osm_link_gdf.rename(
        columns={
            "forwardReferenceId": "shstReferenceId",
            "geometryId": "shstGeometryId",
        },
        inplace=True,
    )

    osm_link_gdf.drop("backReferenceId", axis=1, inplace=True)

    RanchLogger.info(
        "after join, ther are {} osm links from shst extraction, \
    out of which there are {} links that do not have osm info, \
    due to shst extraction (default tile 181224) contains {} osm ids that are not included in latest OSM extraction, \
    e.g. private streets, closed streets.".format(
            len(osm_link_gdf),
            len(osm_link_gdf[osm_link_gdf.osmid.isnull()]),
            osm_link_gdf[osm_link_gdf.osmid.isnull()].wayId.nunique(),
        )
    )

    RanchLogger.info(
        "after join, there are {} shst referencies".format(
            osm_link_gdf.groupby(["shstReferenceId", "shstGeometryId"]).count().shape[0]
        )
    )

    return osm_link_gdf


def highway_attribute_list_to_value(x, highway_to_roadway_dict, roadway_hierarchy_dict):
    """
    clean up osm highway, and map to standard roadway
    """
    if type(x.highway) == list:
        value_list = list(set([highway_to_roadway_dict[c] for c in x.highway]))
        if len(value_list) == 1:
            if value_list[0] != "":
                return value_list[0]
            else:
                if type(x.roadClass) == list:
                    return highway_to_roadway_dict[x.roadClass[0].lower()]
                else:
                    return highway_to_roadway_dict[x.roadClass.lower()]

        else:
            ret_val = value_list[0]
            ret_val_level = roadway_hierarchy_dict[ret_val]
            for c in value_list:
                val_level = roadway_hierarchy_dict[c]
                if val_level < ret_val_level:
                    ret_val = c
                    ret_val_level = val_level
                else:
                    continue
            return ret_val
    else:
        if x.highway == "":
            return highway_to_roadway_dict[x.roadClass.lower()]
        else:
            return highway_to_roadway_dict[x.highway]
