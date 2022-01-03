import glob
import os
from typing import Optional

import docker
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame
from pyproj import CRS

from .logger import RanchLogger
from .parameters import standard_crs

__all__ = ["run_shst_extraction"]


def run_shst_extraction(input_polygon_file: str, output_dir: str):

    """
    run sharedstreet extraction with input polygon

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
    polygon_gdf = polygon_gdf.to_crs(standard_crs)

    # export polygon to geojson for shst node js
    for i in range(len(polygon_gdf.geometry)):

        RanchLogger.info(
            "Exporting boundry file {}".format(
                os.path.join(output_dir, "boundary." + str(i) + ".geojson")
            )
        )

        boundary_gdf = gpd.GeoDataFrame(
            {"geometry": gpd.GeoSeries(polygon_gdf.geometry.iloc[i])}
        )

        boundary_gdf.to_file(
            os.path.join(output_dir, "boundary." + str(i) + ".geojson"),
            driver="GeoJSON",
        )

        RanchLogger.info("extracting for polygon {}".format(i))

        _run_shst_extraction(
            input_file_name="boundary." + str(i), output_dir=output_dir
        )


def _run_shst_extraction(
    input_file_name: str,
    output_dir: str,
):

    """
    actual call shst extraction in container
    """

    client = docker.from_env()

    container = client.containers.run(
        image="shst:latest",
        detach=True,
        volumes={output_dir: {"bind": "/usr/node", "mode": "rw"}},
        tty=True,
        auto_remove=True,
        command="/bin/bash",
    )

    container.exec_run(
        cmd=(
            "shst extract usr/node/"
            + input_file_name
            + ".geojson --out=usr/node/"
            + "extract."
            + input_file_name
            + ".geojson --metadata --tile-hierarchy=8 --tiles"
        )
    )

    container.stop()


def run_shst_match(
    input_network_file: str,
    output_dir: Optional[str] = None,
    input_crs: Optional[CRS] = None,
    input_unqiue_id: Optional[list] = None,
    custom_match_option: Optional[str] = None,
):

    """
    run sharedstreet match with input network

    Args:
        input_network_file: input network file that needs to be conflated
        output_match_file: output file that stores the match
    """

    if not input_network_file:
        msg = "Missing network file for sharedstreets match."
        RanchLogger.error(msg)
        raise ValueError(msg)

    if not output_dir:
        msg = "No output directory specified for match result, will write out to the input network directory."
        RanchLogger.warning(msg)
        output_dir = os.path.dirname(input_network_file)
    else:
        output_dir = output_dir

    if input_network_file:
        filename, file_extension = os.path.splitext(input_network_file)
        if file_extension in [".shp", ".geojson"]:
            network_gdf = gpd.read_file(input_network_file)
            RanchLogger.info(
                "input network {} has crs : {}".format(
                    input_network_file, network_gdf.crs
                )
            )

        else:
            msg = "Invalid network file, should be .shp or .geojson"
            RanchLogger.error(msg)
            raise ValueError(msg)

    # set input crs
    if input_crs:
        network_gdf.crs = input_crs

    # convert to lat-long
    network_gdf = network_gdf.to_crs(standard_crs)

    # check if input network has unique IDs
    if input_unqiue_id:
        filename = input_network_file
        filename = (
            os.path.splitext(input_network_file)[0].replace("\\", "/").split("/")[-1]
        )

    # if not, create unique IDs
    else:
        RanchLogger.info(
            "Input network for shst match does not have unique IDs, generating unique IDs"
        )
        network_gdf["unique_id"] = range(1, 1 + len(network_gdf))

        RanchLogger.info(
            ("Generated {} unique IDs for {} links in the input network").format(
                network_gdf["unique_id"].nunique(), len(network_gdf)
            )
        )

        # export network to geojson for shst node js

        filename = (
            os.path.splitext(input_network_file)[0].replace("\\", "/").split("/")[-1]
        )

        RanchLogger.info(
            "Exporting shst match input - ID-ed geometry file {}".format(
                os.path.join(output_dir, filename + ".geojson")
            )
        )

        network_gdf[["unique_id", "geometry"]].to_file(
            os.path.join(output_dir, filename + ".geojson"), driver="GeoJSON"
        )

        RanchLogger.info(
            "Exporting ID-ed network file {}".format(
                os.path.join(output_dir, filename + ".full.geojson")
            )
        )

        network_gdf.to_file(
            os.path.join(output_dir, filename + ".full.geojson"), driver="GeoJSON"
        )

    if custom_match_option:
        match_option = custom_match_option
    else:
        match_option = "--follow-line-direction --tile-hierarchy=8"

    _run_shst_match(
        input_file_name=filename,
        output_dir=output_dir,
        match_option=match_option,
    )


def _run_shst_match(
    input_file_name: str,
    output_dir: str,
    match_option: str,
):

    """
    actual call shst match in container
    """

    client = docker.from_env()

    container = client.containers.run(
        image="shst:latest",
        detach=True,
        volumes={output_dir: {"bind": "/usr/node", "mode": "rw"}},
        tty=True,
        auto_remove=True,
        command="/bin/bash",
    )

    container.exec_run(
        cmd=(
            "shst match usr/node/"
            + input_file_name
            + ".geojson --out=usr/node/"
            + "match."
            + input_file_name
            + ".geojson "
            + match_option
        )
    )

    container.stop()


def read_shst_extraction(path, suffix):
    """
    read all shst extraction geojson file
    """
    shst_gdf = DataFrame()

    shst_file = glob.glob(path + "**/" + suffix, recursive=True)
    RanchLogger.info("----------start reading shst extraction data-------------")
    for i in shst_file:
        RanchLogger.info("reading shst extraction data : {}".format(i))
        new = gpd.read_file(i)
        new["source"] = i
        shst_gdf = pd.concat([shst_gdf, new], ignore_index=True, sort=False)
    RanchLogger.info("----------finished reading shst extraction data-------------")

    return shst_gdf


def extract_osm_link_from_shst_extraction(shst_extraction_df: GeoDataFrame):
    """
    expand each shst extract record into osm segments
    """
    osm_from_shst_link_list = []

    temp = shst_extraction_df.apply(
        lambda x: _extract_osm_link_from_shst_extraction(x, osm_from_shst_link_list),
        axis=1,
    )

    osm_from_shst_link_df = pd.concat(osm_from_shst_link_list)

    # get complete shst info
    osm_from_shst_link_df = pd.merge(
        osm_from_shst_link_df,
        shst_extraction_df.drop(["roadClass", "metadata", "source"], axis=1),
        how="left",
        left_on="geometryId",
        right_on="id",
    )

    return osm_from_shst_link_df


def _extract_osm_link_from_shst_extraction(
    row: pd.Series,
    osm_from_shst_link_list: list,
):
    link_df = DataFrame(row.get("metadata").get("osmMetadata").get("waySections"))
    link_df["geometryId"] = row.get("metadata").get("geometryId")

    osm_from_shst_link_list.append(link_df)
