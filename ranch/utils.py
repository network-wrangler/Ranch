import glob
import hashlib
import json
import math
from functools import partial
from typing import List

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj

from pandas.core.algorithms import unique
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from pyproj import CRS
from shapely.ops import transform

from .logger import RanchLogger
from .parameters import Parameters, standard_crs


def link_df_to_geojson(df, properties):
    """
    Author: Geoff Boeing:
    https://geoffboeing.com/2015/10/exporting-python-data-geojson/
    """
    geojson = {"type": "FeatureCollection", "features": []}
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": []},
        }
        feature["geometry"]["coordinates"] = [
            [x, y] for (x, y) in list(row["geometry"].coords)
        ]
        for prop in properties:
            feature["properties"][prop] = row[prop]
        geojson["features"].append(feature)
    return geojson


def point_df_to_geojson(df: pd.DataFrame, properties: list):
    """
    Author: Geoff Boeing:
    https://geoffboeing.com/2015/10/exporting-python-data-geojson/
    """

    geojson = {"type": "FeatureCollection", "features": []}
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": []},
        }
        feature["geometry"]["coordinates"] = [row["geometry"].x, row["geometry"].y]
        for prop in properties:
            feature["properties"][prop] = row[prop]
        geojson["features"].append(feature)
    return geojson


def fill_na(df_na):
    """
    fill str NaN with ""
    fill numeric NaN with 0
    """
    df = df_na.copy()
    num_col = list(df.select_dtypes([np.number]).columns)
    RanchLogger.info("numeric columns: {}".format(num_col))
    object_col = list(df.select_dtypes(["object"]).columns)
    RanchLogger.info("str columns: ".format(object_col))

    for x in list(df.columns):
        if x in num_col:
            df[x].fillna(0, inplace=True)
        elif x in object_col:
            df[x].fillna("", inplace=True)

    return df


def identify_dead_end_nodes(links):
    """
    iteratively find the dead end in networks
    """

    A_B_df = pd.concat(
        [links, links.rename(columns={"u": "v", "v": "u"})],
        ignore_index=True,
        sort=False,
    )

    A_B_df.drop_duplicates(inplace=True)

    A_B_df = A_B_df.groupby(["u"]).count().reset_index()

    single_node_list = A_B_df[A_B_df.v == 1].u.tolist()

    return single_node_list


def ox_graph(nodes_df, links_df):
    """
    create an osmnx-flavored network graph
    osmnx doesn't like values that are arrays, so remove the variables
    that have arrays.  osmnx also requires that certain variables
    be filled in, so do that too.
    Parameters
    ----------
    nodes_df : GeoDataFrame
    link_df : GeoDataFrame
    Returns
    -------
    networkx multidigraph
    """
    try:
        graph_nodes = nodes_df.drop(
            ["inboundReferenceId", "outboundReferenceId"], axis=1
        )
    except:
        graph_nodes = nodes_df.copy()

    graph_nodes.gdf_name = "network_nodes"
    graph_nodes["id"] = graph_nodes["shst_node_id"]

    graph_links = links_df.copy()
    graph_links["id"] = graph_links["shstReferenceId"]
    graph_links["key"] = graph_links["shstReferenceId"]

    try:
        G = ox.graph_from_gdfs(graph_nodes, graph_links)
    except AttributeError:
        try:
            # ox 1.7.1
            graph_links.set_index(["u", "v", "key"], inplace=True)
            G = ox.utils_graph.graph_from_gdfs(graph_nodes, graph_links)
        except AttributeError:
            RanchLogger.debug(
                "Please try a different version of your OSMNX package.Version 0.15.1 is recommended."
            )

    return G


proj_wgs84 = pyproj.Proj("+proj=longlat +datum=WGS84")


def geodesic_point_buffer(lat, lon, meters):
    # Azimuthal equidistant projection
    aeqd_proj = "+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0"
    project = partial(
        pyproj.transform, pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)), proj_wgs84
    )
    buf = Point(0, 0).buffer(meters)  # distance in metres
    return Polygon(transform(project, buf).exterior.coords[:])


def find_closest_node(node_df, node_candidates_df, unique_id: list):
    """
    find closest node in node_candidates_df for each node in node_df

    Parameters:
    ------------
    node_df
    node_candidates_df

    return
    ------------
    stops with drive nodes id
    """

    # convert crs
    node_df = node_df.to_crs(CRS("epsg:3857"))
    node_df["X"] = node_df["geometry"].apply(lambda p: p.x)
    node_df["Y"] = node_df["geometry"].apply(lambda p: p.y)

    node_candidates_df = node_candidates_df.to_crs(CRS("epsg:3857"))
    node_candidates_df["X"] = node_candidates_df.geometry.map(lambda g: g.x)
    node_candidates_df["Y"] = node_candidates_df.geometry.map(lambda g: g.y)

    # cKDTree
    inventory_node_ref = node_candidates_df[["X", "Y"]].values
    tree = cKDTree(inventory_node_ref)

    nearest_node_df = pd.DataFrame()

    for i in range(len(node_df)):
        point = node_df.iloc[i][["X", "Y"]].values
        dd, ii = tree.query(point, k=1)
        add_snap_df = (
            gpd.GeoDataFrame(node_candidates_df.iloc[ii])
            .transpose()
            .reset_index(drop=True)
        )

        for c in unique_id:
            add_snap_df[c] = node_df.iloc[i][c]

        nearest_node_df = nearest_node_df.append(
            add_snap_df, ignore_index=True, sort=False
        )
    return nearest_node_df


def reproject(link, node, epsg):
    """
    reporoject link and node geodataframes

    for nodes, update X and Y columns

    """

    link = link.to_crs(epsg=epsg)
    node = node.to_crs(epsg=epsg)

    node["X"] = node["geometry"].apply(lambda p: p.x)
    node["Y"] = node["geometry"].apply(lambda p: p.y)

    return link, node


def num_of_drive_loadpoint_per_centroid(existing_drive_cc_df, existing_node_gdf):
    """
    decide number of loading point for drive access per centroid

    logic: for drive, find the closest points to the existing loading point

    return:
    dataframe
    for each existing drive loading point, number of new loading point needs to be generated. currently set to 1.

    """
    existing_pairs_of_centroid_loadpoint_df = (
        existing_drive_cc_df.groupby(["c", "non_c"])
        .count()
        .reset_index()
        .drop(["A", "B"], axis=1)
    )

    existing_num_of_loadpoint_per_c_df = (
        existing_drive_cc_df.groupby(["c", "non_c"])
        .count()
        .groupby("c")
        .count()[["A"]]
        .rename(columns={"A": "abm_num_load"})
        .reset_index()
    )

    num_drive_loadpoint_new_near_old = pd.merge(
        existing_pairs_of_centroid_loadpoint_df,
        existing_num_of_loadpoint_per_c_df,
        how="left",
        on="c",
    )

    num_drive_loadpoint_new_near_old["osm_num_load"] = 1

    num_drive_loadpoint_new_near_old = pd.merge(
        num_drive_loadpoint_new_near_old,
        existing_node_gdf[["N", "X", "Y"]],
        how="left",
        left_on="non_c",
        right_on="N",
    )
    return num_drive_loadpoint_new_near_old


def num_of_walk_bike_loadpoint_per_centroid(existing_centroid_df):
    """
    decide number of loading point for walk and bike access per centroid

    logic: find 5 closest points to centroid

    return:
    dataframe
    for each centroid, number of loading point needs to be generated.

    """

    num_loadpoint = existing_centroid_df[["N", "X", "Y"]].copy()
    num_loadpoint["osm_num_load"] = np.int(5)
    num_loadpoint.rename(columns={"N": "c"}, inplace=True)

    return num_loadpoint


def find_new_load_point(abm_load_ref_df, all_node):
    """
    find the loading points in osm nodes

    input: osm node, loading point reference input

    output:  dataframe of pairs of centroid and loading point, with point geometry of loading point

    works in epsg = 26915

    """

    all_node_gdf = all_node.copy()

    all_node_gdf = all_node_gdf.to_crs(epsg=26915)
    all_node_gdf["X"] = all_node_gdf["geometry"].apply(lambda g: g.x)
    all_node_gdf["Y"] = all_node_gdf["geometry"].apply(lambda g: g.y)

    inventory_node_df = all_node_gdf.copy()
    inventory_node_ref = inventory_node_df[["X", "Y"]].values
    tree_default = cKDTree(inventory_node_ref)

    new_load_point_gdf = gpd.GeoDataFrame()

    for i in range(len(abm_load_ref_df)):
        point = abm_load_ref_df.iloc[i][["X", "Y"]].values
        c_id = abm_load_ref_df.iloc[i]["c"]
        n_neigh = abm_load_ref_df.iloc[i]["osm_num_load"]

        if "c" in all_node_gdf.columns:
            inventory_node_df = (
                all_node_gdf[all_node_gdf.c == c_id].copy().reset_index()
            )
            if len(inventory_node_df) == 0:
                continue
            else:
                inventory_node_ref = inventory_node_df[["X", "Y"]].values
                tree = cKDTree(inventory_node_ref)

        else:
            inventory_node_df = all_node_gdf.copy()
            tree = tree_default

        dd, ii = tree.query(point, k=n_neigh)
        if n_neigh == 1:
            add_gdf = (
                gpd.GeoDataFrame(
                    inventory_node_df[
                        ["osm_node_id", "shst_node_id", "model_node_id", "geometry"]
                    ].iloc[ii]
                )
                .transpose()
                .reset_index(drop=True)
            )
        else:
            add_gdf = gpd.GeoDataFrame(
                inventory_node_df[
                    ["osm_node_id", "shst_node_id", "model_node_id", "geometry"]
                ].iloc[ii]
            ).reset_index(drop=True)
        add_gdf["c"] = int(abm_load_ref_df.iloc[i]["c"])
        if i == 0:
            new_load_point_gdf = add_gdf.copy()

        else:
            new_load_point_gdf = new_load_point_gdf.append(
                add_gdf, ignore_index=True, sort=False
            )

    return new_load_point_gdf.rename(columns={"geometry": "geometry_ld"})


def generate_centroid_connectors_shape(zone_loading_node_df):
    """
    calls function to generate loading point reference table,
    and calls function to find loading points

    build linestring based on pairs of centroid and loading point

    return centroid connectors and centroids
    """

    # inbound cc
    new_cc_gdf = zone_loading_node_df.copy()
    new_cc_gdf["geometry"] = [
        LineString(xy) for xy in zip(new_cc_gdf["ld_point"], new_cc_gdf["c_point"])
    ]

    new_cc_gdf["fromIntersectionId"] = new_cc_gdf["shst_node_id"]
    new_cc_gdf["shstGeometryId"] = new_cc_gdf["geometry"].apply(
        lambda x: create_unique_shape_id(x)
    )
    new_cc_gdf["id"] = new_cc_gdf["shstGeometryId"]

    new_cc_gdf = new_cc_gdf.rename(
        columns={"osm_node_id": "u", "model_node_id_x": "A", "model_node_id_y": "B"}
    )

    if ("A" not in new_cc_gdf.columns ) and ("B" not in new_cc_gdf.columns):
        new_cc_gdf = new_cc_gdf.rename(columns={"X": "A", "Y": "B"})

    new_cc_gdf = new_cc_gdf[
        ["A", "B", "u", "fromIntersectionId", "shstGeometryId", "id", "geometry"]
    ]

    new_cc_gdf = gpd.GeoDataFrame(new_cc_gdf, crs=standard_crs)

    return new_cc_gdf


def generate_centroid_connectors_link(cc_shape_gdf):
    """
    generate two way connector links based on one-way shape

    input: connector shapes in one direction (inbound)
    output: connector links in both directions
    """

    cc_link_gdf = cc_shape_gdf.copy()

    cc_link_gdf = cc_link_gdf.rename(
        columns={"A": "B", "B": "A", "u": "v", "fromIntersectionId": "toIntersectionId"}
    )

    cc_link_gdf = pd.concat([cc_shape_gdf, cc_link_gdf], sort=False, ignore_index=True)

    return cc_link_gdf


def consolidate_cc(
    link,
    drive_centroid,
    node,
    new_drive_cc,
    new_walk_cc=pd.DataFrame(),
    new_bike_cc=pd.DataFrame(),
):
    link_gdf = link.copy()
    node_gdf = node.copy()
    drive_centroid_gdf = drive_centroid.copy()
    new_drive_cc_gdf = new_drive_cc.copy()

    if len(new_walk_cc) > 0:
        new_walk_cc_gdf = new_walk_cc.copy()
        new_walk_cc_gdf["walk_access"] = int(1)
    else:
        new_walk_cc_gdf = pd.DataFrame()
    if len(new_bike_cc) > 0:
        new_bike_cc_gdf = new_bike_cc.copy()
        new_bike_cc_gdf["bike_access"] = int(1)
    else:
        new_bike_cc_gdf = pd.DataFrame()

    new_drive_cc_gdf["drive_access"] = int(1)
    new_drive_cc_gdf["walk_access"] = int(0)
    new_drive_cc_gdf["bike_access"] = int(0)

    new_cc_gdf = pd.concat(
        [new_drive_cc_gdf, new_walk_cc_gdf, new_bike_cc_gdf],
        sort=False,
        ignore_index=True,
    )

    new_cc_gdf["u"] = new_cc_gdf["u"].astype(np.int64)
    new_cc_gdf["A"] = new_cc_gdf["A"].astype(np.int64)

    new_cc_geometry_gdf = (
        new_cc_gdf[["A", "B", "geometry", "fromIntersectionId", "u"]]
        .drop_duplicates(subset=["A", "B"])
        .copy()
    )

    new_cc_geometry_gdf["shstGeometryId"] = range(1, 1 + len(new_cc_geometry_gdf))
    new_cc_geometry_gdf["shstGeometryId"] = new_cc_geometry_gdf["shstGeometryId"].apply(
        lambda x: "cc" + str(x)
    )
    new_cc_geometry_gdf["id"] = new_cc_geometry_gdf["shstGeometryId"]

    unique_cc_gdf = (
        new_cc_gdf.groupby(["A", "B"])
        .agg({"drive_access": "max", "walk_access": "max", "bike_access": "max"})
        .reset_index()
    )

    unique_cc_gdf = pd.merge(
        unique_cc_gdf, new_cc_geometry_gdf, how="left", on=["A", "B"]
    )

    # add the other direction
    cc_gdf = pd.concat(
        [
            unique_cc_gdf,
            unique_cc_gdf.rename(
                columns={
                    "A": "B",
                    "B": "A",
                    "u": "v",
                    "fromIntersectionId": "toIntersectionId",
                }
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    cc_link_columns_list = [
        "A",
        "B",
        "drive_access",
        "walk_access",
        "bike_access",
        "shstGeometryId",
        "id",
        "u",
        "v",
        "fromIntersectionId",
        "toIntersectionId",
    ]
    cc_link_df = cc_gdf[cc_link_columns_list].copy()

    cc_shape_columns_list = ["id", "geometry", "fromIntersectionId", "toIntersectionId"]
    cc_shape_gdf = cc_gdf[cc_shape_columns_list].drop_duplicates(subset=["id"]).copy()

    return cc_link_df, cc_shape_gdf


def project_geometry(geometry, crs=None, to_crs=None, to_latlong=False):
    """
    Project a shapely geometry from its current CRS to another.
    If to_crs is None, project to the UTM CRS for the UTM zone in which the
    geometry's centroid lies. Otherwise project to the CRS defined by to_crs.
    Parameters
    ----------
    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        the geometry to project
    crs : dict or string or pyproj.CRS
        the starting CRS of the passed-in geometry. if None, it will be set to
        settings.default_crs
    to_crs : dict or string or pyproj.CRS
        if None, project to UTM zone in which geometry's centroid lies,
        otherwise project to this CRS
    to_latlong : bool
        if True, project to settings.default_crs and ignore to_crs
    Returns
    -------
    geometry_proj, crs : tuple
        the projected geometry and its new CRS
    """
    if crs is None:
        crs = CRS("epsg:4326")

    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=crs)
    gdf_proj = project_gdf(gdf, to_crs=to_crs, to_latlong=to_latlong)
    geometry_proj = gdf_proj["geometry"].iloc[0]
    return geometry_proj, gdf_proj.crs


def project_gdf(gdf, to_crs=None, to_latlong=False):
    """
    Project a GeoDataFrame from its current CRS to another.
    If to_crs is None, project to the UTM CRS for the UTM zone in which the
    GeoDataFrame's centroid lies. Otherwise project to the CRS defined by
    to_crs. The simple UTM zone calculation in this function works well for
    most latitudes, but may not work for some extreme northern locations like
    Svalbard or far northern Norway.
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame to be projected
    to_crs : dict or string or pyproj.CRS
        if None, project to UTM zone in which gdf's centroid lies, otherwise
        project to this CRS
    to_latlong : bool
        if True, project to settings.default_crs and ignore to_crs
    Returns
    -------
    gdf_proj : geopandas.GeoDataFrame
        the projected GeoDataFrame
    """
    if gdf.crs is None or len(gdf) < 1:
        raise ValueError("GeoDataFrame must have a valid CRS and cannot be empty")

    # if to_latlong is True, project the gdf to latlong
    if to_latlong:
        gdf_proj = gdf.to_crs(CRS("epsg:4326"))
        # utils.log(f"Projected GeoDataFrame to {settings.default_crs}")

    # else if to_crs was passed-in, project gdf to this CRS
    elif to_crs is not None:
        gdf_proj = gdf.to_crs(to_crs)
        # utils.log(f"Projected GeoDataFrame to {to_crs}")

    # otherwise, automatically project the gdf to UTM
    else:
        # if CRS.from_user_input(gdf.crs).is_projected:
        #   raise ValueError("Geometry must be unprojected to calculate UTM zone")

        # calculate longitude of centroid of union of all geometries in gdf
        avg_lng = gdf["geometry"].unary_union.centroid.x

        # calculate UTM zone from avg longitude to define CRS to project to
        utm_zone = int(math.floor((avg_lng + 180) / 6.0) + 1)
        utm_crs = (
            f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )

        # project the GeoDataFrame to the UTM CRS
        gdf_proj = gdf.to_crs(utm_crs)
        # utils.log(f"Projected GeoDataFrame to {gdf_proj.crs}")

    return gdf_proj


def buffer1(polygon):
    buffer_dist = 10
    poly_proj, crs_utm = project_geometry(polygon)
    poly_proj_buff = poly_proj.buffer(buffer_dist)
    poly_buff, _ = project_geometry(poly_proj_buff, crs=crs_utm, to_latlong=True)

    return poly_buff


def buffer2(polygon):
    return polygon.minimum_rotated_rectangle


def line_buffer(line, buffer_dist):
    line_proj, crs_utm = project_geometry(line, to_crs=CRS("epsg:3857"))
    line_proj_buff = line_proj.buffer(buffer_dist)
    line_buff, _ = project_geometry(
        line_proj_buff, crs=CRS("epsg:3857"), to_latlong=True
    )

    return line_buff


def getAngle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return ang + 360 if ang < 0 else ang


def isDuplicate(a, b, zoneUnique):
    length = len(zoneUnique)
    # print("    unique zone unique length {}".format(length))
    for i in range(length):
        # print("           compare {} with zone unique {}".format(a, zoneUnique[i]))
        ang = getAngle(a, b, zoneUnique[i])

        if (ang < 45) | (ang > 315):
            return None

    zoneUnique += [a]


def haversine_distance(origin: list, destination: list, units="miles"):
    """
    Calculates haversine distance between two points

    Args:
    origin: lat/lon for point A
    destination: lat/lon for point B
    units: either "miles" or "meters"

    Returns: string
    """

    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6378137  # meter

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = {"meters": radius * c}  # meters
    d["miles"] = d["meters"] * 0.000621371  # miles

    return d[units]


def get_non_near_connectors(all_cc_link_gdf, num_connectors_per_centroid, zone_id: str):
    keep_cc_gdf = pd.DataFrame()

    for c in all_cc_link_gdf[zone_id].unique():
        # print("zone id {}".format(c))

        zone_cc_gdf = all_cc_link_gdf[all_cc_link_gdf[zone_id] == c].copy()

        centroid = zone_cc_gdf.c_point.iloc[0]

        # if the zone has less than 4 cc, keep all
        if len(zone_cc_gdf) <= 4:
            keep_cc_gdf = keep_cc_gdf.append(zone_cc_gdf, sort=False, ignore_index=True)

        # if the zone has more than 4 cc
        else:
            zoneUnique = []

            zoneCandidate = zone_cc_gdf["ld_point"].to_list()
            # print("zone candidate {}".format(zoneCandidate))
            for point in zoneCandidate:
                # print("evaluate: {}".format(point))
                if len(zoneUnique) == 0:
                    zoneUnique += [point]
                else:
                    isDuplicate(point, centroid, zoneUnique)
                # print("zone unique {}".format(zoneUnique))
                if len(zoneUnique) == 4:
                    break

            zone_cc_gdf = zone_cc_gdf[
                zone_cc_gdf.ld_point.isin([tuple(z) for z in zoneUnique])
            ]

            keep_cc_gdf = keep_cc_gdf.append(zone_cc_gdf, sort=False, ignore_index=True)

    return keep_cc_gdf


def create_unique_shape_id(line_string: LineString):
    """
    Creates a unique hash id using the coordinates of the geomtery

    Args:
    line_string: Line Geometry as a LineString

    Returns: string
    """

    x1, y1 = list(line_string.coords)[0]  # first co-ordinate (A node)
    x2, y2 = list(line_string.coords)[-1]  # last co-ordinate (B node)

    message = "Geometry {} {} {} {}".format(x1, y1, x2, y2)
    unhashed = message.encode("utf-8")
    hash = hashlib.md5(unhashed).hexdigest()

    return hash


def create_unique_node_id(point: Point):
    """
    Creates a unique hash id using the coordinates of the geomtery

    Args:
    point: Point Geometry

    Returns: string
    """

    x1, y1 = list(point.coords)[0]  # first co-ordinate (A node)

    message = "Geometry {} {}".format(x1, y1)
    unhashed = message.encode("utf-8")
    hash = hashlib.md5(unhashed).hexdigest()

    return hash


def create_unique_link_id(u_node: Point, v_node: Point):
    """
    Creates a unique hash id using the coordinates of the geomtery

    Args:
    line_string: Line Geometry as a LineString

    Returns: string
    """

    x1, y1 = list(u_node.coords)[0]  # first co-ordinate (A node)
    x2, y2 = list(v_node.coords)[0]  # last co-ordinate (B node)

    message = "Geometry {} {} {} {}".format(x1, y1, x2, y2)
    unhashed = message.encode("utf-8")
    hash = hashlib.md5(unhashed).hexdigest()

    return hash
