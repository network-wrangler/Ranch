import re
import os
import json
import pickle

import pytest

import ranch
from ranch import Roadway
from ranch import Parameters
from ranch import roadway
from ranch.utils import link_df_to_geojson, point_df_to_geojson
from ranch.logger import RanchLogger

"""
Run tests from bash/shell
Run just the tests labeled project using `pytest -m roadway`
To run with print statments, use `pytest -s -m roadway`
"""

root_dir = os.path.join("D:/github/Ranch")

parameters = Parameters(ranch_base_dir=root_dir)


@pytest.mark.roadway_step3
@pytest.mark.roadway
@pytest.mark.travis
def test_create_roadway_network_from_extracts(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    roadway_network = Roadway.create_roadway_network_from_extracts(
        shst_extract_dir=os.path.join(
            root_dir, "data", "external", "sharedstreets_extract"
        ),
        osm_extract_dir=os.path.join(root_dir, "data", "external", "osmnx_extract"),
        parameters=parameters,
    )

    RanchLogger.info("Network has {} links".format(roadway_network.links_df.shape[0]))
    RanchLogger.info("Network has {} nodes".format(roadway_network.nodes_df.shape[0]))
    RanchLogger.info("Network has {} shapes".format(roadway_network.shapes_df.shape[0]))

    RanchLogger.info("-------write out shape geojson---------")

    shape_prop = [
        "id",
        "fromIntersectionId",
        "toIntersectionId",
        "forwardReferenceId",
        "backReferenceId",
    ]
    shape_geojson = link_df_to_geojson(roadway_network.shapes_df, shape_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step3_shapes.geojson"), "w"
    ) as f:
        json.dump(shape_geojson, f)

    RanchLogger.info("-------write out node geojson---------")

    node_prop = roadway_network.nodes_df.drop("geometry", axis=1).columns.tolist()
    node_geojson = point_df_to_geojson(roadway_network.nodes_df, node_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step3_nodes.geojson"), "w"
    ) as f:
        json.dump(node_geojson, f)

    RanchLogger.info("-------write out link geojson---------")

    link_prop = roadway_network.links_df.drop("geometry", axis=1).columns.tolist()
    link_geojson = link_df_to_geojson(roadway_network.links_df, link_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step3_links.geojson"), "w"
    ) as f:
        json.dump(link_geojson, f)

    working_network_filename = os.path.join(
        root_dir, "data", "interim", "step3_network.pickle"
    )
    pickle.dump(roadway_network, open(working_network_filename, "wb"))


@pytest.mark.roadway_step5
@pytest.mark.roadway
@pytest.mark.travis
def test_step5_tidy_roadway(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    working_network_filename = os.path.join(
        root_dir, "data", "interim", "step3_network.pickle"
    )

    roadway_network = pickle.load(open(working_network_filename, "rb"))

    roadway_network.tidy_roadway(
        county_boundary_file=os.path.join(
            root_dir, "data", "external", "cb_2018_us_county_500k", "san_joaquin.shp"
        ),
        county_variable_name="NAME",
    )

    RanchLogger.info("Network has {} links".format(roadway_network.links_df.shape[0]))
    RanchLogger.info("Network has {} nodes".format(roadway_network.nodes_df.shape[0]))
    RanchLogger.info("Network has {} shapes".format(roadway_network.shapes_df.shape[0]))

    RanchLogger.info("-------write out shape geojson---------")

    shape_prop = [
        "id",
        "fromIntersectionId",
        "toIntersectionId",
        "forwardReferenceId",
        "backReferenceId",
    ]
    shape_geojson = link_df_to_geojson(roadway_network.shapes_df, shape_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step5_shapes.geojson"), "w"
    ) as f:
        json.dump(shape_geojson, f)

    RanchLogger.info("-------write out node geojson---------")

    node_prop = roadway_network.nodes_df.drop("geometry", axis=1).columns.tolist()
    node_geojson = point_df_to_geojson(roadway_network.nodes_df, node_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step5_nodes.geojson"), "w"
    ) as f:
        json.dump(node_geojson, f)

    RanchLogger.info("-------write out link geojson---------")

    link_prop = roadway_network.links_df.drop("geometry", axis=1).columns.tolist()
    link_geojson = link_df_to_geojson(roadway_network.links_df, link_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step5_links.geojson"), "w"
    ) as f:
        json.dump(link_geojson, f)

    working_network_filename = os.path.join(
        root_dir, "data", "interim", "step5_network.pickle"
    )
    pickle.dump(roadway_network, open(working_network_filename, "wb"))


@pytest.mark.roadway_step7
@pytest.mark.roadway
@pytest.mark.travis
def test_step7_centroid_connectors(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    working_network_filename = os.path.join(
        root_dir, "data", "interim", "step5_network.pickle"
    )

    roadway_network = pickle.load(open(working_network_filename, "rb"))

    roadway_network.build_centroid_connectors(
        build_taz_drive=True,
        build_taz_active_modes=True,
        input_taz_polygon_file=os.path.join(
            root_dir, "data", "external", "taz", "SJ_TAZ_Aug2011.shp"
        ),
    )

    roadway_network.taz_cc_shape_gdf.to_file(
        os.path.join(root_dir, "data", "interim", "taz_connectors.geojson"),
        driver="GeoJSON",
    )

    roadway_network.taz_node_gdf.to_file(
        os.path.join(root_dir, "data", "interim", "taz_centroid.geojson"),
        driver="GeoJSON",
    )

    print("-------write out pickle---------")
    roadway_network.taz_node_gdf.to_pickle(
        os.path.join(root_dir, "data", "interim", "centroid_node.pickle"),
    )
    roadway_network.taz_cc_link_gdf.to_pickle(
        os.path.join(root_dir, "data", "interim", "cc_link.pickle"),
    )
    roadway_network.taz_cc_shape_gdf.to_pickle(
        os.path.join(root_dir, "data", "interim", "cc_shape.pickle"),
    )


@pytest.mark.roadway_step8
@pytest.mark.roadway
@pytest.mark.travis
def test_step8_standard_format(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    working_network_filename = os.path.join(
        root_dir, "data", "interim", "step5_network.pickle"
    )

    roadway_network = pickle.load(open(working_network_filename, "rb"))

    roadway_network.standard_format(
        county_boundary_file=os.path.join(
            root_dir, "data", "external", "cb_2018_us_county_500k", "san_joaquin.shp"
        ),
        county_variable_name="NAME",
    )

    print(roadway_network.links_df.iloc[0])

    RanchLogger.info("Network has {} links".format(roadway_network.links_df.shape[0]))
    RanchLogger.info("Network has {} nodes".format(roadway_network.nodes_df.shape[0]))
    RanchLogger.info("Network has {} shapes".format(roadway_network.shapes_df.shape[0]))

    RanchLogger.info("-------write out shape geojson---------")

    shape_prop = [
        "id",
        "fromIntersectionId",
        "toIntersectionId",
        "forwardReferenceId",
        "backReferenceId",
    ]
    shape_geojson = link_df_to_geojson(roadway_network.shapes_df, shape_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step8_shapes.geojson"), "w"
    ) as f:
        json.dump(shape_geojson, f)

    RanchLogger.info("-------write out node geojson---------")

    node_prop = roadway_network.nodes_df.drop("geometry", axis=1).columns.tolist()
    node_geojson = point_df_to_geojson(roadway_network.nodes_df, node_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step8_nodes.geojson"), "w"
    ) as f:
        json.dump(node_geojson, f)

    RanchLogger.info("-------write out link geojson---------")

    link_prop = roadway_network.links_df.drop("geometry", axis=1).columns.tolist()
    link_geojson = link_df_to_geojson(roadway_network.links_df, link_prop)

    with open(
        os.path.join(root_dir, "data", "interim", "step8_links.geojson"), "w"
    ) as f:
        json.dump(link_geojson, f)

    RanchLogger.info("-------write out link json---------")

    link_prop = roadway_network.links_df.drop(["geometry"], axis=1).columns.tolist()

    out = roadway_network.links_df[link_prop].to_json(orient="records")

    with open(os.path.join(root_dir, "data", "interim", "step8_links.json"), "w") as f:
        f.write(out)

    working_network_filename = os.path.join(
        root_dir, "data", "interim", "step8_network.pickle"
    )
    pickle.dump(roadway_network, open(working_network_filename, "wb"))
