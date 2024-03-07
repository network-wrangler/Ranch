import os

import pytest
import pickle

import pandas as pd
import geopandas as gpd

import ranch
from ranch import Transit
from ranch import Parameters
from ranch import roadway
from ranch import transit
from ranch.logger import RanchLogger
from ranch import sharedstreets

root_dir = os.path.join("D:/github/Ranch")
parameters = Parameters(ranch_base_dir=root_dir)

scratch_dir = os.path.join(root_dir, "tests", "scratch")

working_network_filename = os.path.join(
    root_dir, "data", "interim", "step5_network.pickle"
)
roadway_network = pickle.load(open(working_network_filename, "rb"))


@pytest.mark.transit
@pytest.mark.travis
def test_read_gtfs_feed(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )


@pytest.mark.transit
@pytest.mark.travis
def test_get_representative_trip_for_route(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()

    transit_network.feed.trips.to_csv(
        os.path.join(scratch_dir, "test_trips.txt"), index=False
    )


@pytest.mark.transit
@pytest.mark.travis
def test_snap_stop_to_node(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.snap_stop_to_node()

    transit_network.feed.stops.to_csv(
        os.path.join(scratch_dir, "test_stops.txt"), index=False
    )


@pytest.mark.travis
def test_route_bus_link_osmnx_from_start_to_end(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    transit_network.snap_stop_to_node()

    # transit_network.feed.trips = transit_network.feed.trips[-2:]

    transit_network.route_bus_link_osmnx_from_start_to_end()

    transit_network.trip_osm_link_df.to_csv(
        os.path.join(scratch_dir, "test_routing.txt"), index=False
    )

    trip_osm_link_gdf = pd.merge(
        transit_network.trip_osm_link_df.drop("wayId", axis=1),
        roadway_network.links_df[["u", "v", "shstReferenceId", "geometry"]],
        how="left",
        on=["u", "v", "shstReferenceId"],
    )

    trip_osm_link_gdf = gpd.GeoDataFrame(
        trip_osm_link_gdf, crs=roadway_network.links_df.crs
    )

    trip_osm_link_gdf.to_file(
        os.path.join(scratch_dir, "test_routing.geojson"), driver="GeoJSON"
    )


@pytest.mark.travis
def test_stops_far_from_routed_route(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    transit_network.snap_stop_to_node()

    transit_network.feed.trips = transit_network.feed.trips[:2]

    transit_network.route_bus_link_osmnx_from_start_to_end()

    transit_network.trip_osm_link_df.to_csv(
        os.path.join(scratch_dir, "test_routing_v2.txt"), index=False
    )

    transit_network.set_bad_stops()
    print(transit_network.bad_stop_dict)


@pytest.mark.routing
@pytest.mark.travis
def test_route_bus_link_osmnx_between_stops(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    transit_network.snap_stop_to_node()
    """
    transit_network.feed.trips = transit_network.feed.trips[
        transit_network.feed.trips.trip_id == '202843'
    ]
    """
    print(transit_network.feed.trips)

    transit_network.route_bus_link_osmnx_from_start_to_end()

    transit_network.set_bad_stops()

    transit_network.route_bus_link_osmnx_between_stops()
    transit_network.trip_osm_link_df.to_csv(
        os.path.join(scratch_dir, "test_routing_v3.txt"), index=False
    )

    trip_osm_link_gdf = gpd.GeoDataFrame(
        transit_network.trip_osm_link_df, crs=roadway_network.links_df.crs
    )

    trip_osm_link_gdf.drop("wayId", axis=1).to_file(
        os.path.join(scratch_dir, "test_routing_v2.geojson"), driver="GeoJSON"
    )


@pytest.mark.shst_match
@pytest.mark.travis
def test_match_gtfs_shapes_to_shst(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()

    transit_network.match_gtfs_shapes_to_shst()

    print(transit_network.trip_shst_link_df.head(3))


# @pytest.mark.menow
@pytest.mark.travis
def test_route_bus_trip(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    RanchLogger.info(
        "representative feed has {} trips".format(
            transit_network.feed.trips.trip_id.nunique()
        )
    )

    transit_network.trip_osm_link_df = gpd.read_file(
        os.path.join(scratch_dir, "test_routing.geojson")
    )

    transit_network.route_bus_trip()

    bus_trip_link_df = gpd.GeoDataFrame(
        transit_network.bus_trip_link_df, crs=roadway_network.links_df.crs
    )

    bus_trip_link_df.to_file(
        os.path.join(scratch_dir, "test_routing_v4.geojson"), driver="GeoJSON"
    )

    print(transit_network.bus_trip_link_df.head(3))


# @pytest.mark.menow
@pytest.mark.travis
def test_update_bus_stop_node(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    RanchLogger.info(
        "representative feed has {} trips".format(
            transit_network.feed.trips.trip_id.nunique()
        )
    )

    transit_network.snap_stop_to_node()
    transit_network.feed.stops.to_csv(
        os.path.join(scratch_dir, "stops.csv"), index=False
    )

    transit_network.bus_trip_link_df = gpd.read_file(
        os.path.join(scratch_dir, "test_routing_v3.geojson")
    )

    transit_network.update_bus_stop_node()
    transit_network.bus_stops.to_csv(
        os.path.join(scratch_dir, "bus_stops.csv"), index=False
    )


# @pytest.mark.menow
@pytest.mark.travis
def test_create_freq_table(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    RanchLogger.info(
        "representative feed has {} trips".format(
            transit_network.feed.trips.trip_id.nunique()
        )
    )

    print(transit_network.feed.frequencies)

    transit_network.create_freq_table()

    print(transit_network.feed.frequencies)


# @pytest.mark.menow
@pytest.mark.travis
def test_create_shape_node_table(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    RanchLogger.info(
        "representative feed has {} trips".format(
            transit_network.feed.trips.trip_id.nunique()
        )
    )

    transit_network.bus_trip_link_df = gpd.read_file(
        os.path.join(scratch_dir, "test_routing_v3.geojson")
    )

    transit_network.create_shape_node_table()
    transit_network.shape_point_df.to_csv(
        os.path.join(scratch_dir, "shapes.txt"), index=False
    )


# @pytest.mark.menow
@pytest.mark.travis
def test_write_standard_transit(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2015"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    RanchLogger.info(
        "representative feed has {} trips".format(
            transit_network.feed.trips.trip_id.nunique()
        )
    )

    transit_network.snap_stop_to_node()

    transit_network.bus_trip_link_df = gpd.read_file(
        os.path.join(scratch_dir, "test_routing_v3.geojson")
    )

    # transit_network.route_bus_trip()

    transit_network.update_bus_stop_node()

    transit_network.create_shape_node_table()

    transit_network.create_freq_table()

    transit_network.write_standard_transit()


@pytest.mark.menow
@pytest.mark.travis
def test_create_rail(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "BART"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.build_standard_transit_network(
        multithread_shortest_path=False, multithread_shst_match=True
    )

    transit_network.write_standard_transit(
        path="D:/github/Ranch/tests/scratch/test_BART"
    )

    print(transit_network.unique_rail_links_gdf)
    print(
        transit_network.unique_rail_links_gdf.sort_values(
            by=["from_stop_id", "to_stop_id"]
        )
    )
    print(
        len(
            transit_network.unique_rail_links_gdf.groupby(
                ["from_stop_id", "to_stop_id"]
            ).count()
        )
    )
    print(transit_network.unique_rail_nodes_gdf)
    print(transit_network.unique_rail_nodes_gdf.stop_id.nunique())
    print(
        len(
            transit_network.unique_rail_nodes_gdf.groupby(
                ["shape_pt_lat", "shape_pt_lon", "stop_id"]
            ).count()
        )
    )

    print(transit_network.rail_stops)
    print(transit_network.rail_trip_link_df)

    print(roadway_network.links_df.info())
    print(transit_network.roadway_network.links_df.info())


@pytest.mark.menow
@pytest.mark.travis
def test_create_rail_and_bus(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    transit_network = Transit.load_all_gtfs_feeds(
        gtfs_dir=os.path.join(root_dir, "data", "external", "gtfs", "2019"),
        roadway_network=roadway_network,
        parameters=parameters,
    )

    RanchLogger.info(
        "transit feed has {} routes, they are {}".format(
            transit_network.feed.routes.route_id.nunique(),
            transit_network.feed.routes.route_short_name.unique(),
        )
    )

    transit_network.get_representative_trip_for_route()
    RanchLogger.info(
        "representative feed has {} trips".format(
            transit_network.feed.trips.trip_id.nunique()
        )
    )

    transit_network.snap_stop_to_node()

    transit_network.route_bus_trip()

    transit_network.update_bus_stop_node()

    bus_trip_link_df = gpd.GeoDataFrame(
        transit_network.bus_trip_link_df, crs=roadway_network.links_df.crs
    )

    transit_network.route_rail_trip()

    transit_network.create_shape_node_table()

    transit_network.create_freq_table()

    transit_network.write_standard_transit(
        path="D:/github/Ranch/tests/scratch/test_2019"
    )
