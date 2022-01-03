import pytest

import geopandas as gpd
import osmnx as ox
#from osmnx.geocoder import geocode_to_gdf
import sharedstreets.dataframe


#@pytest.fixture(scope='module')
#def boundary():
#    return geocode_to_gdf("Lodi, California, USA")

@pytest.fixture(scope='module')
def sj_boundary():
    return gpd.read_file(r'C:\Users\uscd675041\WSP O365\RanchData - Documents\data\external\sharedstreets_extract\sanjoaquin.shp')

@pytest.fixture(scope='module')
def boundary_extents(boundary):
    return list(zip(boundary['bbox_west'], boundary['bbox_south'], boundary['bbox_east'], boundary['bbox_north']))


#def test_shst_extraction(boundary_extents):
#    extract = sharedstreets.dataframe.get_bbox(*boundary_extents[0])
#    links = extract.geometries

def test_osm_extraction(sj_boundary):

    G_drive

# @pytest.mark.travis
# def test_shst_extraction(request):
#     """
#     tests that shst extraction is run
#     """
#
#     root_dir = os.path.join("D:/github/Ranch")
#
#     print("\n--Starting:", request.node.name)
#
#     ranch.run_shst_extraction(
#         input_polygon_file = os.path.join(root_dir, "data", "external", "sharedstreets_extract", "sanjoaquin.shp"),
#         output_dir = os.path.join(root_dir, "data", "external", "sharedstreets_extract")
#     )

# @pytest.mark.travis
# def test_read_shst_extraction(request):
#     """
#     tests that reading shst extraction is run
#     """
#
#     root_dir = os.path.join("D:/github/Ranch")
#
#     print("\n--Starting:", request.node.name)
#
#     df = ranch.read_shst_extraction(
#         path = os.path.join(root_dir, "data", "external", "sharedstreets_extract"),
#         suffix = "*.out.geojson"
#     )
#
#     print(df.info())
#
# @pytest.mark.travis
# def test_extract_osm_link_from_shst_extraction(request):
#     """
#     tests that reading shst extraction is run
#     """
#
#     root_dir = os.path.join("D:/github/Ranch")
#
#     print("\n--Starting:", request.node.name)
#
#     shst_df = ranch.read_shst_extraction(
#         path = os.path.join(root_dir, "data", "external", "sharedstreets_extract"),
#         suffix = "*.out.geojson"
#     )
#
#     print(shst_df.info())
#
#     osm_from_shst_df = sharedstreets.extract_osm_link_from_shst_extraction(
#         shst_df
#     )
#
#     print(osm_from_shst_df.info())
#
# @pytest.mark.menow
# @pytest.mark.shst
# def test_shst_match(request):
#     """
#     test that shst match is run
#     """
#
#     root_dir = os.path.join("D:/github/Ranch")
#
#     print("\n--Starting:", request.node.name)
#
#     ranch.run_shst_match(
#         input_network_file = os.path.join(root_dir, "data", "external", "sjmodel", "Network", "2015","TCM_MASTER_2A22_102717.shp"),
#         input_crs = CRS("ESRI:102643"),
#         output_dir = os.path.join(root_dir, "data", "external", "sjmodel", "shst_match"),
#         #custom_match_option = '--tile-hierarchy=8 --search-radius=50'
#         custom_match_option = '--tile-hierarchy=8 --search-radius=50 --snap-intersections'
#     )