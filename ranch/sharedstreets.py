from .logger import RanchLogger

import os
import geopandas as gpd
from pyproj import CRS
import docker 

__all__ = ["run_shst_extraction"]

def run_shst_extraction(
    input_polygon_file: str,
    output_dir: str
):

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
    polygon_gdf = polygon_gdf.to_crs(CRS("EPSG:4269"))

    # export polygon to geojson for shst node js
    for i in range(len(polygon_gdf.geometry)):
    
        RanchLogger.info("Exporting boundry file {}".format(os.path.join(output_dir, "boundary."+str(i)+".geojson")))
    
        boundary_gdf = gpd.GeoDataFrame({"geometry" : gpd.GeoSeries(polygon_gdf.geometry.iloc[i])})

        boundary_gdf.to_file(
            os.path.join(output_dir, "boundary."+str(i)+".geojson"),
            driver = "GeoJSON"
        )

        _run_shst_extraction(
            input_file_name = "boundary."+str(i),
            output_dir = output_dir
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
        image='shst:latest', 
        detach=True,
        volumes={output_dir:{'bind':'/usr/node','mode':'rw'}}, 
        tty=True,
        auto_remove=True, 
        command='/bin/bash'
    )

    container.exec_run(
        cmd=(
            'shst extract usr/node/'+input_file_name+'.geojson --out=usr/node/'+'extract.'+input_file_name+'.geojson --metadata --tile-hierarchy=8 --tiles'
        )
    )

    container.stop()