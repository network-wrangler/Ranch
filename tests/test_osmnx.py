import re
import os

import pytest

import ranch


@pytest.mark.osm
def test_osmnx_extraction(request):
    """
    tests that osmnx extraction is run
    """

    root_dir = os.path.join("D:/github/Ranch")

    print("\n--Starting:", request.node.name)

    ranch.run_osmnx_extraction(
        input_polygon_file=os.path.join(
            root_dir, "data", "external", "sharedstreets_extract", "sanjoaquin.shp"
        ),
        output_dir=os.path.join(root_dir, "data", "external", "osmnx_extract"),
    )
