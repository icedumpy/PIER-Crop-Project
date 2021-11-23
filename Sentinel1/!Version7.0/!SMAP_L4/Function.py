import pyproj
from shapely import wkt
from shapely.ops import transform
#%% Define functions (extract_pixel_values)
PROJECT_47 = pyproj.Transformer.from_crs(
    'epsg:4326',   # source coordinate system
    'epsg:32647',  # destination coordinate system
    always_xy=True # Must have
).transform

PROJECT_48 = pyproj.Transformer.from_crs(
    'epsg:4326',   # source coordinate system
    'epsg:32648',  # destination coordinate system
    always_xy=True # Must have
).transform 

def create_polygon_from_wkt(wkt_polygon, crs="epsg:4326", to_crs=None):
    """
    Create shapely polygon from string (wkt format) "MULTIPOLYGON(((...)))"
    https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects/127432#127432

    Parameters
    ----------
    wkt_polygon: str
        String of polygon (wkt format).
    crs: str
        wkt_polygon's crs (should be "epsg:4326").
    to_crs: str (optional), default None
        Re-project crs to "to_crs".

    Examples
    --------
    >>> create_polygon_from_wkt(wkt_polygon)
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32647")
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32648")
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32660")
    >>> create_polygon_from_wkt(wkt_polygon, crs="epsg:32647", to_crs="epsg:4326")

    Returns
    -------
    Polygon
    """
    polygon = wkt.loads(wkt_polygon)
    if to_crs is not None:
        if crs == "epsg:4326":
            if to_crs == "epsg:32647":
                polygon = transform(PROJECT_47, polygon)
            elif to_crs == "epsg:32648":
                polygon = transform(PROJECT_48, polygon)
        else:
            project = pyproj.Transformer.from_crs(
                crs,     # source coordinate system
                to_crs,  # destination coordinate system
                always_xy=True # Must have
            ).transform
            polygon = transform(project, polygon)
    return polygon