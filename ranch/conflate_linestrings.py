# %%
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import (
    MultiLineString,
    LineString,
    Point,
    Polygon,
    MultiPoint,
    GeometryCollection,
)
from shapely.ops import nearest_points, unary_union, snap, split
from shapely import concave_hull, convex_hull, dwithin, polygonize

from typing import Union, Callable, Literal
from itertools import combinations
from tqdm import tqdm

# from fastDamerauLevenshtein import damerauLevenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# osm = pd.merge

# %%

# TODO add a project card yammal to allow for hand coded modifications to conflation
# on a street by street basis


def conflate_line_segments(
    base_network: gpd.GeoDataFrame,
    join_network: gpd.GeoDataFrame,
    max_matching_distance: float,
    *,
    segment_mapping_method: Literal[
        "meter_to_meter", "seg_to_seg", "all_candidates", "all_potential_matches"
    ] = "meter_to_meter",
    conflation_criteria: Union[Callable, Literal["geometry"]] = "geometry",
    match_if_directions_reversed: bool = True,
    minimum_segment_overlap: float = 5,
    base_attribute_columns: Union[list[str], dict[str, str]] = None,
    join_attribute_columns: Union[list[str], dict[str, str]] = None,
    conflation_options: dict[
        str : tuple[
            str,
            str,
            Union[Literal["term_freq_similarity", "difference", "equals"], Callable],
        ]
    ] = dict(),
    detailed_output: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Conflate attributes from a join network to a base network based on specified criteria.

    This function takes two GeoDataFrames representing line segments (e.g., road segments)
    and conflates the attributes of the join network to the base network. The conflation
    process is controlled by various parameters that determine how segments are matched
    and which attributes are transferred.

    Parameters:
        base_network (gpd.GeoDataFrame): The base GeoDataFrame containing line segments
                                         to which attributes will be joined.
        join_network (gpd.GeoDataFrame): The join GeoDataFrame containing line segments
                                         from which attributes will be sourced.
        max_matching_distance (float): The maximum distance within which segments from
                                       the join network can be matched to the base network
                                       at closest approach.
        segment_mapping_method (Literal["meter_to_meter", "seg_to_seg", "all_candidates",
                                        "all_potential_matches"], optional): The method
                                        used to map segments between networks. Defaults
                                        to "meter_to_meter".
                                        seg_to_seg: segments are matched segment wise,
                                        do not use this option if the join network is
                                        segmented differently to the base network.
                                        all_candidates: returns the segments and matching
                                        criteria scores of all segments within
                                        max_matching_distance
                                        all_potential_matches: returns all segments with a
                                        high score of matching
                                        meter_to_meter: asserts a 1 to 1 relationship
                                        between the length of each segment
        conflation_criteria (Union[Callable, Literal["geometry"]], optional): Criteria used
                                        to determine if segments should be conflated based
                                        on geometry and other attributes. This is a
                                        callable function on the joined attributes of the
                                        final network and should return a final score
                                        be a callable or "geometry". Defaults to "geometry"
                                        where geometry is the only criteria.
        match_if_directions_reversed (bool, optional): If True, allows matching if their
                                        linestring directions of linestrings are reversed.
                                        Defaults to True.
        minimum_segment_overlap (float, optional): The minimum overlap length between
                                        segments required for conflation. Defaults to 5 m.
        base_attribute_columns (Union[list[str], dict[str, str]], optional): List or dict
                                        specifying which attribute columns to retain from
                                        the base network. Defaults to keeping all
                                        columns.
        join_attribute_columns (Union[list[str], dict[str, str]], optional): List or dict
                                        specifying which attribute columns to transfer
                                        from the join network. Defaults to keeping all
                                        columns.
        conflation_options (dict[str, tuple[str, str, Union[Literal["term_freq_similarity",
                                        "difference", "equals"], Callable]]], optional):
                                        Dictionary specifying conflation options for
                                        attribute columns. Defaults to an empty dictionary.
        detailed_output (bool, optional): If True, returns a more detailed output containing
                                        processing steps in dataframe for
                                        debugging. Defaults to False.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: A tuple containing two GeoDataFrames:
            - The first GeoDataFrame is the base network with conflated attributes.
            - The second GeoDataFrame provides details of the conflation process if
              detailed_output is True, otherwise it is an empty GeoDataFrame.
    """
    if base_attribute_columns is None:
        base_attribute_columns = []
    if join_attribute_columns is None:
        join_attribute_columns = []

    # Pre baked algorithms to call upon to make it easier to conflate
    conflation_algorithms = {
        # "edit_distance": lambda x1, x2: damerauLevenshtein(x1, x2, similarity=False),
        "term_freq_similarity": term_freq_similarity,
        "difference": lambda x1, x2: x1 - x2,
        "equals": lambda x1, x2: 1 * (x1 != x2),
    }

    conflation_criterions = {
        "geometry": lambda row: (
            1 * (row["geom_closeness"] < max_matching_distance)
            + 1 * (row["geom_closeness"] < max_matching_distance * 2),
            row["geom_closeness"],
        )
    }

    segment_mapping_methods = {
        "meter_to_meter": meter_to_meter_mapping,
        "seg_to_seg": seg_to_seg_mapping,
        "all_potential_matches": map_potential_matches,
        "all_candidates": map_all_candidates,
    }

    candidate_joins = get_candidate_joins(
        base_network,
        join_network,
        base_attribute_columns,
        join_attribute_columns,
        max_matching_distance=max_matching_distance,
    )

    print(f"number of candidates to joins {len(candidate_joins)}.....")
    candidate_joins = fix_line_orientation(
        candidate_joins,
        match_if_directions_reversed=match_if_directions_reversed,
    )

    print(f"after line orientation {len(candidate_joins)}")
    spatial_similarity_scores = difference_between_shapes(
        candidate_joins,
        minimum_segment_overlap=minimum_segment_overlap,
    )

    assert (spatial_similarity_scores.index == candidate_joins.index).all()

    spatial_similarity_scores = conflate_non_geometric_attributes(
        spatial_similarity_scores,
        candidate_joins,
        conflation_algorithms,
        conflation_options,
    )

    assert "conflate_segment_score" not in spatial_similarity_scores.columns

    # TODO - Maybe break this out into its own independent function
    if isinstance(conflation_criteria, Callable):
        conflation_agg_func = conflation_criteria
    elif conflation_criteria not in conflation_criterions:
        raise KeyError(
            f"Conflation Criteria passed that has no known algorithm, expected one of {conflation_criterions.keys()} or callable, but got {conflation_criteria}"
        )
    else:
        conflation_agg_func = conflation_criterions[conflation_criteria]

    conflation_dec_and_score = spatial_similarity_scores.apply(
        conflation_agg_func, axis=1
    )

    spatial_similarity_scores["conflate_decision"] = conflation_dec_and_score.str[
        0
    ].replace({0: "no match", 1: "candidate match", 2: "match"})
    spatial_similarity_scores["final_score"] = conflation_dec_and_score.str[1]

    base_network_column_subset = _preprocess_dataframe(
        base_network, base_attribute_columns
    )
    join_network_column_subset = _preprocess_dataframe(
        join_network, join_attribute_columns
    )
    # END TODO

    # After analysis join our stuff back to the base network
    matched_scores = spatial_similarity_scores
    if detailed_output:
        matched_score_columns = matched_scores.columns
    else:
        matched_score_columns = [
            "base_index",
            "match_index",
            "conflate_decision",
            "final_score",
        ]

    # get method for how we want to match the segments to the base network now
    # we have a scores from the conflation algorithm, now we just need the
    # joining method
    segment_mapping_function = segment_mapping_methods[segment_mapping_method]

    # Return the segments mapped after deciding if they are conflated
    return segment_mapping_function(
        base_network_column_subset,
        join_network_column_subset,
        matched_scores[matched_score_columns],
    )


# TODO handle a road conflating with itself
# TODO work with dictionary inputs to rename things
# a, b = conflate_line_segments(
#     base_network,
#     join_network,
#     base_attribute_columns=["geometry", "osm_id", "fclass", "name"],
#     join_attribute_columns=["geometry", "ROAD_NAME"],
#     # conflation_options
#     conflation_criteria="geometry",
#     debug_output=True,
# )


def seg_to_seg_mapping(
    base_network_column_subset: gpd.GeoDataFrame,
    join_network_column_subset: gpd.GeoDataFrame,
    matched_scores: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    takes best matching segment, it assumes one to one mapping between segmentation
    ONLY to be used if both networks are segmented the same
    """
    # TODO this causes side effects, ideally we should be explicite about what we pass out
    # Drop segments matched that wasn't the best score
    matched_scores.sort_values(by="final_score", ascending=True, inplace=True)
    matched_scores.drop_duplicates(subset="base_index", keep="first", inplace=True)
    # assert matched_scores["final_score"].notna().all()

    all_matches = map_all_candidates(
        base_network_column_subset,
        join_network_column_subset,
        matched_scores[
            matched_scores["conflate_decision"].isin(["match", "candidate match"])
        ],
    )
    assert all_matches[0].shape[0] == base_network_column_subset.shape[0]

    return all_matches


def meter_to_meter_mapping(
    base_network_column_subset: gpd.GeoDataFrame,
    join_network_column_subset: gpd.GeoDataFrame,
    matched_scores: pd.DataFrame,
    tolerance: float = 1.1,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Matching the best meter of network 1 to the best meter of join network, IE 1
    1 meter of base network cannot be joined to more than 1 meter of join network
    per segment"""
    # could probably do this after mapping all candidates
    # get the length of the whole segment to be joined
    # get the length of the subsegments that were matched

    matched_scores = matched_scores[
        matched_scores["conflate_decision"].isin(["match", "candidate match"])
    ]
    base_network_lengths = base_network_column_subset.length.to_frame(
        name="base_seg_len"
    )
    join_network_lengths = join_network_column_subset.length.to_frame(
        name="join_seg_len"
    )
    matched_scores = matched_scores.merge(
        base_network_lengths, how="left", left_on="base_index", right_index=True
    )
    matched_scores = matched_scores.merge(
        join_network_lengths, how="left", left_on="base_index", right_index=True
    )
    nl = LineString([(0, 0), (0, 0)])  # Null linestring
    matched_scores["base_conf_len"] = (
        matched_scores["geometry"]
        .apply(lambda mls: mls.geoms[0] if mls is not None else nl)
        .length
    )
    matched_scores["join_conf_len"] = (
        matched_scores["geometry"]
        .apply(lambda mls: mls.geoms[1] if mls is not None else nl)
        .length
    )

    # sort values so we add roads with good scores first
    matched_scores = matched_scores.sort_values(
        by=["base_index", "final_score"], ascending=True
    )
    base_group = matched_scores.groupby("match_index")
    matched_scores["cum_road_length_conflated"] = base_group["base_conf_len"].cumsum()
    # These are useful for QA later
    matched_scores["total_road_conflated_onto_base"] = base_group["join_conf_len"].sum()
    matched_scores["amount_of_join_segment_conflated"] = matched_scores.groupby(
        "match_index"
    )["join_conf_len"].sum()

    matched_scores["in_best_meters"] = (
        matched_scores["cum_road_length_conflated"]
        <= tolerance * matched_scores["base_seg_len"]
    )

    in_best_meters_review_decision = {
        "match": False,  # its a match and it has a segment
        "candidate match": False,  # it is close to a match and the segment hasnt found a closer match, this is good
    }
    out_best_meters_review_decision = {
        "match": True,  # Was a good match but we have found better matches? review?
        "candidate match": False,  # it was only a partial match and we have better roads
    }

    matched_scores.loc[matched_scores["in_best_meters"], "final_action"] = (
        matched_scores.loc[
            matched_scores["in_best_meters"], "conflate_decision"
        ].replace(in_best_meters_review_decision)
    )
    matched_scores.loc[~matched_scores["in_best_meters"], "final_action"] = (
        matched_scores.loc[
            ~matched_scores["in_best_meters"], "conflate_decision"
        ].replace(out_best_meters_review_decision)
    )

    return map_all_candidates(
        base_network_column_subset,
        join_network_column_subset,
        matched_scores[matched_scores["in_best_meters"]],
    )


def map_potential_matches(
    base_network_column_subset: gpd.GeoDataFrame,
    join_network_column_subset: gpd.GeoDataFrame,
    matched_scores: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    return map_all_candidates(
        base_network_column_subset,
        join_network_column_subset,
        matched_scores[
            matched_scores["conflate_decision"].isin(["match", "candidate match"])
        ],
    )


def map_all_candidates(
    base_network_column_subset: gpd.GeoDataFrame,
    join_network_column_subset: gpd.GeoDataFrame,
    matched_scores: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    # Map everything back to the network
    return (
        pd.merge(
            base_network_column_subset,
            matched_scores,
            left_index=True,
            right_on="base_index",
            how="left",
        ).merge(
            join_network_column_subset.drop(columns="geometry"),
            left_on="match_index",
            right_index=True,
            how="left",
        ),
        matched_scores,
    )


def conflate_non_geometric_attributes(
    spatial_similarity_scores: pd.DataFrame,
    candidate_joins: pd.DataFrame,
    conflation_algorithms: dict[str : Union[Callable, str]],
    conflation_options: dict[str : tuple[str, str, Union[str, Callable]]],
):
    """ """
    for conflation_name, (base_col, match_col, criteria) in conflation_options.items():
        if isinstance(criteria, Callable):
            conf_func = criteria
        elif criteria not in conflation_algorithms:
            raise KeyError(
                f"Conflation optioned passed in that has no known algorithm, expected one of {conflation_algorithms.keys()} or a callable but got {conflation_name}"
            )
        else:
            conf_func = conflation_algorithms[criteria]

        spatial_similarity_scores[criteria] = [
            conf_func(x1, x2)
            for x1, x2 in zip(candidate_joins[base_col], candidate_joins[match_col])
        ]
    # Technically do not need to return spatial similarity scores, but this is more explicit
    return spatial_similarity_scores


# %%
def _join_col_options(
    attribute_columns: Union[list[str], dict[str, str]],
    conf_options: dict[str : tuple[str, str, str]],
    base_or_match: str,
):
    """
    For joining column options for filtering  and including conflation options
    """
    if base_or_match == "base":
        conf_option_unpack_index = 0
    elif base_or_match == "match":
        conf_option_unpack_index = 1
    else:
        raise ValueError(
            "'base' or 'match' must be passed in as the base_or_match variable"
        )

    if isinstance(attribute_columns, list):
        return conf_options + [
            unpack_tup[conf_option_unpack_index] for unpack_tup in conf_options.values()
        ]

    elif isinstance(attribute_columns, dict):
        return conf_options | {
            unpack_tup[conf_option_unpack_index]: unpack_tup[conf_option_unpack_index]
            for unpack_tup in conf_options.values()
        }
    else:
        raise ValueError("attribute_columns must by type list or type dict")


def term_freq_similarity(text1, text2):
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]


def strip_z_coord(geometry):
    return LineString([(x[0], x[1]) for x in geometry.coords])


def coerce_geom_to_linestring(geo_series: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Coerces GeoSeries so that inputs are linestrings.
    """

    geom_type = geo_series.type.unique()
    assert len(geom_type) == 1, "Different Geometries in GeoSeries Passed"
    geom_type = geom_type[0]

    # If the geometries are already LineStrings, return the original GeoSeries
    if geom_type == "LineString":

        return geo_series.apply(strip_z_coord)

    # If the geometries are MultiLineStrings, convert them to LineStrings
    elif geom_type == "MultiLineString":
        # TODO - This should probably be set to false
        return geo_series.explode(index_parts=True).apply(strip_z_coord)

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")


def get_candidate_joins(
    base_df: gpd.GeoDataFrame,
    match_df: gpd.GeoDataFrame,
    base_attribute_columns: Union[list[str], dict[str, str]],
    match_attribute_columns: Union[list[str], dict[str, str]],
    max_matching_distance: float,
    # names used for processing geometry
    base_geom_name: str = "base_geom",
    base_index_name: str = "base_index",
    match_geom_name: str = "match_geom",
    match_index_name: str = "match_index",
) -> pd.DataFrame:
    """
    base_df: dataframe to have attributes mapped onto it
    match_df: dataframe to match attributes to base_df
    base_attribute_columns: attribute columns in base df to be filtered after conflation will keep all columns if empty list
    match_attribute_columns: attribute columns match df to be filtered after conflation will keep all columns if empty list
    conflation_columns: dictionary containing conflation method between columns
    """
    # TODO you shouldnt need to preprocess here
    # just need to make sure each dataframe has geometry and unique index
    base_df = _preprocess_dataframe(base_df, base_attribute_columns)

    match_df = _preprocess_dataframe(match_df, match_attribute_columns)

    for added_column in [
        base_geom_name,
        base_index_name,
        match_geom_name,
        match_index_name,
    ]:
        if added_column in base_df.columns or added_column in match_df.columns:
            raise KeyError(
                f"""Conflict in final names provided in base_attribute_columns or match_attributes_columns, please ensure {added_column} is not in base_attribute_columns or match_attribute_columns"""
            )

    # Getting warnings for below IDK why
    base_df.loc[:, "base_index"] = base_df.index
    base_df.loc[:, "base_geom"] = base_df.geometry
    match_df.loc[:, "match_index"] = match_df.index
    match_df.loc[:, "match_geom"] = match_df.geometry

    base_df.loc[:, "geometry"] = base_df.buffer(max_matching_distance)

    return gpd.sjoin(
        base_df,
        match_df,
        how="inner",
        predicate="intersects",
    )


def _preprocess_dataframe(
    df: gpd.GeoDataFrame, attribute_columns: Union[list[str], dict[str, str]]
):
    """Coerce dataframe"""
    assert df.index.is_unique, "require unique indexes in input dataframes"

    if len(attribute_columns) == 0:
        attribute_columns = list(df.columns)

    if isinstance(attribute_columns, dict):
        # copy to avoid side effects
        df = df.rename(columns=attribute_columns)[list(attribute_columns.values())]
    elif isinstance(attribute_columns, list):
        df = df[attribute_columns]
    else:
        raise TypeError("Base attribute columns should be of type Dict or List")

    df.loc[:, "geometry"] = list(coerce_geom_to_linestring(df.geometry))
    return df


# %%
def fix_line_orientation(
    geoms_to_compare: pd.DataFrame, match_if_directions_reversed: bool
) -> pd.DataFrame:

    segments_have_different_dir = _segments_different_directions(
        geoms_to_compare["base_geom"], geoms_to_compare["match_geom"]
    )
    if match_if_directions_reversed:
        # force directions to be the same direction
        geoms_to_compare.loc[segments_have_different_dir, "match_geom"] = gpd.GeoSeries(
            geoms_to_compare.loc[segments_have_different_dir, "match_geom"]
        ).reverse()
    else:
        # we can remove roads that are reversed direction since we know they
        # are not the same road
        geoms_to_compare = geoms_to_compare[~segments_have_different_dir]

    assert (
        ~_segments_different_directions(
            geoms_to_compare["base_geom"], geoms_to_compare["match_geom"]
        )
    ).all()
    return geoms_to_compare


# %%
def difference_between_shapes(
    geoms_to_compare: pd.DataFrame,
    minimum_segment_overlap: float = 5,
) -> pd.DataFrame:
    """
    get a measurement of the geometric closeness of two dataframes
    see... for methodology

    assumes - indexes are unique
    assumes - there are no - self intersecting line segment geometries
    """
    # ---------- Make all segments Face the correct direction---------------
    print("matching geoms direcions")

    print("processing geom - step 1")
    # -------- Find Relevant Swept Area between Two Road Segments -------
    # TODO add logging points in all these steps
    geoms_to_compare = _add_closest_end_points(geoms_to_compare)
    geoms_to_compare = mark_important_points(geoms_to_compare)

    print("finding boundary of polygon")
    polygon_boundaries = geoms_to_compare.apply(process_row_for_polygon, axis=1)
    geoms_to_compare["geometry"] = polygon_boundaries.apply(
        lambda x: x.geoms[0:2] if x is not None else None
    )
    dot_score = polygon_boundaries.apply(apply_dot_score).clip(lower=0)
    # TODO optimise this step, should be run in < 5 seconds
    # tecnically we dont need this step, but without it linestring_to_poly fails
    # with some examples without this stepsegments_mapped
    print("orienting polygons and finding area")
    oriented_boundary = polygon_boundaries.apply(orient_line_strings)
    spanned_area = gpd.GeoSeries(oriented_boundary.apply(linestring_to_polygon)).area

    print("post processing for output")
    geoms_to_compare["spanned_area"] = spanned_area
    geoms_to_compare["dot_score"] = dot_score
    geoms_to_compare["conflation_length"] = get_length_of_convolved_roads(
        geoms_to_compare
    )
    # ignore segments that only overlap a little bit
    slicer = geoms_to_compare["conflation_length"] <= minimum_segment_overlap

    geoms_to_compare.loc[slicer, "conflation_length"] = np.NaN
    geoms_to_compare.loc[:, "geom_closeness"] = geoms_to_compare["spanned_area"] / (
        geoms_to_compare["conflation_length"] * geoms_to_compare["dot_score"]
    )

    return gpd.GeoDataFrame(
        geoms_to_compare[
            [
                "base_index",
                "match_index",
                "spanned_area",
                "dot_score",
                "conflation_length",
                "geom_closeness",
            ]
        ],
        geometry=geoms_to_compare["geometry"],
    )


def _add_closest_end_points(pairwise_geometries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    finds the closest point of the end of each linestring to its corresponding geometry
    we use this to find the overlapping lengths of of segments to compare spatially
    """

    def _get_nearest_point(row, name_seg_a, name_seg_b, index_to_attack):
        try:
            return nearest_points(
                row[name_seg_a], row[name_seg_b].boundary.geoms[index_to_attack]
            )
        except IndexError:
            # TODO Address the below for comparing roundabouts
            # I suspect occurs if one of the geometries is a perfect circle
            # IE starts an end ends at same point, then len(boundary.geoms) == 0
            # for now we will skip
            return np.NAN

    # we want to find the nearest point on the other line given the current line
    # for each end of each line (4 times)
    get_nearest_point_func_arguments = [
        ("base_geom_P1", ("base_geom", "match_geom", 0)),
        ("base_geom_P2", ("base_geom", "match_geom", 1)),
        ("match_geom_P1", ("match_geom", "base_geom", 0)),
        ("match_geom_P2", ("match_geom", "base_geom", 1)),
    ]

    for column_name, args in get_nearest_point_func_arguments:
        pairwise_geometries[column_name] = pairwise_geometries.apply(
            lambda row: _get_nearest_point(row, *args), axis=1
        )

    return pairwise_geometries


def _segments_different_directions(s1: gpd.GeoSeries, s2: gpd.GeoSeries) -> pd.Series:
    """
    given two linestring inputs, returns true if they are facing the same way,
    this is calculated if the dot product of the start to end of the line sting is
    negative or positive
    """

    # TODO Im pretty sure there is a more concise and neat way to do this but this works
    assert (s1.index == s2.index).all()
    s1_starts = s1.apply(lambda g: g.coords[0])
    s1_ends = s1.apply(lambda g: g.coords[-1])
    s2_starts = s2.apply(lambda g: g.coords[0])
    s2_ends = s2.apply(lambda g: g.coords[-1])

    s1_x_vec = s1_ends.str[0] - s1_starts.str[0]
    s1_y_vec = s1_ends.str[1] - s1_starts.str[1]
    s2_x_vec = s2_ends.str[0] - s2_starts.str[0]
    s2_y_vec = s2_ends.str[1] - s2_starts.str[1]

    # find the dot product to find out if roads are in same direction
    dot_product = s1_x_vec * s2_x_vec + s1_y_vec * s2_y_vec

    # if dot product is less than zero the roads are travelling in oposite directions
    # one of the
    return dot_product < 0


def mark_important_points(geom_with_end_points: pd.DataFrame) -> pd.DataFrame:
    """
    Cut line segments so there relative similarity can be compared for the
    relevant length
    """

    def _get_length_between_tup_in_pd_series(
        s: pd.Series,  # [tuple[Point, Point]]
    ) -> pd.Series:
        """Given a series of tuples of points, find the distance between the two points"""
        P1 = gpd.GeoSeries(s.str[0])
        P2 = gpd.GeoSeries(s.str[1])
        return P1.distance(P2)

    def _add_column_with_smallest_distance_from_subset(
        df: pd.DataFrame,
        columns_to_compare: list[str],
        final_column_name: str,
    ) -> pd.Series:
        """
        adds a new column to the dataframe based on the smallest internal distance
        of any of the columns to compare
        """
        # fill with non types so pandas does no know how much memory to reserve
        df[final_column_name] = [None] * len(df)

        lengths = []
        for name in columns_to_compare:
            lengths.append(_get_length_between_tup_in_pd_series(df[name]))
        end_lengths = pd.concat(lengths, axis=1)
        min_index = end_lengths.idxmin(axis=1)
        for index, column_name in enumerate(columns_to_compare):
            slicer = min_index == index
            df.loc[slicer, final_column_name] = df.loc[slicer, column_name]
        return df

    # Ensures No side effects, can probably be removed to improve performance
    return_df = geom_with_end_points.copy()

    return_df = _add_column_with_smallest_distance_from_subset(
        return_df,
        ["base_geom_P1", "match_geom_P1"],
        "start_of_spanned_area",
    )
    return_df = _add_column_with_smallest_distance_from_subset(
        return_df,
        ["base_geom_P2", "match_geom_P2"],
        "end_of_spanned_area",
    )

    return return_df


def process_row_for_polygon(
    row: pd.Series, snap_tolerance: float = 0.01
) -> MultiLineString:
    """
    processes geometry to create boundary of polygon in terms of a multilinesting
    primarily focused around trimming the roads about the relevant sections to compare
    """

    def _get_bounding_geom(geoms: MultiLineString, points: MultiPoint):
        """
        after we split the geometry at a set of points, the only points
        we should include are in either end of the boundary
        """
        # warning comparing floats in below code for equality, for now shapely handles
        # it

        relevant_geoms = [
            geom
            for geom in geoms.geoms
            if (geom.boundary.geoms[0] in points.geoms)
            and (geom.boundary.geoms[1] in points.geoms)
        ]
        if len(relevant_geoms) == 1:
            return LineString(relevant_geoms[0])
        elif len(relevant_geoms) > 1:
            # we returned Multiple geoms, we want them merged into 1
            # Note assuming the split() method does not change linestring order
            return LineString(
                [point for geom in relevant_geoms for point in geom.coords]
            )
        else:
            raise Exception("Didnt Find Geom in boundary")

    # print(row["base_index"])
    base_geom = row["base_geom"]
    match_geom = row["match_geom"]
    points = MultiPoint(
        [
            row["start_of_spanned_area"][0],
            row["start_of_spanned_area"][1],
            row["end_of_spanned_area"][0],
            row["end_of_spanned_area"][1],
        ]
    )
    list_points = list(points.geoms)
    if dwithin(
        row["start_of_spanned_area"][0],
        MultiPoint(list_points[2:4]),
        snap_tolerance * 2,
    ) or dwithin(
        row["start_of_spanned_area"][1],
        MultiPoint(list_points[2:4]),
        snap_tolerance * 2,
    ):
        # one link is entirely past teh end of another link and
        # the mapping length is zero, we can ignore this case
        return None

    for point in points.geoms:
        base_geom = snap(base_geom, point, snap_tolerance)
        match_geom = snap(match_geom, point, snap_tolerance)
    # return gpd.GeoSeries([base_geom, match_geom, points]).explore()

    split_base = split(base_geom, points)
    relevant_base = _get_bounding_geom(split_base, points)

    split_match = split(match_geom, points)
    relevant_match = _get_bounding_geom(split_match, points)

    return MultiLineString(
        [
            relevant_base,
            relevant_match,
            LineString(row["start_of_spanned_area"]),
            LineString(row["end_of_spanned_area"]),
        ]
    )


def get_boundary_of_polygon(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def try_catch_apply(input):
        try:
            return process_row_for_polygon(input)
        except Exception:
            return np.NAN

    return df.apply(try_catch_apply, axis=1)


def orient_line_strings(multi_geom: Union[None, MultiLineString]) -> Polygon:
    """
    given a set of linestrings that form the boundary of a polygon, this
    will stitch them together into a multipolygon
    """
    # ops.poligonize should do this but doesnt work for some reason (probably orientation)
    if multi_geom is None:
        return None

    num_geoms = len(multi_geom.geoms)

    # if there are more then 4 geoms this will lead to significant slow downs
    # this algorithm slows down exponentially with each additional geometry
    assert num_geoms <= 4

    # there are faster ways to do this but for now we will find the smallest
    # length of boundaries out of all combinations of attatching the components
    # head to tail
    # POTENTIAL OPTIMISATION: We can double the speed because we know the orientation
    # of one of the geometries

    if num_geoms == 4:
        template = list(multi_geom.geoms)
        permuted_list = [
            template,
            [template[0], template[1], template[3], template[2]],
            [template[0], template[3], template[1], template[2]],
        ]
    else:
        permuted_list = [list(multi_geom.geoms)]

    for geom_perm in permuted_list:
        final_linestrings = []
        for binary_mask in range(2**num_geoms):
            reversed_geoms = [char == "1" for char in f"{binary_mask:04b}"][
                (4 - num_geoms) :
            ]
            final_geoms = [
                geom.reverse() if reverse_geom else geom
                for geom, reverse_geom in zip(geom_perm, reversed_geoms)
            ]
            flattened_list = [point for geom in final_geoms for point in geom.coords]
            single_linestring = LineString(flattened_list)
            final_linestrings.append(single_linestring)

    # print([g.length for g in final_linestrings])

    return min(final_linestrings, key=lambda x: x.length)


# def linestrings_to_polygon(s: gpd.GeoSeries) -> gpd.GeoDataFrame:
#     # TODO delete this preferring a straight apply
#     def try_catch_apply(input):
#         try:
#             return orient_line_strings(input)
#         except Exception:
#             return np.NAN

#     return gpd.GeoSeries(s.apply(try_catch_apply))


def get_length_of_convolved_roads(df: gpd.GeoDataFrame) -> list:
    """
    returns length of
    """
    # there are slightly better ways of doing this, for now we will take the
    # straight line path
    # if the roads are very cuvey, it may under - represent the roads match
    start_conv = df["start_of_spanned_area"].apply(LineString)
    end_conv = df["end_of_spanned_area"].apply(LineString)
    nearest_point_pairs = [
        nearest_points(start, end) for start, end in zip(start_conv, end_conv)
    ]
    return [LineString(point_pair).length for point_pair in nearest_point_pairs]


def get_dot_score(segment_1: LineString, segment_2, num_interp_points: int = 10):

    def _get_normalised_vector(p1: Point, p2: Point) -> Point:
        """Returns the normalized vector (p1 - p2) / ||p1 - p2||."""
        # Calculate the difference vector (dx, dy)
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        magnitude = (dx**2 + dy**2) ** 0.5
        return Point(dx / magnitude, dy / magnitude)

    # TODO remove this into a generic function
    def dot_product(p1: Point, p2: Point):
        try:
            return p1.x * p2.x + p1.y * p2.y
        except Exception:
            print("failed dot product")
            return 0.8

    # might have to change to cumulative multiply
    road_a_points = [
        segment_1.interpolate(dist)
        for dist in np.linspace(0, segment_1.length, num_interp_points)
    ]
    road_b_points = [
        segment_2.interpolate(dist)
        for dist in np.linspace(0, segment_2.length, num_interp_points)
    ]

    # travel along the road and average out all the dot products
    # I think it is technically more correct to do np.cumproduct ** (1/num_interp_points)
    return (
        sum(
            [
                max(
                    0,
                    dot_product(
                        _get_normalised_vector(a1, a2), _get_normalised_vector(b1, b2)
                    ),
                )
                for a1, a2, b1, b2 in zip(
                    road_a_points[1:],
                    road_a_points[:-1],
                    road_b_points[1:],
                    road_b_points[:-1],
                )
            ]
        )
        / num_interp_points
    )


def apply_dot_score(mult_geom: Union[None, MultiLineString]):
    # we can do this because we know the road segments are in slot 0 and 1 of multi geom
    if mult_geom is None:
        return None
    return get_dot_score(mult_geom.geoms[0], mult_geom.geoms[1])


# %%


def linestring_to_polygon(geom: LineString) -> Polygon:
    if geom is None:
        return None
    multi_line = unary_union(geom)
    if isinstance(multi_line, LineString):
        coord_list = list(multi_line.coords)
        if len(coord_list) == 2:
            first_coord = coord_list[0]
            second_coord = coord_list[1]
            return Polygon([first_coord, second_coord, second_coord, first_coord])
        try:
            return Polygon(multi_line)
        except Exception:
            print(multi_line)
            print("error converting polygon")
            return None
    elif isinstance(multi_line, MultiLineString):
        return polygonize(list(multi_line.geoms))
    else:
        raise Exception("Non Linestring Geometry Passed")


# %%
