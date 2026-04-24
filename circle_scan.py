"""circle_scan.py
Multi-Diameter Monte Carlo Circular Scan-Area Sampling
FracArea© Andrea Bistacchi & Stefano Casiraghi, 2026
"""

# --- Initialize script ---
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as mpl_circle
from matplotlib.patches import Patch as mpl_patch
from matplotlib.lines import Line2D as mpl_line
from scipy.stats import levene, f_oneway, shapiro
import scipy.stats as stats
import os
from joblib import Parallel, delayed
import multiprocessing
import re
import seaborn as sns
import time
from datetime import datetime

# --- Start timing ---
analysis_start_time = time.time()

# --- SET ANALYSIS OPTIONS HERE ---

# --- Runtime options ---
GENERATE_DATA = True
PLOT_FIGURES = True  # Set True to generate per-diameter maps
PLOT_HISTOGRAMS = True  # Set True to generate per-diameter histograms
PLOT_CUM_BOXPLOT = True # Set True to generate cumulative box/line plots for all diameters and realizations
PLOT_BOXPLOTS = True  # Set True to generate box/line plots for each realization separately
CREATE_GIFS = True  # Set True to create GIFs from per-diameter maps and histograms

alpha = 0.05  # Significance level for statistical tests

# --- Base analysis parameters PONTRELLI ---
input_folder = "data_pontrelli"
boundary_file = os.path.join(input_folder, "Interpretation-boundary.shp")
lineaments_file = os.path.join(input_folder, "FN_set_1.shp")

n_iterations = 100          # Monte Carlo iterations per diameter
n_points_target = 100       # Circles per iteration

# Min and Max diameter for analysis always required. Intermediate diameter ans n_steps used to have more points
# in the smaller diameter range, where more variability is expected. Only used if spacing_type is "linear".
diameter_min = 2 # Minimum circle diameter
diameter_max = 52 # Maximum circle diameter
n_steps = 26
spacing_type = "linear"  # Options: "linear", "exponential", "log"
diameter_intermediate = False
n_steps_intermediate = False
# diameter_intermediate = 0.6
# n_steps_intermediate = 10 # Number of radii

# --- END ANALYSIS OPTIONS HERE ---

#################################################################################################
# Functions are below this line


def get_next_output_folder(base_folder=None):
    """
    Determines the next sequentially numbered folder within the specified base
    folder. If the latest folder is empty, it will be reused. Otherwise, a new
    folder with the next sequential number is created.
    """
    if not os.path.exists(base_folder):
        print(f"ERROR: Folder not found: {base_folder}")
        exit(1)
    existing = [
        f
        for f in os.listdir(base_folder)
        if re.match(r"^\d{3}$", f) and os.path.isdir(os.path.join(base_folder, f))
    ]
    if not existing:
        next_num = 1
    else:
        nums = sorted([int(f) for f in existing])
        last_num = nums[-1]
        last_folder = os.path.join(base_folder, f"{last_num:03d}")
        # If the last folder is empty, use it
        if not os.listdir(last_folder):
            return last_folder
        next_num = last_num + 1
    next_folder = os.path.join(base_folder, f"{next_num:03d}")
    os.makedirs(next_folder, exist_ok=True)

    print(f"Output of this analysis will be saved in: {next_folder}")

    return next_folder


def load_and_validate_data(boundary_file=None, lineaments_file=None):
    """
    Loads and validates geographical data consisting of boundary polygons and lineament lines.
    The function performs the following:
    - Validates the file paths for both boundary and lineament data.
    - Reads and checks the validity of boundary polygon(s), ensuring it contains only one valid polygon.
    - Reads and checks the validity of lineaments, ensuring they are line geometries with a compatible CRS.
    - Computes the maximum inscribed circle for the boundary polygon and its associated properties.
    - Optionally visualizes and saves plots of the boundary polygon, circle, center point, and other related features.

    :param boundary_file: Path to the boundary file to validate and analyze.
    :param lineaments_file: Path to the lineaments file to validate and analyze.
    :return: Tuple containing boundary geodataframe, lineaments geodataframe, and maximum admissible diameter.
    :rtype: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, float]
    """
    # --- Load Data ---
    data_load_t0 = time.perf_counter()
    if not os.path.exists(boundary_file):
        print(f"ERROR: Boundary file not found: {boundary_file}")
        exit(1)
    if not os.path.exists(lineaments_file):
        print(f"ERROR: Lineaments file not found: {lineaments_file}")
        exit(1)

    bnd_gdf = gpd.read_file(boundary_file)
    if bnd_gdf.shape[0] == 0:
        print(f"ERROR: Boundary polygon is empty: {boundary_file}")
        exit(1)
    if bnd_gdf.shape[0] > 1:
        print(f"WARNING: Boundary polygon has multiple polygons: {boundary_file}")
        exit(1)
    if bnd_gdf.geometry.type.iloc[0] != "Polygon":
        print(f"ERROR: Boundary polygon is not a polygon: {boundary_file}")
        exit(1)

    lineaments_gdf = gpd.read_file(lineaments_file).to_crs(bnd_gdf.crs)
    if lineaments_gdf.shape[0] == 0:
        print(f"ERROR: Lineaments file is empty: {lineaments_file}")
        exit(1)
    if lineaments_gdf.geometry.type.iloc[0] not in ["LineString", "MultiLineString"]:
        print(f"ERROR: Lineaments file is not a line: {lineaments_file}")
        exit(1)

    print(
        f"✅ Loaded data: {len(bnd_gdf)} polygons, {len(lineaments_gdf)} lineaments in {time.perf_counter() - data_load_t0:.3f}s"
    )

    # --- Compute maximum admissible circle diameter for the boundary polygon ---
    max_circle = shapely.maximum_inscribed_circle(
        bnd_gdf.geometry.iloc[0]
    )  #########################
    center_coords = list(max_circle.coords)[0]
    center_pt = shapely.geometry.Point(center_coords)
    diameter_coords = list(max_circle.coords)[1]
    diameter_pt = shapely.geometry.Point(diameter_coords)
    max_admissible_diameter = max_circle.length * 2
    print(
        f"Center point of maximum inscribed circle: ({center_pt.x:.2f}, {center_pt.y:.2f})"
    )
    print(
        f"Circle point of maximum inscribed circle: ({diameter_pt.x:.2f}, {diameter_pt.y:.2f})"
    )
    print(f"Maximum admissible circle diameter: {max_admissible_diameter:.2f}")

    # --- Plot boundary, max circle, center, and first point ---
    if PLOT_FIGURES:
        fig_1, ax1 = plt.subplots(figsize=(8, 8))
        bnd_gdf.plot(ax=ax1, facecolor="none", edgecolor="black", linewidth=1.5)
        circle_patch = mpl_circle(
            (center_pt.x, center_pt.y),
            max_admissible_diameter / 2,
            edgecolor="blue",
            facecolor="none",
            linewidth=2,
            label="Max Admissible Circle",
        )
        ax1.plot(center_pt.x, center_pt.y, "o", color="green", markersize=12)
        ax1.add_patch(circle_patch)
        legend_handles = [
            mpl_line([0], [0], color="black", lw=1.5, label="Boundary"),
            mpl_patch(
                edgecolor="blue",
                facecolor="none",
                linewidth=2,
                label="Max Admissible Circle",
            ),
            mpl_line(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=12,
                label="Circle Center",
            ),
        ]
        plt.title("Boundary, Maximum Inscribed Circle, Center, and First Point")
        plt.legend(handles=legend_handles)
        plt.axis("equal")
        plt.tight_layout()
        fig_path_png = os.path.join(output_folder, "boundary_max_circle.png")
        fig_path_svg = os.path.join(output_folder, "boundary_max_circle.svg")
        fig_1.savefig(fig_path_png, dpi=300)
        fig_1.savefig(fig_path_svg, format="svg")
        print(
            f"✅ Boundary and max inscribed circle figure saved: {fig_path_png}, {fig_path_svg}"
        )

    return bnd_gdf, lineaments_gdf, max_admissible_diameter


def generate_diameter_list(
    diameter_min=None,
    diameter_max=None,
    n_steps=None,
    spacing_type=None,
    max_admissible_diameter=None,
):
    """
    Generates a list of radii based on the given parameters and spacing type. The
    function allows for 'linear', 'exponential', or 'log' spacing to distribute the
    diameter values. Any diameter values that exceed the specified maximum admissible
    diameter are filtered out.

    :param diameter_min: The minimum diameter value to start from.
    :param diameter_max: The maximum diameter value to end on.
    :param n_steps: The number of steps or intervals for dividing the range.
    :param spacing_type: The spacing method to use; options are 'linear',
        'exponential', or 'log'.
    :param max_admissible_diameter: The maximum allowable diameter value; any
        values above this will be excluded.
    :return: A list of diameter values distributed as per the specified spacing type.
    """
    if spacing_type == "linear":
        if diameter_intermediate and n_steps_intermediate:
                diameter_list_1 = np.linspace(diameter_min, diameter_intermediate, n_steps_intermediate)
                diameter_list_2 = np.linspace(diameter_intermediate, diameter_max, n_steps)
                diameter_list = np.unique(np.concatenate((diameter_list_1, diameter_list_2)))
        else:
            diameter_list = np.linspace(diameter_min, diameter_max, n_steps)


    elif spacing_type == "exponential":
        diameter_list = np.geomspace(diameter_min, diameter_max, n_steps)
    elif spacing_type == "log":
        diameter_list = np.logspace(
            np.log10(diameter_min), np.log10(diameter_max), n_steps
        )
    else:
        raise ValueError(
            "Invalid spacing_type: choose 'linear', 'exponential', or 'log'"
        )

    diameter_list = [d for d in diameter_list if d <= max_admissible_diameter]
    print(
        f"Diameters to process ({spacing_type} spacing): {np.round(diameter_list, 2)}"
    )

    return diameter_list


def run_iteration(
    iter_id=None,
    inner_poly=None,
    lineaments_gdf=None,
    n_points_target=None,
    circle_diameter=None,
    crs=None,
):
    """
    Runs an iteration to generate random circles within a polygon, compute their intersection
    with given lineaments, and calculate related geometric metrics.

    :param iter_id: The ID of the current iteration.
    :type iter_id: int
    :param inner_poly: The inner polygon within which random points and circles are generated.
    :type inner_poly: shapely.geometry.Polygon
    :param lineaments_gdf: A GeoDataFrame containing lineaments (geometric features) that will be
        spatially joined with generated circles.
    :type lineaments_gdf: geopandas.GeoDataFrame
    :param n_points_target: The target number of random points to generate within the inner polygon.
    :type n_points_target: int
    :param circle_diameter: The diameter of the circles to be created around sampled points.
    :type circle_diameter: float
    :param crs: The coordinate reference system (CRS) used for the generated GeoDataFrame objects.
    :type crs: str or pyproj.CRS

    :return: A GeoDataFrame containing the generated circles, including their spatial metrics:
        clipped lengths of lineament intersections, areas, coordinates, and IDs.
    :rtype: geopandas.GeoDataFrame
    """
    # Random points inside inner polygon
    minx, miny, maxx, maxy = inner_poly.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    poly_area = inner_poly.area
    scaling_factor = poly_area / bbox_area * 1.1 if bbox_area > 0 else np.nan

    # Handle empty or invalid inner polygon
    if poly_area == 0 or np.isnan(scaling_factor) or np.isinf(scaling_factor):
        # Return circles_gdf with zero clipped_length and other columns
        circles_gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
        circles_gdf["diameter"] = []
        circles_gdf["area"] = []
        circles_gdf["realization_id"] = []
        circles_gdf["circle_id"] = []
        circles_gdf["clipped_length"] = []
        circles_gdf["x"] = []
        circles_gdf["y"] = []
        circles_gdf["P21"] = []
        circles_gdf['residuals'] = []
        return circles_gdf

    n_points_to_generate = (
        int(n_points_target / scaling_factor * 1.5)
        if not np.isnan(scaling_factor) and not np.isinf(scaling_factor)
        else n_points_target * 2
    )
    xs = np.random.uniform(minx, maxx, n_points_to_generate)
    ys = np.random.uniform(miny, maxy, n_points_to_generate)
    points = gpd.GeoDataFrame(
        geometry=[shapely.geometry.Point(x, y) for x, y in zip(xs, ys)], crs=crs
    )
    points_within = points[points.within(inner_poly)].reset_index(drop=True)
    points_sample = points_within.sample(
        n_points_target, random_state=None
    ).reset_index(drop=True)

    # Create circles with a given diameter, using a buffer of diameter/2 to create a circle around each point.
    circles_gdf = gpd.GeoDataFrame(
        geometry=points_sample.geometry.buffer(circle_diameter / 2, resolution=360), crs=crs
    ).reset_index(drop=True)
    circles_gdf["circle_id"] = circles_gdf.index
    circles_gdf["realization_id"] = iter_id

    # Spatial join of circles with lineaments
    joined = gpd.sjoin(
        lineaments_gdf, circles_gdf[["circle_id", "geometry"]], predicate="intersects"
    )

    # If no lineaments intersect any circles, return circles_gdf with zero clipped_length
    if joined.empty:
        circles_gdf["diameter"] = circle_diameter
        circles_gdf["area"] = circles_gdf.geometry.area
        circles_gdf["clipped_length"] = 0
        circles_gdf["x"] = points_sample.geometry.x
        circles_gdf["y"] = points_sample.geometry.y
        circles_gdf["P21"] = 0
        circles_gdf['residuals'] = 0
        return circles_gdf

    # Vectorized intersection and length using Shapely 2.0 APIs
    # Align each lineament geometry with its corresponding circle geometry
    circle_geom_arr = circles_gdf.geometry.values
    circle_ids = joined["circle_id"].to_numpy()
    joined_circle_geoms = circle_geom_arr[circle_ids]

    # Compute intersections and lengths in vectorized form
    inter_geoms = shapely.intersection(joined.geometry.values, joined_circle_geoms)
    lengths_arr = shapely.length(inter_geoms)
    # Set length to 0 where intersection is empty
    empty_mask = shapely.is_empty(inter_geoms)
    if empty_mask.any():
        lengths_arr = np.where(empty_mask, 0.0, lengths_arr)

    # Aggregate and merge
    joined["clipped_length"] = lengths_arr
    lengths_per_circle = (
        joined.groupby("circle_id")["clipped_length"].sum().reset_index()
    )
    circles_gdf = circles_gdf.merge(lengths_per_circle, on="circle_id", how="left")
    circles_gdf["diameter"] = circle_diameter
    circles_gdf["area"] = circles_gdf.geometry.area
    circles_gdf["clipped_length"] = circles_gdf["clipped_length"].fillna(0)
    circles_gdf["x"] = points_sample.geometry.x
    circles_gdf["y"] = points_sample.geometry.y
    circles_gdf["P21"] = circles_gdf["clipped_length"] / circles_gdf["area"]
    circles_gdf['residuals'] = circles_gdf['P21'] - circles_gdf['P21'].mean()

    circles_gdf = circles_gdf[
        [
            "diameter",
            "area",
            "realization_id",
            "circle_id",
            "clipped_length",
            "x",
            "y",
            "P21",
            "geometry",
            "residuals",
        ]
    ]

    return circles_gdf


def process_diameter(diameter=None, bnd_gdf=None, lineaments_gdf=None, map_figs_list=None):
    """
    Processes the given diameter by performing several operations including
    polygon buffering, data concatenation, and file saving. Iterations are
    conducted to compute results, stored in a geodataframe, which gets saved
    to a CSV file. Optionally, plots can be generated for visualization if
    enabled by a global flag.

    :param bnd_gdf:
    :param lineaments_gdf:
    :param diameter: The diameter value (in meters) to process.
    :type diameter: float
    :return: Geodataframe containing the results for the given diameter.
    :rtype: geopandas.GeoDataFrame
    """
    # print(f"\n▶️ Diameter = {diameter:.2f}")
    t0 = time.perf_counter()
    buffer_distance = -diameter / 2

    # Inner polygon for this diameter (do not mutate bnd_gdf)
    inner_poly = bnd_gdf.geometry.buffer(buffer_distance, resolution=360).union_all()

    # Run all iterations (sequential within this diameter)
    iter_t0 = time.perf_counter()
    results_list = []
    for i in range(1, n_iterations + 1):
        result = run_iteration(
            iter_id=i,
            inner_poly=inner_poly,
            lineaments_gdf=lineaments_gdf,
            n_points_target=n_points_target,
            circle_diameter=diameter,
            crs=bnd_gdf.crs,
        )
        # deep copy the result geodataframe for this iteration, to be plotted at the end of the diameter loop
        last_iteration_result = result.copy(deep=True)
        # drop geometry and other columns to save memory
        # result['residuals'] = result['P21'] - result['P21'].mean() # moved to run_iteration

        result.drop(
            ["geometry", "area", "circle_id", "clipped_length"], axis=1, inplace=True
        )
        results_list.append(result)
    # print(f"  Iterations done in {time.perf_counter()-iter_t0:.3f}s")

    # Combine iterations for this diameter
    concat_t0 = time.perf_counter()
    results_gdf = pd.concat(results_list, ignore_index=True)
    # print(f"  Concatenation in {time.perf_counter()-concat_t0:.3f}s")

    if PLOT_FIGURES:
        # Plot colored by P21
        fig, ax = plt.subplots(figsize=(8, 8))
        bnd_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            label="Boundary",
        )
        gpd.GeoSeries(inner_poly).plot(
            ax=ax,
            facecolor="none",
            edgecolor="red",
            linestyle="--",
            label="Buffer",
        )
        lineaments_gdf.plot(ax=ax, color="gray", linewidth=0.5, label="Lineaments")
        last_iteration_result.plot(
            ax=ax,
            column="P21",
            cmap="viridis",
            legend=True,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.3,
            label="Circles",
            # legend_kwds={"label": "P21", "orientation": "vertical"}
        )
        # Custom legend handles
        legend_handles = [
            mpl_line([0], [0], color="black", lw=1.2, label="Boundary"),
            mpl_line([0], [0], color="red", lw=1.2, linestyle="--", label="Buffer"),
            mpl_line([0], [0], color="gray", lw=0.5, label="Lineaments"),
            mpl_patch(
                edgecolor="black",
                facecolor="none",
                linewidth=0.3,
                label="Circular scanareas",
            ),
        ]
        plt.title(
            f"Diameter = {diameter:.2f} m — Circular scanareas from last Iteration colored by P21"
        )
        plt.legend(handles=legend_handles, loc="upper left")
        plt.axis("equal")
        plt.tight_layout()

        # Save figures
        fig_path_png = os.path.join(output_folder, f"map_scanareas_d{diameter:.2f}.png")
        fig_path_svg = os.path.join(output_folder, f"map_scanareas_d{diameter:.2f}.svg")
        fig.savefig(fig_path_png, dpi=300)
        fig.savefig(fig_path_svg, format="svg")
        plt.close()
        # print(f"  Figures saved in diameter loop")

    return results_gdf


def run_levene_test(
    alldata_df=None,
    diameter_column="diameter",
    realization_column="realization_id",
    tested_variable="residuals",
    alpha=alpha,
):
    """
    Perform Levene's test for equality of variances between consecutive radius groups.

    Parameters:
        alldata_df (DataFrame): The dataframe containing P21 values and radii.
        diameter_column (str): The name of the column containing radius values.
        tested_variable (str): The name of the column containing P21 values.
        alpha (float): Significance level for the test.

    Returns:
        DataFrame: A DataFrame with radius pairs, test statistic, p-values, and result interpretation.

    IF NEEDED, TO RUN THIS IN PARALLEL, WE CAN ALTERNATIVELY
    FIX THE realization TO EQUAL TO A VALUE AND THEN
    FEED THE FUNCTION TO A PARALLEL LOOP
    """
    # rows_levene = []
    #
    # diameter_sorted = sorted(alldata_df[diameter_column].unique())
    #
    # for groups_in_step in range(2, n_steps-1, 1):
    #     # print(f"Levene groups_in_step: {groups_in_step}")
    #
    #     #calculate group size
    #
    #     group_size_y = groups_in_step +1
    #
    #     for dia_min_idx in range(n_steps - groups_in_step):
    #         dia_min = diameter_sorted[dia_min_idx]
    #         dia_max = diameter_sorted[dia_min_idx + groups_in_step]
    #         # print(f"dia_min_idx: {dia_min_idx}, dia_min: {dia_min}, dia_max: {dia_max}")
    #         # print(" ")
    #
    #         for realization in alldata_df[realization_column].unique():
    #             group1 = alldata_df[
    #                 (alldata_df[diameter_column] == dia_min)
    #                 & (alldata_df[realization_column] == realization)
    #             ][tested_variable]
    #             group2 = alldata_df[
    #                 (alldata_df[diameter_column] == dia_max)
    #                 & (alldata_df[realization_column] == realization)
    #             ][tested_variable]
    #
    #             stat_levene, p_levene = levene(group1, group2, center="median")
    #
    #             result = (
    #                 "Different variances"
    #                 if p_levene < alpha
    #                 else "No significant difference"
    #             )
    #
    #             rows_levene.append(
    #                 {
    #                     "dia_min": dia_min,
    #                     "dia_max": dia_max,
    #                     "y": group_size_y,
    #                     "realization": realization,
    #                     "statistic": stat_levene,
    #                     "p_value": p_levene,
    #                     "result": result,
    #                 }
    #             )
    #
    #             # print(
    #             #     f"Levene test between diameters {dia_min} and {dia_max}, realization {realization}:"
    #             # )
    #             # print(f"  Statistic = {stat_levene:.4f}, p-value = {p_levene:.4f}")
    #             # print(f"  Result: {result}")
    #             # print("\n")
    #
    # return pd.DataFrame(rows_levene)

    rows_levene = []
    diameter_sorted = sorted(alldata_df[diameter_column].unique())

    # --- CHANGE: range(2, ...) ensures the first loop is a window of 3 points ---
    for groups_in_step in range(2, n_steps - 1):
        group_size_y = groups_in_step + 1 # offset of 2 indices = 3 diameters

        for dia_min_idx in range(n_steps - groups_in_step):
            dia_min = diameter_sorted[dia_min_idx]
            dia_max = diameter_sorted[dia_min_idx + groups_in_step]

            for realization in alldata_df[realization_column].unique():
                # Filter data for this realization and diameter pair
                data_dia_real = alldata_df[
                    (
                            (alldata_df[diameter_column] <= dia_max)
                            & (alldata_df[diameter_column] >= dia_min)
                    )
                    & (alldata_df[realization_column] == realization)
                ]
                grouped = [group[tested_variable].values for _, group in data_dia_real.groupby(diameter_column)]
                if len(grouped) < 2:
                    continue
                # group2 = alldata_df[
                #     (alldata_df[diameter_column] == dia_max)
                #     & (alldata_df[realization_column] == realization)
                # ][tested_variable]

                # Ensure valid data exists before testing
                if not data_dia_real.empty:
                    stat_levene, p_levene = levene(*grouped)
                    result = "Different variances" if p_levene < alpha else "No significant difference"

                    rows_levene.append({
                        "dia_min": dia_min,
                        "dia_max": dia_max,
                        "y": group_size_y, # Now correctly shows '3' for the first plot
                        "realization": realization,
                        "statistic": stat_levene,
                        "p_value": p_levene,
                        "result": result,
                    })

    return pd.DataFrame(rows_levene)


def check_normality_error_variables(
        alldata_df=None,
        diameter_column="diameter",
        realization_column="realization_id",
        tested_variable="residuals",
        alpha=alpha,
):
    """
    Perform Shapiro-Wilk test for normality on residuals for each radius group.

    Parameters:
        alldata_df (DataFrame): DataFrame containing residuals and radii.
        diameter_column (str): Column name for radii.
        residuals_column (str): Column name for residuals.
        alpha (float): Significance level.

    Returns:
        DataFrame: Results with radius, statistic, p-value, and interpretation.

    IF NEEDED, TO RUN THIS IN PARALLEL, WE CAN ALTERNATIVELY
    FIX THE realization TO EQUAL TO A VALUE AND THEN
    FEED THE FUNCTION TO A PARALLEL LOOP
    """
    rows_residuals = []

    diameter_sorted = sorted(alldata_df[diameter_column].unique())

    for dia in diameter_sorted:
        for realization in alldata_df[realization_column].unique():
            data_dia_real = alldata_df[
                (alldata_df[diameter_column] == dia)
                & (alldata_df[realization_column] == realization)
                ][tested_variable]

            if len(data_dia_real) < 3:
                # Shapiro-Wilk requires at least 3 observations
                rows_residuals.append(
                    {
                        "dia": dia,
                        "realization": realization,
                        "statistic": None,
                        "p_value": None,
                        "result": "Too few samples",
                    }
                )
                print(f"Diameter {dia} skipped due to too few samples for Shapiro-Wilk.")
                continue

            stat_residuals, p_val_residuals = shapiro(data_dia_real)
            result = "Not normal" if p_val_residuals < alpha else "Normal"

            rows_residuals.append(
                {
                    "dia": dia,
                    "realization": realization,
                    "statistic": stat_residuals,
                    "p_value": p_val_residuals,
                    "result": result,
                }
            )

    return pd.DataFrame(rows_residuals)


def run_anova_test(
    alldata_df=None,
    dia_min=None,
    dia_max=None,
    diameter_column="diameter",
    realization_column="realization_id",
    tested_variable="P21",
    alpha=alpha,
):
    """
    Perform one-way ANOVA to test if mean P21 values differ across diameter groups.

    Parameters:
        alldata_df (DataFrame): Input dataframe with P21 values and diameter.
        dia_min - dia_max to filter the REV range.
        diameter_column (str): Name of the radius column.
        tested_variable (str): Name of the P21 value column.
        alpha (float): Significance threshold (default 0.05).

    Returns:
        DataFrame: A one-row DataFrame with F-statistic, p-value, result label, and tested range.
    """
    rows_anova = []

    for realization in alldata_df[realization_column].unique():
        filtered = alldata_df[
            (alldata_df[diameter_column] >= dia_min)
            & (alldata_df[diameter_column] <= dia_max)
            & (alldata_df[realization_column] == realization)
        ]

        grouped = [
            group[tested_variable].values for _, group in filtered.groupby(diameter_column)
        ]

        f_stat, p_val = f_oneway(*grouped)
        result = (
            "At least one group mean is different"
            if p_val < alpha
            else "No significant difference"
        )

        rows_anova.append(
            {
                "dia_min": dia_min,
                "dia_max": dia_max,
                "realization": realization,
                "f_statistic": f_stat,
                "p_value": p_val,
                "result": result,
            }
        )

        print(f"ANOVA test on diameters between {dia_min} and {dia_max}, realization {realization}:")
        print(f"  F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}")
        print(f"  Result: {result}")
        print("\n")


    return pd.DataFrame(rows_anova)


#################################################################################################
# Main script execution starts here

# --- Run analysis in parallel ---
n_cores = max(multiprocessing.cpu_count() - 1, 1)
print(f"Using {n_cores} CPU cores for parallel execution.")

if GENERATE_DATA:
    # generate data for all diameters and realizations, and save combined CSV

    # --- Create output folder ---
    output_folder = get_next_output_folder(base_folder=input_folder)

    # --- Load, validate and plot input data ---
    bnd_gdf, lineaments_gdf, max_admissible_diameter = load_and_validate_data(
        boundary_file=boundary_file, lineaments_file=lineaments_file
    )

    # --- Compute diameter list ---
    diameter_list = generate_diameter_list(
        diameter_min=diameter_min,
        diameter_max=diameter_max,
        n_steps=n_steps,
        spacing_type=spacing_type,
        max_admissible_diameter=max_admissible_diameter,
    )

    # Parallel processing for all diameters, with progress bar
    all_results = Parallel(n_jobs=n_cores, verbose=5)(
        delayed(process_diameter)(
            diameter=d, bnd_gdf=bnd_gdf, lineaments_gdf=lineaments_gdf,
        )
        for d in diameter_list
    )

    # Combine results from all diameters and realizations into a single DataFrame
    alldata_df = pd.concat(all_results, ignore_index=True)

    alddata_csv = os.path.join(
        output_folder, f"circle_p21_alldata_{spacing_type}_spacing.csv"
    )
    alldata_df.to_csv(alddata_csv, index=False)

    # --- Show geoprocessing time ---
    geoprocessing_time = time.time()

    geoprocessing_seconds = geoprocessing_time - analysis_start_time
    hours = int(geoprocessing_seconds // 3600)
    minutes = int((geoprocessing_seconds % 3600) // 60)
    seconds = int(geoprocessing_seconds % 60)
    print(
        f"\n⏱️ Total simulation time: {hours}h {minutes}m {seconds}s ({geoprocessing_seconds:.2f} seconds)"
    )

    print(
        f"\n🏁 All diameters and realizations processed. Combined CSV saved to:\n{alddata_csv}"
    )

else:
    # load previously generated combined CSV
    current_folder = os.getcwd() + "/" + input_folder
    print(f"Current working directory: {current_folder}")
    realization_folder = input("Select the realization folder containing the combined CSV file: ")
    output_folder = current_folder + "/" + realization_folder
    alldata_csv = os.path.join(output_folder, f"circle_p21_alldata_{spacing_type}_spacing.csv")
    if not os.path.exists(alldata_csv):
        print(f"ERROR: Combined CSV file not found: {alldata_csv}")
        exit(1)
    alldata_df = pd.read_csv(alldata_csv)
    print(f"✅ Loaded combined CSV from {alldata_csv} with {len(alldata_df)} rows.")

    diameter_list = sorted(alldata_df["diameter"].unique())
    max_admissible_diameter = max(diameter_list)

    # set analysis time to 0.0 since we are skipping the geoprocessing step
    geoprocessing_time = time.time()
    geoprocessing_seconds = 0.0

# --- Histogram of P21 for each diameter ---
if PLOT_HISTOGRAMS:
    for diameter in sorted(alldata_df["diameter"].unique()):
        subset = alldata_df[alldata_df["diameter"] == diameter]
        plt.figure(figsize=(8, 6))
        sns.histplot(subset["P21"], bins="doane", kde=True, color="royalblue")
        plt.title(f"Histogram of P21 for diameter {diameter:.2f}")
        plt.xlabel("P21")
        plt.ylabel("Count")
        plt.tight_layout()
        hist_png = os.path.join(output_folder, f"histogram_P21_d{diameter:.2f}.png")
        hist_svg = os.path.join(output_folder, f"histogram_P21_d{diameter:.2f}.svg")
        plt.savefig(hist_png, dpi=300)
        plt.savefig(hist_svg, format="svg")
        plt.close()
    print("✅ Histograms of P21 saved for all diameters.")

# --- Plot cumulative box-and-whisker plot of P21 for all diameters and realizations ---
if PLOT_CUM_BOXPLOT:
    fig_2 = plt.figure(figsize=(12, 7))
    sns.boxplot(x="diameter", y="P21", data=alldata_df, color="lightblue")
    plt.title("Box-and-Whisker Plot of P21 by Circle Diameter")
    plt.xlabel("Circle Diameter")
    plt.ylabel("P21")
    plt.xticks(rotation=45)
    plt.tight_layout()
    boxplot_png = os.path.join(output_folder, "boxplot_P21_by_diameter.png")
    boxplot_svg = os.path.join(output_folder, "boxplot_P21_by_diameter.svg")
    fig_2.savefig(boxplot_png, dpi=300)
    fig_2.savefig(boxplot_svg, format="svg")
    plt.close()
    print("✅ Cumulative box-and-whisker plot of P21 by diameter saved as PNG and SVG.")

if PLOT_BOXPLOTS:
    for realization_id in sorted(alldata_df["realization_id"].unique()):
        subset = alldata_df[alldata_df["realization_id"] == realization_id]
        fig_realization = plt.figure(figsize=(12, 7))
        sns.boxplot(x="diameter", y="P21", data=subset, color="lightcoral")
        plt.title(f"Box-and-Whisker Plot of P21 by Diameter for Realization {realization_id}")
        plt.xlabel("Circle Diameter")
        plt.ylabel("P21")
        plt.xticks(rotation=45)
        plt.tight_layout()
        boxplot_realization_png = os.path.join(output_folder, f"boxplot_P21_by_diameter_realization_{realization_id}.png")
        boxplot_realization_svg = os.path.join(output_folder, f"boxplot_P21_by_diameter_realization_{realization_id}.svg")
        fig_realization.savefig(boxplot_realization_png, dpi=300)
        fig_realization.savefig(boxplot_realization_svg, format="svg")
        plt.close()
    print("✅ Box-and-whisker plots of P21 by diameter for each realization saved.")

# --- Run Shapiro residuals test ---
shapiro_residuals_df = check_normality_error_variables(alldata_df=alldata_df)
shapiro_residuals_csv = os.path.join(output_folder, "circle_p21_shapiro_residuals_results.csv")
shapiro_residuals_df.to_csv(shapiro_residuals_csv, index=False)
# plot levene results here

# --- Run Levene test ---
levene_df = run_levene_test(alldata_df=alldata_df)
levene_csv = os.path.join(output_folder, "circle_p21_levene_results.csv")
levene_df.to_csv(levene_csv, index=False)
# plot levene results here
print("Shapiro on residuals completed --------------------------------")

# --- Show analysis time ---
analysis_time = time.time()

# --- Run ANOVA test ---
anova_diameter_min = float(input("anova_diameter_min:"))
anova_diameter_max = float(input("anova_diameter_max:"))
anova_diameter_min = min(diameter_list, key=lambda x: abs(x - anova_diameter_min))
anova_diameter_max = min(diameter_list, key=lambda x: abs(x - anova_diameter_max))
print(f"ANOVA will be run between diameters {anova_diameter_min} and {anova_diameter_max}")

anova_df = run_anova_test(alldata_df=alldata_df, dia_min=anova_diameter_min, dia_max=anova_diameter_max)
anova_csv = os.path.join(output_folder, "circle_p21_anova_results.csv")
anova_df.to_csv(anova_csv, index=False)
# plot levene results here

# --- Show stats and total analysis time ---
analysis_end_time = time.time()

stats_seconds = analysis_end_time - geoprocessing_time
hours = int(stats_seconds // 3600)
minutes = int((stats_seconds % 3600) // 60)
seconds = int(stats_seconds % 60)
print(
    f"\n⏱️ Stats analysis time: {hours}h {minutes}m {seconds}s ({stats_seconds:.2f} seconds)"
)

total_seconds = analysis_end_time - analysis_start_time
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
print(
    f"\n⏱️ Total analysis time: {hours}h {minutes}m {seconds}s ({total_seconds:.2f} seconds)"
)

# --- Save analysis parameters summary CSV ---
params = {
    "date": datetime.today().strftime("%Y-%m-%d"),
    "input_folder": input_folder,
    "output_folder": output_folder,
    "boundary_file": boundary_file,
    "lineaments_file": lineaments_file,
    "n_iterations": n_iterations,
    "n_points_target": n_points_target,
    "diameter_min": diameter_min,
    "diameter_max": diameter_max,
    "n_steps": n_steps,
    "spacing_type": spacing_type,
    "diameter_list": list(np.round(diameter_list, 2)),
    "n_cores": n_cores,
    "max_admissible_diameter": max_admissible_diameter,
    "geoprocessing_time_seconds": round(geoprocessing_seconds, 2),
    "stats_time_seconds": round(stats_seconds, 2),
    "total_time_seconds": round(total_seconds, 2),
}
params_df = pd.DataFrame([params])
params_csv = os.path.join(output_folder, "circle_p21_analysis_summary.csv")
params_df.to_csv(params_csv, index=False)
print(f"✅ Analysis parameters summary saved to: {params_csv}")

# ==========================================
# 1. GLOBAL CONFIGURATION & SETUP
# ==========================================
# BASE_DATA_DIR = 'data_pontrelli'
SUB_FOLDER = output_folder  # Update this to '002', '003', etc.
# DATA_PATH = os.path.join(BASE_DATA_DIR, SUB_FOLDER)14

DATA_PATH = os.path.join(SUB_FOLDER)

# File Paths
FILE_ALL_DATA = os.path.join(DATA_PATH, 'circle_p21_alldata_linear_spacing.csv')
FILE_SHAPIRO  = os.path.join(DATA_PATH, 'circle_p21_shapiro_residuals_results.csv')
FILE_LEVENE   = os.path.join(DATA_PATH, 'circle_p21_levene_results.csv')
FILE_ANOVA    = os.path.join(DATA_PATH, 'circle_p21_anova_results.csv')
FILE_SUMMARY  = os.path.join(DATA_PATH, 'circle_p21_analysis_summary.csv')

# Parameters
ITERATION_TO_PLOT = 30
SIGNIFICANCE_LEVEL = 0.05
SKIP_FIRST_RADIUS = True # Set to True to skip the first radius in the plots (if it has very different scale or too few samples)

# Output Root
OUTPUT_ROOT = f'Integrated_Results_{SUB_FOLDER}'
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Styles
COLOR_NORMAL = 'orange'
COLOR_NOT_NORMAL = 'gray'

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def create_subfolder(folder_name):
    path = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def annotate_boxplot(ax, box, counts, positions):
    for x, n, whisker, flier in zip(positions, counts, box['whiskers'][1::2], box['fliers']):
        top_whisker = whisker.get_ydata()[1]
        outliers = flier.get_ydata()
        y = outliers.max() if len(outliers) > 0 else top_whisker
        ax.text(x, y + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f'n={n}',
                ha='center', va='bottom', fontsize=10, color='black')

# ==========================================
# 3. EDA: BOXPLOTS, MEAN, VARIANCE
# ==========================================
df_main = pd.read_csv(FILE_ALL_DATA)
df_iter = df_main[df_main['realization_id'] == ITERATION_TO_PLOT]
unique_diameters = sorted(df_iter['diameter'].unique())

if SKIP_FIRST_RADIUS:
    unique_diameters.pop(0)

eda_folder = create_subfolder('01_EDA_Plots')

# Logic for Split vs Single Boxplot
if len(unique_diameters) <= 30:
    plt.figure(figsize=(16, 8))
    data = [df_iter[df_iter['diameter'] == r]['P21'].values for r in unique_diameters]
    box = plt.boxplot(data, positions=unique_diameters, widths=0.6, patch_artist=True)
    #annotate_boxplot(plt.gca(), box, [len(d) for d in data], unique_diameters)
    plt.title(f'Box Plot of P21 by Diameter (Iteration {ITERATION_TO_PLOT}, Scan areas n = {n_points_target})', fontsize=16)
    plt.xticks(unique_diameters, [f"{r:.3f}" for r in unique_diameters], rotation=90)
    plt.savefig(os.path.join(eda_folder, 'boxplot_standard.png'), dpi=300)
    plt.close()
# (Chunked logic omitted for brevity but remains the same as previous)

# Variance of p21 for each radius

# Get sorted unique radii
unique_diameter = sorted(df_iter['diameter'].unique())

# Compute variance of p21 for each radius
variances = [df_iter[df_iter['diameter'] == r]['P21'].var(ddof=0) for r in unique_diameter]

plt.figure(figsize=(12, 6))
plt.plot(unique_diameter, variances, marker='o', linestyle='-', color='blue')
plt.xlabel('Scan area diameter [m]', fontsize=14)
plt.ylabel('P21 variance', fontsize=14)
#plt.ylim([0, 10])
#plt.title(f'Variance of p21 for each scan area radius (Iteration {iteration_to_plot})', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, f'variance_p21_iteration_{ITERATION_TO_PLOT}.png'), dpi=300)
plt.close()

# Mean of p21 for each radius

# Get sorted unique radii
unique_diameter = sorted(df_iter['diameter'].unique())

# Compute mean of p21 for each radius
means = [df_iter[df_iter['diameter'] == r]['P21'].mean() for r in unique_diameter]

plt.figure(figsize=(12, 6))
plt.plot(unique_diameter, means, marker='o', linestyle='-', color='orange')
plt.xlabel('Scan area diameter [m]', fontsize=14)
plt.ylabel('P21 mean [m^-1]', fontsize=14)
#plt.title(f'Mean of p21 for each scan area radius (Iteration {iteration_to_plot})', fontsize=16)
plt.grid(True)

output_folder = 'mean_p21_by_diameter_Bristol_100_scan_area_0.005_18_NEW'
os.makedirs(output_folder, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, f'mean_p21_iteration_{ITERATION_TO_PLOT}.png'), dpi=300)
plt.close()


# ==========================================
# 4. SHAPIRO-WILK (Stacked Bar)
# ==========================================
shapiro_folder = create_subfolder('02_Shapiro_Results')
df_s = pd.read_csv(FILE_SHAPIRO)
df_s['res_bool'] = np.where(df_s['p_value'] < SIGNIFICANCE_LEVEL, 0, 1)
pivot_s = df_s.groupby(['dia', 'res_bool']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
pivot_s[[1, 0]].plot(kind='bar', stacked=True, color=[COLOR_NORMAL, COLOR_NOT_NORMAL], ax=ax)
ax.set_xticklabels([f"{l:.3f}" for l in pivot_s.index], rotation=90)
plt.savefig(os.path.join(shapiro_folder, 'shapiro_stacked.png'))
plt.close()

# ==========================================
# 5. LEVENE TEST (Fix: using dia_min/dia_max)
# ==========================================
levene_folder = create_subfolder('03_Levene_Results')
df_l = pd.read_csv(FILE_LEVENE)
df_l['res_bool'] = np.where(df_l['p_value'] < SIGNIFICANCE_LEVEL, 0, 1)

# Create the tuple from dia_min and dia_max and round them
df_l['Interval_tuple'] = df_l.apply(lambda row: (round(row['dia_min'], 3), round(row['dia_max'], 3)), axis=1)

grouped_l = df_l.groupby(['y', 'Interval_tuple', 'res_bool']).size().unstack(fill_value=0)

for y_val in sorted(df_l['y'].unique()):
    data = grouped_l.loc[y_val].sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    data[[1, 0]].plot(kind='bar', stacked=True, ax=ax, color=[COLOR_NORMAL, COLOR_NOT_NORMAL])
    max_h = data.sum(axis=1).max()
    ax.axhline(0.8 * max_h, color='yellow', lw=3, ls='--', label='80% Threshold')
    ax.set_ylabel('Count')
    ax.set_title(f'Levene Results - Group Size {y_val}')
    plt.tight_layout()
    plt.savefig(os.path.join(levene_folder, f'levene_group_{y_val}.png'))
    plt.close()

# ==========================================
# 6. ANOVA RESULTS SUMMARY
# ==========================================
anova_folder = create_subfolder('04_ANOVA_Results')
df_a = pd.read_csv(FILE_ANOVA)

# Define significance (assumes SIGNIFICANCE_LEVEL is 0.05 from your global config)
df_a['is_accepted'] = df_a['p_value'] >= SIGNIFICANCE_LEVEL

# Group by the diameter range to calculate summary statistics
summary_rows = []
for (d_min, d_max), group_df in df_a.groupby(['dia_min', 'dia_max']):
    acceptances = group_df['is_accepted'].sum()
    rejections = len(group_df) - acceptances
    total = len(group_df)

    # Threshold logic (80% of total)
    test_result = "accepted" if (acceptances / total) > 0.8 else "rejected"

    summary_rows.append({
        "Analyzed group": f"{d_min}m to {d_max}m group",
        "ANOVA acceptances": acceptances,
        "ANOVA rejections": rejections,
        "Test result": test_result
    })

# Create the summary DataFrame
df_anova_summary = pd.DataFrame(summary_rows)

# Save as a new CSV file
anova_summary_csv = os.path.join(anova_folder, 'anova_summary_results.csv')
df_anova_summary.to_csv(anova_summary_csv, index=False)

print(f"ANOVA Summary saved to: {anova_summary_csv}")

# ==========================================
# 7. PROBABILITY PLOTS
# ==========================================
prob_folder = create_subfolder('05_ProbPlots')
df_sum = pd.read_csv(FILE_ALL_DATA)
df_sum_iter = df_sum[df_sum['realization_id'] == ITERATION_TO_PLOT]

# Get all unique diameters (sorted for convenience)
unique_diameters = sorted(df_sum_iter['diameter'].unique())

for r in unique_diameters:
    # Probability plot for residuals
    residuals_values = df_sum_iter[df_sum_iter['diameter'] == r]['residuals']
    plt.figure(figsize=(12, 12))
    stats.probplot(residuals_values, dist="norm", plot=plt)
    plt.title(f'Normal Probability Plot for residuals (diameter={r})\nInteraction {ITERATION_TO_PLOT}', fontsize=20)
    plt.xlabel('Theoretical Quantiles', fontsize=18)
    plt.ylabel('Ordered Values', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(prob_folder, f'prob_r_{r}.png'))
    plt.close()

# ==========================================
# 7. RESIDUAL PLOTS
# ==========================================

res_folder = create_subfolder('06_Residuals_Plots')

df_res = pd.read_csv(FILE_ALL_DATA)
df_res_iter = df_sum[df_sum['realization_id'] == ITERATION_TO_PLOT]

diameters = sorted(df_res_iter["diameter"].unique())
min_group_size = 3 #set them depending on your needs
max_group_size = 20

for group_size in range(min_group_size, max_group_size + 1):
    for i in range(len(diameters) - group_size + 1):
        diameters_group = diameters[i:i + group_size]
        df_group = df_res_iter[df_res_iter["diameter"].isin(diameters_group)].copy()

        mean_res = df_group["residuals"].mean()
        std_res = df_group["residuals"].std(ddof=0)
        df_group["std_residuals"] = (df_group["residuals"] - mean_res) / std_res

        stds = []
        for radius in diameters_group:
            std_local = df_group[df_group["diameter"] == radius]["std_residuals"].std(ddof=0)
            stds.append(std_local)

        stds = np.array(stds)
        upper = 1.96 * stds
        lower = -1.96 * stds

        fig, ax = plt.subplots(figsize=(8, 5))

        # Highlight area between -1.96*std and +1.96*std for each radius
        ax.fill_between(diameters_group, lower, upper, color='orange', alpha=0.3, label='±1.96 local std region')
        # Thicken the borders
        ax.plot(diameters_group, upper, color='black', linewidth=2)
        ax.plot(diameters_group, lower, color='black', linewidth=2)

        for diameter in diameters_group:
            subset = df_group[df_group["diameter"] == diameter]
            y = subset["std_residuals"]
            inside = abs(y) <= 1.96
            ax.scatter(
                [diameter] * sum(inside), y[inside],
                facecolors='none', edgecolors='blue', label='Within ±1.96' if diameter == diameters_group[0] else ""
            )
            outside = abs(y) > 1.96
            ax.scatter(
                [diameter] * sum(outside), y[outside],
                color='red', edgecolors='red', label='Outside ±1.96' if diameter == diameters_group[0] else ""
            )

        ax.set_xlabel("Scan area diameter [m]")
        ax.set_ylabel("Standardized Residuals")
        #ax.set_title(f"Standardized Residual plot: radius {radius_group[0]} to {radius_group[-1]} (group size {group_size})")
        ax.legend()
        ax.grid(True)

        filename = f"group{group_size}_std_residual_iter{ITERATION_TO_PLOT}_rad{diameters_group[0]}to{diameters_group[-1]}.png"
        fig.savefig(os.path.join(res_folder, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f"Workflow Complete for {SUB_FOLDER}.")

print(f"Finished processing folder: {SUB_FOLDER}")
print(f"Results saved in: {os.path.abspath(OUTPUT_ROOT)}")