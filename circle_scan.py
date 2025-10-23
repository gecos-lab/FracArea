"""circle_scan.py
Multi-Radius Monte Carlo Circular Scan-Area Sampling
FracArea© Andrea Bistacchi
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
import os
from joblib import Parallel, delayed
import multiprocessing
import re
import seaborn as sns
import time
from datetime import datetime

# --- Start timing ---
analysis_start_time = time.time()

# --- Runtime options ---
PLOT_FIGURES = True  # Set True to generate per-radius maps
PLOT_HISTOGRAMS = True  # Set True to generate per-radius histograms
PLOT_BOXPLOTS = True  # Set True to generate summary box/line plots

# --- Base analysis parameters TEST ---
n_iterations = 100  # Monte Carlo iterations per radius
n_points_target = 100  # Circles per iteration
radius_min = 5000  # Minimum circle radius
radius_max = 60000  # Maximum circle radius
n_steps = 12  # Number of radii
spacing_type = "linear"  # Options: "linear", "exponential", "log"

# --- Locate input folder and files TEST ---
input_folder = "data_test"
boundary_file = os.path.join(input_folder, "boundary.shp")
lineaments_file = os.path.join(input_folder, "lineaments.shp")

# # --- Base analysis parameters PONTRELLI ---
# n_iterations = 100          # Monte Carlo iterations per radius
# n_points_target = 100       # Circles per iteration
# radius_min = 1              # Minimum circle radius
# radius_max = 26             # Maximum circle radius
# n_steps = 26                 # Number of radii
# spacing_type = "linear"  # Options: "linear", "exponential", "log"
#
# # --- Locate input folder and files PONTRELLI ---
# input_folder = "data_pontrelli"
# boundary_file = os.path.join(input_folder, "Interpretation-boundary.shp")
# lineaments_file = os.path.join(input_folder, "FN_set_1.shp")


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


def run_iteration(
    iter_id=None,
    inner_poly=None,
    lineaments_gdf=None,
    n_points_target=None,
    circle_radius=None,
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
    :param circle_radius: The radius of the circles to be created around sampled points.
    :type circle_radius: float
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
        circles_gdf["circle_id"] = []
        circles_gdf["iteration_id"] = []
        circles_gdf["clipped_length"] = []
        circles_gdf["radius"] = []
        circles_gdf["area"] = []
        circles_gdf["length_per_area"] = []
        circles_gdf["x"] = []
        circles_gdf["y"] = []
        # circles_gdf["inside_outer"] = []
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

    # Circles
    circles_gdf = gpd.GeoDataFrame(
        geometry=points_sample.geometry.buffer(circle_radius), crs=crs
    ).reset_index(drop=True)
    circles_gdf["circle_id"] = circles_gdf.index
    circles_gdf["iteration_id"] = iter_id

    # Spatial join of circles with lineaments
    joined = gpd.sjoin(
        lineaments_gdf, circles_gdf[["circle_id", "geometry"]], predicate="intersects"
    )

    # If no lineaments intersect any circles, return circles_gdf with zero clipped_length
    if joined.empty:
        circles_gdf["clipped_length"] = 0
        circles_gdf["radius"] = circle_radius
        circles_gdf["area"] = circles_gdf.geometry.area
        circles_gdf["length_per_area"] = 0
        circles_gdf["x"] = points_sample.geometry.x
        circles_gdf["y"] = points_sample.geometry.y
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
    circles_gdf["clipped_length"] = circles_gdf["clipped_length"].fillna(0)
    circles_gdf["radius"] = circle_radius
    circles_gdf["area"] = circles_gdf.geometry.area
    circles_gdf["length_per_area"] = circles_gdf["clipped_length"] / circles_gdf["area"]
    circles_gdf["x"] = points_sample.geometry.x
    circles_gdf["y"] = points_sample.geometry.y
    return circles_gdf


def process_radius(radius=None, bnd_gdf=None, lineaments_gdf=None):
    """
    Processes the given radius by performing several operations including
    polygon buffering, data concatenation, and file saving. Iterations are
    conducted to compute results, stored in a geodataframe, which gets saved
    to a CSV file. Optionally, plots can be generated for visualization if
    enabled by a global flag.

    :param bnd_gdf:
    :param lineaments_gdf:
    :param radius: The radius value (in meters) to process.
    :type radius: float
    :return: Geodataframe containing the results for the given radius.
    :rtype: geopandas.GeoDataFrame
    """
    print(f"\n▶️ Radius = {radius:.2f}")
    t0 = time.perf_counter()
    buffer_distance = -radius

    # Inner polygon for this radius (do not mutate bnd_gdf)
    inner_poly = bnd_gdf.geometry.buffer(buffer_distance).union_all()

    # Run all iterations (sequential within this radius)
    iter_t0 = time.perf_counter()
    results_list = []
    for i in range(1, n_iterations + 1):
        result = run_iteration(
            iter_id=i,
            inner_poly=inner_poly,
            lineaments_gdf=lineaments_gdf,
            n_points_target=n_points_target,
            circle_radius=radius,
            crs=bnd_gdf.crs,
        )
        result["radius_tested"] = radius
        results_list.append(result)
        last_iteration_result = result
    print(f"  Iterations done in {time.perf_counter()-iter_t0:.3f}s")

    # Combine iterations for this radius
    concat_t0 = time.perf_counter()
    results_gdf = pd.concat(results_list, ignore_index=True)
    print(f"  Concatenation in {time.perf_counter()-concat_t0:.3f}s")

    # Save CSV
    csv_t0 = time.perf_counter()
    csv_path = os.path.join(output_folder, f"circle_density_r{radius:.2f}.csv")
    results_gdf.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"  CSV written in {time.perf_counter()-csv_t0:.3f}s")

    if PLOT_FIGURES:
        # Plot colored by length_per_area
        fig, ax = plt.subplots(figsize=(8, 8))
        bnd_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            label="Outer Polygon",
        )
        gpd.GeoSeries(inner_poly).plot(
            ax=ax,
            facecolor="none",
            edgecolor="red",
            linestyle="--",
            label="Inner Polygon",
        )
        lineaments_gdf.plot(ax=ax, color="gray", linewidth=0.5, label="Lineaments")
        last_iteration_result.plot(
            ax=ax,
            column="length_per_area",
            cmap="viridis",
            legend=True,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.3,
        )
        # Custom legend handles
        legend_handles = [
            mpl_line([0], [0], color="black", lw=1.2, label="Outer Polygon"),
            mpl_line(
                [0], [0], color="red", lw=1.2, linestyle="--", label="Inner Polygon"
            ),
            mpl_line([0], [0], color="gray", lw=0.5, label="Lineaments"),
            mpl_patch(
                edgecolor="black",
                facecolor="none",
                linewidth=0.3,
                label="Circle (length_per_area)",
            ),
        ]
        plt.title(f"Radius = {radius:.2f} m — Last Iteration (Colored by Density)")
        plt.legend(handles=legend_handles)
        plt.axis("equal")
        plt.tight_layout()

        # Save figures
        fig_path_png = os.path.join(output_folder, f"figure_r{radius:.2f}.png")
        fig_path_svg = os.path.join(output_folder, f"figure_r{radius:.2f}.svg")
        fig.savefig(fig_path_png, dpi=300)
        fig.savefig(fig_path_svg, format="svg")
        plt.close()
        print(f"  Figures saved in radius loop")

    print(f"✅ Saved CSV for radius {radius:.2f} (total {time.perf_counter()-t0:.3f}s)")

    return results_gdf


def generate_radius_list(
    radius_min=None, radius_max=None, n_steps=None, spacing_type=None
):
    if spacing_type == "linear":
        radius_list = np.linspace(radius_min, radius_max, n_steps)
    elif spacing_type == "exponential":
        radius_list = np.geomspace(radius_min, radius_max, n_steps)
    elif spacing_type == "log":
        radius_list = np.logspace(np.log10(radius_min), np.log10(radius_max), n_steps)
    else:
        raise ValueError(
            "Invalid spacing_type: choose 'linear', 'exponential', or 'log'"
        )

    print(f"Radii to process ({spacing_type} spacing): {np.round(radius_list, 2)}")

    return radius_list


#################################################################################################
# Main script execution starts here

# --- Create output folder ---
output_folder = get_next_output_folder(base_folder=input_folder)

# --- Compute radius list ---
radius_list = generate_radius_list(
    radius_min=radius_min,
    radius_max=radius_max,
    n_steps=n_steps,
    spacing_type=spacing_type,
)

# --- Load Data ---
data_load_t0 = time.perf_counter()
if not os.path.exists(boundary_file):
    print(f"ERROR: Boundary file not found: {boundary_file}")
    exit(1)
if not os.path.exists(lineaments_file):
    print(f"ERROR: Lineaments file not found: {lineaments_file}")
    exit(1)
bnd_gdf = gpd.read_file(boundary_file)
lineaments_gdf = gpd.read_file(lineaments_file).to_crs(bnd_gdf.crs)
print(
    f"✅ Loaded data: {len(bnd_gdf)} polygons, {len(lineaments_gdf)} lineaments in {time.perf_counter()-data_load_t0:.3f}s"
)

# --- Compute maximum admissible circle radius for the boundary polygon ---
boundary_poly = bnd_gdf.geometry.iloc[0]
max_radius = shapely.maximum_inscribed_circle(boundary_poly)
center_coords = list(max_radius.coords)[0]
center_pt = shapely.geometry.Point(center_coords)
radius_coords = list(max_radius.coords)[1]
radius_pt = shapely.geometry.Point(radius_coords)
max_admissible_radius = max_radius.length
print(
    f"Center point of maximum inscribed circle: ({center_pt.x:.2f}, {center_pt.y:.2f})"
)
print(
    f"Radius point of maximum inscribed circle: ({radius_pt.x:.2f}, {radius_pt.y:.2f})"
)
print(f"Maximum admissible circle radius: {max_admissible_radius:.2f}")

# --- Plot boundary, max circle, center, and first point ---
if PLOT_FIGURES:
    fig_1, ax1 = plt.subplots(figsize=(8, 8))
    bnd_gdf.plot(ax=ax1, facecolor="none", edgecolor="black", linewidth=1.5)
    circle_patch = mpl_circle(
        (center_pt.x, center_pt.y),
        max_admissible_radius,
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
    fig_path_png = os.path.join(output_folder, "boundary_max_radius_center.png")
    fig_path_svg = os.path.join(output_folder, "boundary_max_radius_center.svg")
    fig_1.savefig(fig_path_png, dpi=300)
    fig_1.savefig(fig_path_svg, format="svg")
    print(f"✅ Figure saved: {fig_path_png}, {fig_path_svg}")


# --- Run analysis in parallel ---
n_cores = max(multiprocessing.cpu_count() - 1, 1)
print(f"Using {n_cores} CPU cores for parallel execution.")

all_results = Parallel(n_jobs=n_cores, verbose=5)(
    delayed(process_radius)(radius=r, bnd_gdf=bnd_gdf, lineaments_gdf=lineaments_gdf)
    for r in radius_list
)
master_df = pd.concat(all_results, ignore_index=True)
master_csv = os.path.join(
    output_folder, f"circle_density_ALL_{spacing_type}_spacing.csv"
)
master_df.drop(columns="geometry").to_csv(master_csv, index=False)
print(f"\n🏁 All radii processed. Combined CSV saved to:\n{master_csv}")

# --- Histogram of length_per_area for each radius ---
if PLOT_HISTOGRAMS:
    for radius in sorted(master_df["radius_tested"].unique()):
        subset = master_df[master_df["radius_tested"] == radius]
        plt.figure(figsize=(8, 6))
        sns.histplot(
            subset["length_per_area"], bins="doane", kde=True, color="royalblue"
        )
        plt.title(f"Histogram of length_per_area for radius {radius:.2f}")
        plt.xlabel("length_per_area")
        plt.ylabel("Count")
        plt.tight_layout()
        hist_png = os.path.join(
            output_folder, f"histogram_length_per_area_r{radius:.2f}.png"
        )
        hist_svg = os.path.join(
            output_folder, f"histogram_length_per_area_r{radius:.2f}.svg"
        )
        plt.savefig(hist_png, dpi=300)
        plt.savefig(hist_svg, format="svg")
        plt.close()
    print("✅ Histograms of length_per_area saved for all radii.")

# --- Box-and-whisker plot of length_per_area for all radii ---
if PLOT_BOXPLOTS:
    fig_2 = plt.figure(figsize=(12, 7))
    sns.boxplot(
        x="radius_tested", y="length_per_area", data=master_df, color="lightblue"
    )
    plt.title("Box-and-Whisker Plot of length_per_area by Circle Radius")
    plt.xlabel("Circle Radius")
    plt.ylabel("length_per_area")
    plt.xticks(rotation=45)
    plt.tight_layout()
    boxplot_png = os.path.join(output_folder, "boxplot_length_per_area_by_radius.png")
    boxplot_svg = os.path.join(output_folder, "boxplot_length_per_area_by_radius.svg")
    fig_2.savefig(boxplot_png, dpi=300)
    fig_2.savefig(boxplot_svg, format="svg")
    plt.close()
    print("✅ Box-and-whisker plot of length_per_area by radius saved as PNG and SVG.")

# --- Stop timer ---
analysis_end_time = time.time()
elapsed_seconds = analysis_end_time - analysis_start_time
# Print total time in hours, minutes, seconds (and seconds in parentheses)
hours = int(elapsed_seconds // 3600)
minutes = int((elapsed_seconds % 3600) // 60)
seconds = int(elapsed_seconds % 60)
print(
    f"\n⏱️ Total analysis time: {hours}h {minutes}m {seconds}s ({elapsed_seconds:.2f} seconds)"
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
    "radius_min": radius_min,
    "radius_max": radius_max,
    "n_steps": n_steps,
    "spacing_type": spacing_type,
    "radius_list": list(np.round(radius_list, 2)),
    "n_cores": n_cores,
    "max_admissible_radius": max_admissible_radius,
    "analysis_time_seconds": round(elapsed_seconds, 2),
}
params_df = pd.DataFrame([params])
params_csv = os.path.join(output_folder, "analysis_parameters_summary.csv")
params_df.to_csv(params_csv, index=False)
print(f"✅ Analysis parameters summary saved to: {params_csv}")
