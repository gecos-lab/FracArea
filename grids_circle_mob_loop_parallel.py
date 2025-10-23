from functools import total_ordering

import pyvista as pv
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, MultiLineString
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import levene, f_oneway, shapiro
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import concurrent.futures

alpha = 0.05  # Significance level for statistical tests

def compute_p21_statistics(fracs, boundary, rads, n_points, plot=True):
    """
    Compute P21 fracture intensity statistics by randomly placing circular sampling windows within a boundary.

    Parameters:
        fracs (GeoDataFrame): Fracture geometries.
        boundary (GeoSeries): Boundary polygons to sample within.
        rads (iterable): List or array of radii to use for circular sampling windows.
        n_points (int): Number of random centers to sample per radius per boundary.
        plot (bool): If True, show a PyVista plot of the sampling discs.

    Returns:
        GeoDataFrame: A GeoDataFrame with circle centers and associated statistics (radius, area, total fracture length, p21).
    """
    circles = []
    areas = []
    radius = []
    total_lengths = []
    centers = []
    p21 = []

    plotter = pv.Plotter() if plot else None

    for b in boundary:
        minx, miny, maxx, maxy = b.bounds

        for r in rads:
            appender = vtkAppendPolyData()
            c = 0
            attempt = 0
            while c < n_points:
                center = (np.random.uniform(minx, maxx), np.random.uniform(miny, maxy), 0)
                pnt = Point(center)
                pnt_buff = pnt.buffer(r, resolution=360)

                attempt += 1
                print(f"Radius: {r}, Attempt: {attempt}, Accepted: {c}")

                if b.covers(pnt_buff):
                    c += 1
                    circle = pv.Disc(center=center, inner=0, outer=r, c_res=360)
                    circle.cell_data['rad'] = [r] * circle.GetNumberOfCells()
                    appender.AddInputData(circle)
                    appender.Update()

                    clipped = gpd.clip(fracs, pnt_buff)
                    tot_length = clipped.geometry.length.sum()

                    circles.append(pnt_buff)
                    total_lengths.append(tot_length)
                    areas.append(pnt_buff.area)
                    centers.append(pnt)
                    radius.append(r)
                    p21.append(tot_length / pnt_buff.area)

            if plot:
                plotter.add_mesh(appender.GetOutput(), opacity=0.2)

    if plot:
        plotter.show()

    result_gdf = gpd.GeoDataFrame(geometry=centers, crs=fracs.crs)
    result_gdf['radius'] = radius
    result_gdf['areas'] = areas
    result_gdf['total_length'] = total_lengths
    result_gdf['p21'] = p21
    result_gdf['center'] = centers

    # calculate P21 residuals

    result_gdf['residuals'] = result_gdf['p21'] - result_gdf.groupby('radius')['p21'].transform('mean')

    return result_gdf

def run_levene_test(areas_gpd, radii_column='radius', p21_column='p21', alpha=alpha):
    """
    Perform Levene's test for equality of variances between consecutive radius groups.

    Parameters:
        areas_gdf (GeoDataFrame): The dataframe containing P21 values and radii.
        radii_column (str): The name of the column containing radius values.
        p21_column (str): The name of the column containing P21 values.
        alpha (float): Significance level for the test.

    Returns:
        DataFrame: A DataFrame with radius pairs, test statistic, p-values, and result interpretation.
    """
    rows_levene = []

    # Sort the unique radius values
    radii_sorted = sorted(areas_gpd['radius'].unique())

    for i in range(len(radii_sorted) - 1):
        r1 = radii_sorted[i]
        r2 = radii_sorted[i + 1]

        group1 = areas_gpd[areas_gpd[radii_column] == r1][p21_column]
        group2 = areas_gpd[areas_gpd[radii_column] == r2][p21_column]

        stat_levene, p_levene = levene(group1, group2, center='median')

        result = "Different variances" if p_levene < alpha else "No significant difference"

        rows_levene.append({
            'r1': r1,
            'r2': r2,
            'statistic': stat_levene,
            'p_value': p_levene,
            'result': result
        })

        print(f"Levene test between radius {r1} and {r2}:")
        print(f"  Statistic = {stat_levene:.4f}, p-value = {p_levene:.4f}")
        print(f"  Result: {result}")
        print("-" * 50)

    return pd.DataFrame(rows_levene)

def run_shapiro_test(areas_gpd, radii_column='radius', p21_column='p21', alpha=alpha):
    """
    Perform Shapiro-Wilk test for normality on p21 values for each radius group.

    Parameters:
        areas_gdf (GeoDataFrame): The dataframe containing P21 values and radii.
        radii_column (str): Column name for radii.
        p21_column (str): Column name for P21 values.
        alpha (float): Significance level (default 0.1 as per Mooi et al.).

    Returns:
        DataFrame: A DataFrame with radius, test statistic, p-values, and result interpretation.
    """
    rows_shapiro = []

    # Sort the unique radius values
    radii_sorted = sorted(areas_gpd['radius'].unique())

    for r in radii_sorted:
        group = areas_gpd[areas_gpd[radii_column] == r][p21_column]

        if len(group) < 3:
            # Shapiro-Wilk requires at least 3 observations
            rows_shapiro.append({
                'radius': r,
                'statistic': None,
                'p_value': None,
                'result': 'Too few samples'
            })
            print(f"Radius {r} skipped (too few samples for Shapiro-Wilk).")
            continue

        stat_shapiro, p_shapiro = shapiro(group)
        result = "Not normal" if p_shapiro < alpha else "Normal"

        rows_shapiro.append({
            'radius': r,
            'statistic': stat_shapiro,
            'p_value': p_shapiro,
            'result': result
        })

        print(f"Normality test on radius {r}:")
        print(f"  Statistic = {stat_shapiro:.4f}, p-value = {p_shapiro:.4f}")
        print(f"  Result: {result}")
        print("-" * 50)

    return pd.DataFrame(rows_shapiro)

def check_normality_error_variables(areas_gpd, radii_column='radius', residuals_column='residuals', alpha=alpha):
    """
    Perform Shapiro-Wilk test for normality on residuals for each radius group.

    Parameters:
        areas_gpd (GeoDataFrame): DataFrame containing residuals and radii.
        radii_column (str): Column name for radii.
        residuals_column (str): Column name for residuals.
        alpha (float): Significance level.

    Returns:
        DataFrame: Results with radius, statistic, p-value, and interpretation.
    """
    rows_residuals = []
    radii_sorted = sorted(areas_gpd[radii_column].unique())

    for r in radii_sorted:
        group = areas_gpd[areas_gpd[radii_column] == r][residuals_column]

        if len(group) < 3:
            rows_residuals.append({
                'radius': r,
                'statistic': None,
                'p_value': None,
                'result': 'Too few samples'
            })
            print(f"Radius {r} skipped (too few samples for Shapiro-Wilk).")
            continue

        stat_residuals, p_val_residuals = shapiro(group)
        result = "Not normal" if p_val_residuals < alpha else "Normal"

        rows_residuals.append({
            'radius': r,
            'statistic': stat_residuals,
            'p_value': p_val_residuals,
            'result': result
        })

        print(f"Normality test on residuals for radius {r}:")
        print(f"  Statistic = {stat_residuals:.4f}, p-value = {p_val_residuals:.4f}")
        print(f"  Result: {result}")
        print("-" * 50)

    return pd.DataFrame(rows_residuals)

def run_anova_test(areas_gpd, radius_range, radii_column='radius', p21_column='p21', alpha=alpha):
    """
    Perform one-way ANOVA to test if mean P21 values differ across radius groups.

    Parameters:
        areas_gpd (GeoDataFrame): Input dataframe with P21 values and radius.
        radius_range (tuple): (min_radius, max_radius) to filter the REV range.
        radii_column (str): Name of the radius column.
        p21_column (str): Name of the P21 value column.
        alpha (float): Significance threshold (default 0.05).

    Returns:
        DataFrame: A one-row DataFrame with F-statistic, p-value, result label, and tested range.
    """
    min_radius, max_radius = radius_range
    filtered = areas_gpd[
        (areas_gpd[radii_column] >= min_radius) &
        (areas_gpd[radii_column] <= max_radius)
    ]

    grouped = [group[p21_column].values for _, group in filtered.groupby(radii_column)]

    # Check for sufficient data
    if any(len(g) < 2 for g in grouped) or len(grouped) < 2:
        return pd.DataFrame([{
            'min_radius': min_radius,
            'max_radius': max_radius,
            'f_statistic': None,
            'p_value': None,
            'result': 'Too few groups or samples'
        }])

    f_stat, p_val = f_oneway(*grouped)
    result = "At least one group mean is different" if p_val < alpha else "No significant difference"

    print(f"ANOVA test on radii between {min_radius} and {max_radius}:")
    print(f"  F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}")
    print(f"  Result: {result}")
    print("-" * 50)

    return pd.DataFrame([{
        'min_radius': min_radius,
        'max_radius': max_radius,
        'f_statistic': f_stat,
        'p_value': p_val,
        'result': result
    }])

def full_statistical_analysis(fracs, boundary, rads, n_points, radius_range=(700, 1300), alpha=alpha, plot=False):
    """
    Execute full statistical analysis on fracture data: P21 computation, Levene, Shapiro, ANOVA, and residuals normality.

    Parameters:
        fracs (GeoDataFrame): Fracture geometries.
        boundary (GeoSeries): Interpretation boundary.
        rads (array-like): Radii to sample.
        n_points (int): Number of sampling points per radius.
        radius_range (tuple): REV range (min_radius, max_radius) for ANOVA.
        alpha (float): Significance threshold for statistical tests.
        plot (bool): Whether to show PyVista plot of discs.

    Returns:
        areas_gpd (GeoDataFrame): P21 statistics per point
        levene_df (DataFrame): Levene test results
        shapiro_df (DataFrame): Shapiro-Wilk test results
        anova_df (DataFrame): ANOVA test result (1 row)
        residuals_shapiro_df (DataFrame): Shapiro-Wilk test results on residuals
    """
    print("Step 1: Computing P21 statistics...")
    areas_gpd = compute_p21_statistics(fracs, boundary, rads, n_points, plot=plot)

    print("Step 2: Running Levene test...")
    levene_df = run_levene_test(areas_gpd=areas_gpd, alpha=alpha)

    print("Step 3: Running Shapiro-Wilk test...")
    shapiro_df = run_shapiro_test(areas_gpd=areas_gpd, alpha=alpha)

    print("Step 4: Running one-way ANOVA...")
    anova_df = run_anova_test(areas_gpd=areas_gpd, radius_range=radius_range, alpha=alpha)

    print("Step 5: Running Shapiro-Wilk test on residuals...")
    residuals_shapiro_df = check_normality_error_variables(areas_gpd=areas_gpd, alpha=alpha)

    return areas_gpd, levene_df, shapiro_df, anova_df, residuals_shapiro_df

def run_one_analysis(i, fracs, boundary, rads, n_points, radius_range, alpha):
    """
    Single run of full_statistical_analysis for parallel execution.
    Returns: (shapiro_df, levene_df, anova_df, residuals_shapiro_df) with 'iteration' column added.
    """
    areas_gpd, levene_df, shapiro_df, anova_df, residuals_shapiro_df = full_statistical_analysis(
        fracs=fracs,
        boundary=boundary,
        rads=rads,
        n_points=n_points,
        radius_range=radius_range,
        alpha=alpha,
        plot=False
    )

    # Add iteration column to each result
    levene_df['iteration'] = i
    shapiro_df['iteration'] = i
    anova_df['iteration'] = i
    residuals_shapiro_df['iteration'] = i
    areas_gpd['iteration'] = i

    return shapiro_df, levene_df, anova_df, residuals_shapiro_df, areas_gpd

def parallel_statistical_analysis(fracs, boundary):
    all_shapiro = []
    all_levene = []
    all_anova = []
    all_residuals_shapiro = []
    all_areas_gpd = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_one_analysis, i, fracs, boundary, rads, n_points, radius_range, alpha)
            for i in range(n_iterations)
        ]

        for future in futures:
            shapiro_df, levene_df, anova_df, residuals_shapiro_df, areas_gpd = future.result()
            all_shapiro.append(shapiro_df)
            all_levene.append(levene_df)
            all_anova.append(anova_df)
            all_residuals_shapiro.append(residuals_shapiro_df)
            all_areas_gpd.append(areas_gpd)

    # Combine results into final DataFrames
    shapiro_final = pd.concat(all_shapiro, ignore_index=True)
    levene_final = pd.concat(all_levene, ignore_index=True)
    anova_final = pd.concat(all_anova, ignore_index=True)
    residuals_shapiro_final = pd.concat(all_residuals_shapiro, ignore_index=True)
    areas_gpd_final = pd.concat(all_areas_gpd, ignore_index=True)

    return shapiro_final, levene_final, anova_final, residuals_shapiro_final, areas_gpd_final

def geom_to_polydata(coords):
    xy = np.array(coords)
    xyz = np.hstack((xy, np.zeros((len(xy), 1))))
    return pv.lines_from_points(xyz)

#############################################################################################

# LOAD SHAPEFILES (Interpretation boundary and linestring fractures)

data = gpd.read_file('boundary_area_4.shp')
fracs = gpd.read_file('Area4_reprojected_length.shp')

#############################################################################################

# # ACTIVATE THIS FOR SHAPEFILES WITH Z ALREADY INCLUDED IN THE GEOMETRY (e.g. seismic slices)

# center = np.array(data.centroid[0].coords).flatten()
#
# data = data.translate(-center[0], -center[1])
# fracs = fracs.translate(-center[0], -center[1])
#
# frac_appender = vtkAppendPolyData()
#
# # Loop over 3D LineStrings
# for frac in fracs.geometry:
#     xyz = np.array(frac.coords)
#     line = pv.lines_from_points(xyz)
#     frac_appender.AddInputData(line)
#
# # Finalize VTK structure
# frac_appender.Update()
#
# boundary = data.geometry
# boundary_contour = boundary.boundary
# lines = boundary_contour.explode()
#
# # if isinstance(boundary_contour, MultiLineString):
# #     lines = [line for line in boundary_contour.geoms]
# # else:
# #     lines = [boundary_contour]
#
# plotter = pv.Plotter()

#############################################################################################

# ACTIVATE THIS FOR SHAPEFILES WITHOUT Z IN THE GEOMETRY (e.g. digitalized fractures from orthomosaics)

center = np.array(data.centroid[0].coords).flatten()

data = data.translate(-center[0], -center[1])
fracs = fracs.translate(-center[0], -center[1])

frac_appender = vtkAppendPolyData()

#without progress bar

# for frac in fracs.geometry:
#     xy = np.array(frac.coords)
#     xyz = np.hstack((xy, np.zeros((len(xy), 1))))
#     frac = pv.lines_from_points(xyz)
#     frac_appender.AddInputData(frac)
# frac_appender.Update()

#with progress bar

# for frac_geom in tqdm(fracs.geometry, total=len(fracs.geometry), desc="Processing fractures"):
#     xy = np.array(frac_geom.coords)
#     xyz = np.hstack((xy, np.zeros((len(xy), 1))))
#     frac = pv.lines_from_points(xyz)
#     frac_appender.AddInputData(frac)
#
# frac_appender.Update()

#parallel version, mandatory for large datasets (>100,000 linestring)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # Prepare the list of coordinates
    coords_list = [geom.coords for geom in fracs.geometry]

    # Parallel conversion to PolyData
    with concurrent.futures.ProcessPoolExecutor() as executor:
        polydata_list = list(tqdm(
            executor.map(geom_to_polydata, coords_list),
            total=len(coords_list),
            desc="Processing fractures"
        ))

    frac_appender = vtkAppendPolyData()
    for frac in polydata_list:
        frac_appender.AddInputData(frac)
    frac_appender.Update()


boundary = data.geometry
boundary_contour = boundary.boundary

lines = boundary_contour.explode()
# if isinstance(boundary_contour, MultiLineString):
#     lines = [line for line in boundary_contour.geoms]
# else:
#     lines = [boundary_contour]

plotter = pv.Plotter()

#############################################################################################

#run function in parallel

min_rad = 0.005  # minimum radius
intermediate_rad_1 = 3  # intermediate radius
intermediate_rad_2 = 3.7  # intermediate radius
max_rad = 18     # maximum radius
n_values_1 = 10  # number of radii in first range
n_values_2 = 10
n_values_3 = 50# number of radii in second range
n_points = 100 # number of scan areas

# First range: from min_rad to intermediate_rad
rads_1 = np.linspace(min_rad, intermediate_rad_1, n_values_1, endpoint=False)

# Second range: from intermediate_rad to max_rad
rads_2 = np.linspace(intermediate_rad_1, intermediate_rad_2, n_values_2)

# Third range: from intermediate_rad_2 to max_rad
rads_3 = np.linspace(intermediate_rad_2, max_rad, n_values_3, endpoint=True)

# Merge into a single array
rads = np.concatenate((rads_1, rads_2, rads_3))

n_iterations = 100 # number of iterations
radius_range = (7, 10) # # REV range for ANOVA
alpha = 0.05

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # Call your parallel execution here
    shapiro_df, levene_df, anova_df, residuals_shapiro_df, areas_gpd = parallel_statistical_analysis(
        fracs=fracs,
        boundary=boundary
    )

    # Save to CSV
    shapiro_df.to_csv("shapiro_results_Bristol_100_scan_area_0.005_3_3.7_18.csv", index=False)
    levene_df.to_csv("levene_results_Bristol_100_scan_area_0.005_3_3.7_18.csv", index=False)
    anova_df.to_csv("anova_results_Bristol_100_scan_area_0.005_3_3.7_18.csv", index=False)
    residuals_shapiro_df.to_csv("residuals_shapiro_results_Bristol_100_scan_area_0.005_3_3.7_18.csv", index=False)
    areas_gpd.to_csv('areas_gpd_Bristol_100_scan_area_0.005_3_3.7_18.csv', index=False)