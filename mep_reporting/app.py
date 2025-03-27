import marimo

__generated_with = "0.9.27"
app = marimo.App(
    width="medium",
    layout_file="layouts/app.grid.json",
    css_file="assets/custom.css",
)


@app.cell(hide_code=True)
def _():
    # Import initial libraries
    import marimo as mo
    import pandas as pd
    import io
    import yaml
    import altair as alt
    import numpy as np
    import matplotlib.pyplot as plt

    from utils import to_snake_case_inplace
    return alt, io, mo, np, pd, plt, to_snake_case_inplace, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Data import""")
    return


@app.cell(hide_code=True)
def _(mo):
    # Define the import buttons
    buildings_file_button = mo.ui.file(kind = "button", filetypes = [".csv"], label="Upload", multiple = False)

    systems_file_button = mo.ui.file(kind = "button", filetypes = [".csv"], label="Upload", multiple = False)
    return buildings_file_button, systems_file_button


@app.cell(hide_code=True)
def _(buildings_file_button, mo, systems_file_button):
    # Define the validation messages
    if buildings_file_button.name() is not None: 
        buildings_file_message = mo.md("✅ File uploaded successfully !")
    else: 
        buildings_file_message = mo.md("⚠️ No file uploaded yet")

    if systems_file_button.name() is not None: 
        systems_file_message = mo.md("✅ File uploaded successfully !")
    else: 
        systems_file_message = mo.md("⚠️ No file uploaded yet")

    # Building boolean assessing if all files have been uploaded
    all_files_uploaded = buildings_file_button.name() is not None and systems_file_button.name() is not None

    if all_files_uploaded: 
        file_message = mo.callout(value = "✅ File uploaded successfully !", kind = "success")
    else: 
        file_message = mo.callout(value = "⚠️ No file uploaded yet. Please upload file to pursue the analysis", kind = "warn")

    # Group all import components together
    file_buttons = mo.vstack([
        mo.hstack([mo.md("Upload your buildings file"), buildings_file_button], justify = "start"),
        mo.hstack([mo.md("Upload your systems file"), systems_file_button], justify = "start"),
        file_message
    ])

    # Printing the content
    mo.vstack([
        mo.md("### Upload Your Data"), 
        file_buttons
    ])
    return (
        all_files_uploaded,
        buildings_file_message,
        file_buttons,
        file_message,
        systems_file_message,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Reading the files content""")
    return


@app.cell(hide_code=True)
def _(buildings_file_button, io, mo, pd, systems_file_button):
    # Read the files
    buildings = pd.read_csv(io.BytesIO(buildings_file_button.value[0].contents))
    systems = pd.read_csv(io.BytesIO(systems_file_button.value[0].contents))

    # Present files as accordion
    mo.accordion(
        {"Original buildings file": buildings,
        "Original systems file": systems}
    )
    return buildings, systems


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Renaming the columns of the files""")
    return


@app.cell
def _(buildings, mo, systems, to_snake_case_inplace):
    # Convert the column names
    to_snake_case_inplace(buildings)
    to_snake_case_inplace(systems)

    # Present files as accordion
    mo.accordion(
        {"New buildings file": buildings,
         "New systems file": systems}
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Creating Building type and Dataset dataframes""")
    return


@app.cell(hide_code=True)
def _(pd, to_snake_case_inplace):
    # Creating dataframes from Building type and Dataset (that will be used to expand the dataframe buildings)
    # ============================

    # Building type
    building_type_df = pd.read_csv("assets/Building type.csv", sep=";")
    to_snake_case_inplace(building_type_df)

    # Dataset - Energy systems
    energy_systems_df = pd.read_csv("assets/Energy systems table 1.csv", sep=";")
    to_snake_case_inplace(energy_systems_df)

    # Dataset - Energy carriers
    emissions_factor_df = pd.read_csv(
        "assets/Emissions factor table 1.csv", sep=";"
    )
    to_snake_case_inplace(emissions_factor_df)
    return building_type_df, emissions_factor_df, energy_systems_df


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Data expansion""")
    return


@app.cell(hide_code=True)
def cell_11(building_type_df, buildings, mo):
    # Create an anchor
    anchor_cell_11 = None

    # Expand buildings dataframes

    # ========= Buildings =========

    # Add the 'building_sector' and 'parent_type' columns to buildings

    try:
        buildings.drop(columns=['parent_type','building_sector'], inplace=True)
    except KeyError:
        pass

    print(buildings.columns.to_list())
    print(building_type_df.columns.to_list())
    buildings[['building_sector','parent_type']] = buildings.merge(building_type_df, how="left", on="building_type")[['building_sector','parent_type']]

    # Fill missing values with "Unassigned" using explicit assignment - SHOULD I DELETE THAT? (IF SO, THEN I NEED TO CHANGE THE PART ON FINAL ENERGY DEMAND)
    #buildings["parent_type"] = buildings["parent_type"].fillna("Unassigned")
    #buildings["building_sector"] = buildings["building_sector"].fillna("Unassigned")

    # Determine if "Construction_period" contains ">" (returns True/False)
    buildings["greater_than_construction_period"] = buildings["construction_period"].astype(str).str.contains(">", na=False)

    # Display the DataFrame
    mo.accordion(
        {
            "Expanded buildings file": mo.ui.table(buildings, selection=None, show_column_summaries=False)
        }
    )
    return (anchor_cell_11,)


@app.cell(hide_code=True)
def cell_12(
    anchor_cell_11,
    buildings,
    emissions_factor_df,
    energy_systems_df,
    mo,
    systems,
):
    # Link to anchor of cell 11
    _ = anchor_cell_11

    # Create anchor of cell 12
    anchor_cell_12 = None

    # Expand systems dataframes

    # ========= Systems =========

    # 1. Add e≠rgycarrierenergy_carrier from e≠rgysystemsdfenergy_systems_df using systemtypesystem_type
    try:
        systems.drop(columns=['energy_carrier'], inplace=True)
    except KeyError:
        pass

    systems['energy_carrier'] = systems.merge(energy_systems_df, 
                            how="left", 
                            left_on="system_type", 
                            right_on="energy_systems")['energy_carrier']

    # 2. Compute final energy consumption final_energy_consumption 
    fuel_cols = ["gas_consumption", "oil_consumption", "wood_chips_consumption", "wood_logs_consumption", "pellets_consumption"]
    systems["final_energy_consumption"] = systems[fuel_cols].sum(axis=1)

    # If the systemtypesystem_type is "Heat transfer station", add heat⊃plyheat_supply to final energy consumption final_energy_consumption
    systems.loc[systems["system_type"] == "Heat transfer station", "final_energy_consumption"] +=systems["heat_supply"]

    # 3. Compute ghgemissionsghg_emissions 
    # Merge expandedsystems1expanded_systems1 with emissionsfac→rdfemissions_factor_df to obtain emission factors

    try:
        systems.drop(columns=['ghg_emissions'], inplace=True)
    except KeyError:
        pass

    systems['ghg_emissions'] = systems.merge(emissions_factor_df[['energy_carriers', 'ghg_emissions']], 
                            how="left",
                            left_on="energy_carrier", 
                            right_on="energy_carriers")['ghg_emissions']

    # Convert ghgemissionsghg_emissions values from string format (e.g., "2,5") to float format (e.g., "2.5")
    systems["ghg_emissions"] = systems["ghg_emissions"].str.replace(",", ".").astype(float)

    # Multiply the GHG emission factor by the final energy consumption
    systems["ghg_emissions"] *=systems["final_energy_consumption"]

    # 4. Add build∈gsec→rbuilding_sector
    try:
        systems.drop(columns=['building_sector'], inplace=True)
    except KeyError:
        pass

    systems['building_sector'] = systems.merge(buildings[['building_id', 'building_sector']], 
                                              how="left", 
                                              left_on="building_id", 
                                              right_on="building_id")['building_sector']

    # Display the DataFrame
    mo.accordion(
        {
            "Expanded systems file": mo.ui.table(systems, selection=None, show_column_summaries=False)
        }
    )
    return anchor_cell_12, fuel_cols


@app.cell(hide_code=True)
def cell_13(anchor_cell_11, buildings, mo, pd):
    # Link to anchor of cell 11
    _ = anchor_cell_11

    # Create dataframes for graph generation

    # ========= Construction periods =========

    # Provide values for "construction_periods"
    construction_periods_values = ["", "1200-1918", "1919-1948", "1949-1978", "1979-1990", 
                            "1991-2000", "2001-2010", "2011-2019", ">2019"]

    # Create DataFrame with the given column names
    construction_periods_df = pd.DataFrame({
        "construction_periods": construction_periods_values,
        "energy_reference_area": [None] * len(construction_periods_values),  # To be filled later
        "number_of_buildings": [None] * len(construction_periods_values),  # To be filled later
        "labels": ["Unassigned"] + construction_periods_values[1:]  # First row labeled "Unassigned", others same as periods
    })

    # Compute the values for "energy_reference_area"
    construction_periods_df["energy_reference_area"] = construction_periods_df["construction_periods"].apply(
        lambda x: buildings.loc[buildings["greater_than_construction_period"] == True, "energy_reference_area"].sum()
        if ">" in x else buildings.loc[buildings["construction_period"] == x, "energy_reference_area"].sum()
    )

    # Compute the values for "number_of_buildings"
    construction_periods_df["number_of_buildings"] = construction_periods_df["construction_periods"].apply(
        lambda x: buildings.loc[buildings["greater_than_construction_period"] == True].shape[0]
        if ">" in x else buildings.loc[(buildings["construction_period"] == x) & (buildings["building_id"].notna())].shape[0]
    )

    # Display the DataFrame
    mo.accordion(
        {
            "Construction periods": mo.ui.table(construction_periods_df, selection=None, show_column_summaries=False)
        }
    )
    return construction_periods_df, construction_periods_values


@app.cell(hide_code=True)
def _(anchor_cell_11, buildings, mo, pd):
    # Link to anchor of cell 11
    _ = anchor_cell_11

    # ========= Heat demand according to Building sector =========

    # Provide values for "building_sector"
    unique_building_sectors = buildings["building_sector"].dropna().unique()

    # Create heat_demand_building_sector_df with the necessary columns
    heat_demand_building_sector_df = pd.DataFrame({
        "building_sector": unique_building_sectors
    })

    # Ensure consistent formatting for matching keys
    buildings["building_sector"] = buildings["building_sector"].astype(str).str.strip()
    heat_demand_building_sector_df["building_sector"] = heat_demand_building_sector_df["building_sector"].astype(str).str.strip()

    # Fill NaN values in relevant numerical columns before summing
    for cols in ["energy_reference_area", "total_heat_demand", "space_heating_demand", "domestic_hot_water_demand"]:
        buildings[cols] = buildings[cols].fillna(0)

    # Compute values for "number_of_buildings"
    heat_demand_building_sector_df["number_of_buildings"] = heat_demand_building_sector_df["building_sector"].map(
        buildings.groupby("building_sector")["building_sector"].count()
    ).fillna(0).astype(int)

    # Compute values for "energy_reference_area"
    heat_demand_building_sector_df["energy_reference_area"] = heat_demand_building_sector_df["building_sector"].map(
        buildings.groupby("building_sector")["energy_reference_area"].sum()
    ).fillna(0)

    # Compute values for "total_heat_demand"
    heat_demand_building_sector_df["total_heat_demand"] = heat_demand_building_sector_df["building_sector"].map(
        buildings.groupby("building_sector")["total_heat_demand"].sum()
    ).fillna(0).div(1_000_000).round(1)

    # Compute values for "space_heating_demand"
    heat_demand_building_sector_df["space_heating_demand"] = heat_demand_building_sector_df["building_sector"].map(
        buildings.groupby("building_sector")["space_heating_demand"].sum()
    ).fillna(0).div(1_000_000).round(1)

    # Compute values for "hot_water_demand"
    heat_demand_building_sector_df["hot_water_demand"] = heat_demand_building_sector_df["building_sector"].map(
        buildings.groupby("building_sector")["domestic_hot_water_demand"].sum()
    ).fillna(0).div(1_000_000).round(1)

    # Display the DataFrame
    mo.accordion(
        {
            "Heat demand according to Building sector": mo.ui.table(heat_demand_building_sector_df, selection=None, show_column_summaries=False)
        }
    )
    return cols, heat_demand_building_sector_df, unique_building_sectors


@app.cell(hide_code=True)
def _(anchor_cell_11, buildings, mo, pd):
    # Link to anchor of cell 11
    _ = anchor_cell_11

    # Create dataframes for graph generation

    # ========= Heat demand according to Parent type ========= 

    # Provide values for "parent_type"
    unique_parent_types = buildings["parent_type"].dropna().unique()

    # Create heat_demand_parent_type_df with the necessary columns
    heat_demand_parent_type_df = pd.DataFrame({
        "parent_type": unique_parent_types
    })

    # Ensure consistent formatting for matching keys
    buildings["parent_type"] = buildings["parent_type"].astype(str).str.strip()
    heat_demand_parent_type_df["parent_type"] = heat_demand_parent_type_df["parent_type"].astype(str).str.strip()

    # Fill NaN values in relevant numerical columns before summing
    for column in ["energy_reference_area", "total_heat_demand", "space_heating_demand", "domestic_hot_water_demand"]:
        buildings[column] = buildings[column].fillna(0)

    # Compute values for "number_of_buildings"
    heat_demand_parent_type_df["number_of_buildings"] = heat_demand_parent_type_df["parent_type"].map(
        buildings.groupby("parent_type")["parent_type"].count()
    ).fillna(0).astype(int)

    # Compute values for "energy_reference_area"
    heat_demand_parent_type_df["energy_reference_area"] = heat_demand_parent_type_df["parent_type"].map(
        buildings.groupby("parent_type")["energy_reference_area"].sum()
    ).fillna(0)

    # Compute values for "total_heat_demand"
    heat_demand_parent_type_df["total_heat_demand"] = heat_demand_parent_type_df["parent_type"].map(
        buildings.groupby("parent_type")["total_heat_demand"].sum()
    ).fillna(0).div(1_000_000).round(1)

    # Compute values for "space_heating_demand"
    heat_demand_parent_type_df["space_heating_demand"] = heat_demand_parent_type_df["parent_type"].map(
        buildings.groupby("parent_type")["space_heating_demand"].sum()
    ).fillna(0).div(1_000_000).round(1)

    # Compute values for "hot_water_demand"
    heat_demand_parent_type_df["hot_water_demand"] = heat_demand_parent_type_df["parent_type"].map(
        buildings.groupby("parent_type")["domestic_hot_water_demand"].sum()
    ).fillna(0).div(1_000_000).round(1)

    # Compute the "rank" based on "total_heat_demand"
    heat_demand_parent_type_df["rank"] = heat_demand_parent_type_df["total_heat_demand"].rank(
        method="min", ascending=False
    ).fillna("").astype(str)  # Convert NaN ranks to empty strings

    # Display the DataFrame
    mo.accordion(
        {
            "Heat demand according to Parent type": mo.ui.table(heat_demand_parent_type_df, selection=None, show_column_summaries=False)
        }
    )
    return column, heat_demand_parent_type_df, unique_parent_types


@app.cell(hide_code=True)
def cell_16(anchor_cell_12, mo, np, pd, systems):
    # Link to anchor of cell 12
    _ = anchor_cell_12

    # Create dataframes for graph generation

    # ========= Final energy demand per building sector and energy carrier ========= TO CHANGE : WRONG VALUES. I THINK IT IS BECAUSE OF HOW Commercial and traffic, Residential etc are written in the dataframes systems and buildings

    # Convert string "nan" to actual NaN values
    systems["building_sector"] = systems["building_sector"].replace("nan", np.nan)

    # Fill NaN values with "Unassigned" or another appropriate category
    systems["building_sector"] = systems["building_sector"].fillna("Unassigned")

    # Standardize column formatting
    systems["building_sector"] = systems["building_sector"].astype(str).str.strip().str.lower()
    systems["final_energy_consumption"] = systems["final_energy_consumption"].fillna(0)

    # Provide values for "energy_carriers"
    unique_energy_carriers = systems["energy_carrier"].dropna().unique()

    # Create an empty DataFrame
    energy_consumption_df = pd.DataFrame({"energy_carriers": unique_energy_carriers})

    # Define expected sectors
    sectors = ["commercial and traffic", "residential", "industry and production", "public buildings"]  # Include "Unassigned" now

    # Compute energy consumption for each sector
    for sector in sectors:
        energy_consumption_df[sector] = energy_consumption_df["energy_carriers"].apply(
            lambda x: systems.loc[
                (systems["energy_carrier"] == x) & 
                (systems["building_sector"] == sector),  
                "final_energy_consumption"
            ].sum() / 1000000  # Convert kWh to MWh
        )

    # Compute the total energy consumption per energy carrier
    energy_consumption_df["total"] = energy_consumption_df[sectors].sum(axis=1)

    # Add the last row ("Total" row)
    total_row = pd.DataFrame({
        "energy_carriers": ["Total"],
        "commercial and traffic": [energy_consumption_df["commercial and traffic"].sum()],
        "residential": [energy_consumption_df["residential"].sum()],
        "industry and production": [energy_consumption_df["industry and production"].sum()],
        "public buildings": [energy_consumption_df["public buildings"].sum()],
        "total": [energy_consumption_df["total"].sum()]
    })

    energy_consumption_df = pd.concat([energy_consumption_df, total_row], ignore_index=True)

    # Compute the "relative" column
    total_energy = energy_consumption_df.loc[energy_consumption_df["energy_carriers"] == "Total", "total"].values[0]
    energy_consumption_df["relative"] = energy_consumption_df["total"] / total_energy

    # Display the DataFrame
    mo.accordion(
        {
            "Final energy demand per building sector and energy carrier": mo.ui.table(energy_consumption_df, selection=None, show_column_summaries=False)
        }
    )
    return (
        energy_consumption_df,
        sector,
        sectors,
        total_energy,
        total_row,
        unique_energy_carriers,
    )


@app.cell(hide_code=True)
def _():
    # Create dataframes for graph generation

    # ========= GHG emissions per building sector and energy carrier ========= TO DO (haven't done cause same as previous and still problems)
    return


@app.cell(hide_code=True)
def _(mo, pd, systems):
    # Create dataframes for graph generation

    # ========= Energy systems =========

    # Create heating_system_type_df
    print(systems.columns.to_list())
    heating_system_type_df = pd.DataFrame({"system_type": systems["system_type"].dropna().unique()})

    # Add "min" and "max" rows
    heating_system_type_df = pd.concat([heating_system_type_df, pd.DataFrame({"system_type": ["min", "max"]})], ignore_index=True)

    # Define system size categories (column names and min/max values)
    size_ranges = {
        "Up to 10 kW": (0, 10),
        "10 - 25 kW": (10, 25),
        "25 - 50 kW": (25, 50),
        "50 - 100 kW": (50, 100),
        "More than 100 kW": (100, 1_000_000_000)  # Large upper bound
    }

    # Initialize the new columns
    for col in size_ranges.keys():
        heating_system_type_df[col] = 0

    # Fill min/max values and count matching systems
    for system in heating_system_type_df["system_type"]:
        for col, (min_val, max_val) in size_ranges.items():
            if system in ["min", "max"]:
                heating_system_type_df.loc[heating_system_type_df["system_type"] == system, col] = min_val if system == "min" else max_val
            else:
                heating_system_type_df.loc[heating_system_type_df["system_type"] == system, col] = systems[
                    (systems["system_type"] == system) &
                    (systems["installed_system_size"] > min_val) &
                    (systems["installed_system_size"] <= max_val)
                ].shape[0]  # Count matching rows

    # Display the DataFrame
    mo.accordion(
        {
            "Energy systems": mo.ui.table(heating_system_type_df, selection=None, show_column_summaries=False)
        }
    )
    return (
        col,
        heating_system_type_df,
        max_val,
        min_val,
        size_ranges,
        system,
    )


@app.cell(hide_code=True)
def _(building_type_df, buildings, mo):
    # Create dataframes for graph generation

    # ========= Heat demand according to building type =========

    # Extract unique building types & sectors from "Building type" DataFrame
    heat_demand_building_type_df = building_type_df[["building_sector", "building_type"]].drop_duplicates()

    # Compute energy reference area per building type
    print(buildings.columns.to_list())
    energy_reference_area = buildings.groupby("building_type")["energy_reference_area"].sum().reset_index()

    # Step 3: Compute building counts per building type
    building_counts = buildings["building_type"].value_counts().reset_index()
    building_counts.columns = ["building_type", "building_counts"]

    # Merge energy reference area and building counts into heat_demand_building_type_df
    heat_demand_building_type_df = heat_demand_building_type_df.merge(energy_reference_area, 
                                                                      how="left", 
                                                                      on="building_type")

    heat_demand_building_type_df = heat_demand_building_type_df.merge(building_counts, 
                                                                      how="left", 
                                                                      on="building_type")

    # Fill missing values with 0 (if no buildings exist for a type)
    heat_demand_building_type_df["energy_reference_area"] = heat_demand_building_type_df["energy_reference_area"].fillna(0)
    heat_demand_building_type_df["building_counts"] = heat_demand_building_type_df["building_counts"].fillna(0).astype(int)

    # Remove rows where building_counts is 0
    heat_demand_building_type_df = heat_demand_building_type_df[heat_demand_building_type_df["building_counts"] > 0]

    # Display the DataFrame
    mo.accordion(
        {
            "Heat demand according to Building type": mo.ui.table(
                heat_demand_building_type_df.reset_index(drop=True),  # <-- Reset index here
                selection=None,
                show_column_summaries=False
            )
        }
    )
    return (
        building_counts,
        energy_reference_area,
        heat_demand_building_type_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate your graphs""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Generate the same graphs as used in the reporting template""")
    return


@app.cell(hide_code=True)
def _(heat_demand_building_type_df, mo):
    import plotly.graph_objects as go

    # Step 1: Aggregate energy reference area by sector
    sector_sum_df = heat_demand_building_type_df.groupby('building_sector')['energy_reference_area'].sum().reset_index()

    # Step 2: Find top 2 building types by energy reference area within each sector
    top_types_per_sector = heat_demand_building_type_df.groupby('building_sector') \
        .apply(lambda x: x.nlargest(2, 'energy_reference_area')) \
        .reset_index(drop=True)

    # Aggregate building counts by sector
    building_counts_by_sector = heat_demand_building_type_df.groupby('building_sector')['building_counts'].sum().reset_index()

    # Merge the counts back into the sector data
    sector_sum_df = sector_sum_df.merge(building_counts_by_sector, on='building_sector', how='left')

    # Update sector labels with a prefix for clarity
    sector_sum_df['labeled_sector'] = "Sector - " + sector_sum_df['building_sector']

    # Update building type labels with a prefix for clarity
    top_types_per_sector['labeled_type'] = "Type - " + top_types_per_sector['building_type']

    # Create the inner doughnut for sectors
    inner_pie = go.Pie(labels=sector_sum_df['labeled_sector'],
                       values=sector_sum_df['energy_reference_area'],
                       name="Sector",
                       domain={'x': [0.2, 0.8], 'y': [0.2, 0.8]},  # Adjusted for better fit
                       hole=0.6,  # Adjusted hole size
                       sort=False,
                       textinfo='value',  # Display values
                       textposition='inside',  # Position text inside the slices
                       insidetextorientation='radial',  # Ensure text orientation is readable
                       customdata=sector_sum_df['building_counts'],  # Use custom data for displaying counts
                       hoverinfo='label+percent+value+name',  # Hover info includes all details
                       texttemplate="%{customdata}")  # Template for text display

    # Create the outer doughnut for top 2 building types in each sector
    outer_pie = go.Pie(labels=top_types_per_sector['labeled_type'],
                       values=top_types_per_sector['energy_reference_area'],
                       name="Building Type",
                       domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},  # Adjusted for better fit
                       hole=0.8,  # Make the outer hole larger so the two charts touch
                       sort=False,
                       textinfo='none')  # Remove text from outer doughnut to avoid clutter

    fig = go.Figure(data=[inner_pie, outer_pie])

    # Update layout and annotations for clarity
    fig.update_layout(
        title='Building types sized by energy reference area',
        title_x=0.5,  # Center the title
        margin=dict(t=50, l=0, b=0, r=0),  # Top margin increased to accommodate title
        legend_title_text='Categories'  # Optional: Add a title to the legend
    )

    # Display the chart
    mo.ui.plotly(
        figure=fig,
        label='Energy Reference Area Distribution'
    )
    return (
        building_counts_by_sector,
        fig,
        go,
        inner_pie,
        outer_pie,
        sector_sum_df,
        top_types_per_sector,
    )


@app.cell(hide_code=True)
def _(construction_periods_df, go, mo):
    # Drop NaNs and filter out 'Unassigned'
    data = construction_periods_df.dropna(subset=['number_of_buildings'])
    data = data[data['labels'] != "Unassigned"]

    # Create the bar chart with a pale, nice color
    figu = go.Figure(data=[
        go.Bar(
            x=data['labels'],
            y=data['number_of_buildings'],
            marker_color='powderblue',      # Pale, clean color
            marker_line_color='grey',       # Light border for contrast
            marker_line_width=0.7
        )
    ])

    figu.update_layout(
        title='Construction Periods',
        xaxis_title='Construction Period',
        yaxis_title='Number of Buildings',
        xaxis={'type': 'category'},
        yaxis=dict(type='linear'),
        plot_bgcolor='white'
    )

    mo.ui.plotly(
        figure=figu,
        label='Construction Periods Chart'
    )
    return data, figu


@app.cell(hide_code=True)
def _(construction_periods_df, go, mo):
    import plotly.express as px  # Importing Plotly Express for color access

    # Assuming your DataFrame constructionperiodsdfconstruction_periods_df is already prepared and filled with the correct data
    data_cons = construction_periods_df.dropna(subset=['number_of_buildings'])  # Ensure there are no NaN values in the data to be plotted
    data_cons = data_cons[data_cons['labels'] != "Unassigned"]  # Exclude rows where the label is 'Unassigned'

    # Creating the pie chart using Plotly
    figur = go.Figure(data=[
        go.Pie(
            labels=data_cons['labels'],  # 'labels' column for pie chart labels
            values=data_cons['number_of_buildings'],  # 'number_of_buildings' column for pie chart values
            marker_colors=px.colors.qualitative.Pastel,  # Optional: specifying colors using Plotly Express
            textinfo='percent',  # Display only percentage on the chart
            hoverinfo='label+percent',  # Display labels and percentage on hover
            texttemplate='%{percent:.0%}'  # Rounding percentages to whole numbers
        )
    ])

    figur.update_layout(
        title='Number of Buildings per Construction Period'
    )

    # Displaying the figure in a Marimo UI component
    mo.ui.plotly(
        figure=figur,
        label='Buildings Distribution Chart'
    )
    return data_cons, figur, px


@app.cell(hide_code=True)
def _(go, heating_system_type_df, mo):
    # Remove 'min' and 'max' rows
    filtered_df = heating_system_type_df[~heating_system_type_df["system_type"].isin(["min", "max"])]

    # Get kW category columns
    kw_categories = list(filtered_df.columns[1:])  # skip 'system_type'

    # Create the bar chart
    fig_bar = go.Figure()

    # Add a trace for each heating system type
    for idx, row in filtered_df.iterrows():
        system_type = row['system_type']
        counts = [row[col] for col in kw_categories]  # get counts across kW categories
        fig_bar.add_trace(go.Bar(
            x=kw_categories,
            y=counts,
            name=system_type
        ))

    # Layout
    fig_bar.update_layout(
        title="Number of Heating Systems by Type and Size",
        xaxis_title="System Size",
        yaxis_title="Number of Heating Systems",
        barmode='group',  # Show bars side by side
        template="plotly"
    )

    # Display in Marimo
    mo.ui.plotly(figure=fig_bar, label="Energy systems")
    return (
        counts,
        fig_bar,
        filtered_df,
        idx,
        kw_categories,
        row,
        system_type,
    )


@app.cell(hide_code=True)
def _(go, heat_demand_building_sector_df, mo):
    # Filter out 'nan' or empty strings from building_sector
    df_filtered = heat_demand_building_sector_df[
        (heat_demand_building_sector_df["building_sector"].notna()) &
        (heat_demand_building_sector_df["building_sector"].str.lower() != "nan") &
        (heat_demand_building_sector_df["building_sector"].str.strip() != "")
    ]

    # Create the stacked bar chart
    fig_barchart = go.Figure()

    # Add Space Heating Demand
    fig_barchart.add_trace(go.Bar(
        x=df_filtered["building_sector"],
        y=df_filtered["space_heating_demand"],
        name="Space Heating",
        hovertemplate='Space Heating: %{y} GWh/y<br>Sector: %{x}<extra></extra>'
    ))

    # Add Hot Water Demand
    fig_barchart.add_trace(go.Bar(
        x=df_filtered["building_sector"],
        y=df_filtered["hot_water_demand"],
        name="Hot Water",
        hovertemplate='Hot Water: %{y} GWh/y<br>Sector: %{x}<extra></extra>'
    ))

    # Update layout for stacking
    fig_barchart.update_layout(
        barmode='stack',
        title='Total heat demand per building sector [GWh/y]',
        xaxis_title='Building Sector',
        yaxis_title='Heat Demand [GWh/y]',
        legend_title='Demand Type',
        hovermode='x unified'
    )

    # Show the plot in Marimo
    mo.ui.plotly(
        figure=fig_barchart,
        label="Total heat demand per building sector [GWh/y]"
    )
    return df_filtered, fig_barchart


if __name__ == "__main__":
    app.run()
