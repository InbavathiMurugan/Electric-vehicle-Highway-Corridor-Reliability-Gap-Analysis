# EV Highway Corridor Reliability Gap Analysis

A Python-based data analysis and geospatial project examining electric-vehicle (EV) charging coverage along major national highways in India and its relationship with state- and Union Territory-level EV registrations.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives and Analysis Questions](#objectives-and-analysis-questions)
- [Datasets and Sources](#datasets-and-sources)
- [Tools and Technologies](#tools-and-technologies)
- [Project Workflow](#project-workflow)
- [Data Cleaning and Transformation](#data-cleaning-and-transformation)
- [Analysis Performed](#analysis-performed)
- [Key Findings](#key-findings)
- [Highway Coverage Scorecard](#highway-coverage-scorecard)
- [Recommendations](#recommendations)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements and Data Attribution](#acknowledgements-and-data-attribution)
- [Author](#author)

## Project Overview

**Industry:** Electric Vehicles (EVs) / Transportation and Mobility

The growth of electric vehicles in India makes dependable charging infrastructure important not only in cities but also along major highway corridors. This project combines EV registration data, charging-station locations, national highway geometry, and state/Union Territory boundaries to examine potential fast-charging infrastructure gaps.

The project focuses on six national highways:

- NH-44
- NH-48
- NH-16
- NH-66
- NH-544
- NH-52

The analysis also compares charging infrastructure with EV registrations across all 36 states and Union Territories.

## Problem Statement

A high number of charging stations within a state does not necessarily mean that fast chargers are conveniently distributed along its highways. This project evaluates highway-level charging coverage, identifies long stretches without a nearby fast charger under the project's distance rule, and examines how these gaps relate to state-level EV registrations and charging-infrastructure availability.

## Objectives and Analysis Questions

The analysis addresses the following questions:

- How well are the selected highways covered by fast-charging infrastructure?
- Which highway segments have no fast charger within the defined distance threshold?
- What are the longest uncovered highway stretches?
- Does charger count alone represent corridor-level coverage adequately?
- How does charging infrastructure compare with EV registrations across states and Union Territories?
- Which states show a potential mismatch between EV demand indicators and charging availability?
- How do the results change under different gap distances and fast-charger definitions?
- How stable are highway ratings under alternative analytical assumptions?

## Datasets and Sources

The project uses four dataset groups.

| Dataset | Use | Source |
|---|---|---|
| EV registration data | State/UT and year-level EV registrations, including all EVs, electric cars, and electric buses | [Vahan Dashboard](https://vahan.parivahan.gov.in/vahan4dashboard/) |
| EV charging-station data | Charger location, operational status, and maximum charging power | [Open Charge Map API](https://openchargemap.org/site/develop/api) |
| National highway data | Highway network geometry and attributes | MoRTH via PM GatiShakti, accessed through [India Geodata](https://yashveeeeeeer.github.io/india-geodata/) |
| State and Union Territory boundaries | Spatial assignment and mapping of chargers and highway segments | [India Geodata](https://yashveeeeeeer.github.io/india-geodata/) |

### Registration data scope

The Vahan data used in this project covers calendar years **2017–2026**, the `ELECTRIC(BOV)` and `PURE EV` categories, active registrations, and all 36 states and Union Territories. **2026 is a partial year**, so it should not be interpreted as a complete annual total.

### Charging-station data scope

The charging-station dataset contained **1,979 raw records** before cleaning. Open Charge Map data is crowdsourced and licensed under **CC BY 4.0**; see [Acknowledgements and Data Attribution](#acknowledgements-and-data-attribution).

## Tools and Technologies

- Python
- Jupyter Notebook
- NumPy
- Pandas
- GeoPandas
- Shapely
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn

**Coordinate reference system:** `EPSG:7755` is used for projected distance and length calculations.

## Project Workflow

**Industry Selection → Problem Identification → Dataset Collection → Data Cleaning → Data Transformation → Data Analysis → Data Visualization → Insights → Recommendations**

## Data Cleaning and Transformation

The cleaning notebooks prepare the source datasets for analysis. The documented cleaning tasks include handling duplicate records, missing values, invalid geometries, coordinate issues, inconsistent state names, highway attributes, and charger operational status.

### Documented data counts

| Dataset / stage | Count |
|---|---:|
| Charging-station raw records | 1,979 |
| Charging-station cleaned records | 1,841 |
| Operational chargers used in the final analysis | 1,822 |
| Fast chargers using the primary ≥25 kW definition | 1,421 |
| National highway records before cleaning | 10,317 |
| National highway records after cleaning | 9,507 |

### Fast-charger definition

The primary analysis defines a fast charger as an **operational charger with maximum power ≥25 kW**. The dataset's own fast-charge flag disagreed with the power-based definition for 152 stations, so the flag is evaluated as an alternative definition during sensitivity analysis.

### Highway segmentation and gap rule

- The six selected highways are divided into segments of approximately **25 km**.
- The nearest operational fast charger is identified using each segment's midpoint.
- Distances are calculated as straight-line distances in the projected coordinate system.
- A segment is considered a gap when its nearest fast charger is **more than 50 km away**.
- For the highway scorecard, chargers within a **5 km corridor buffer** of the corresponding highway are counted.

These are project-specific analytical rules, not universal definitions of charging accessibility.

## Analysis Performed

### 1. EV registration analysis

- EV registrations by state and Union Territory.
- Year-based EV registration patterns.
- Comparison of all EV registrations with electric-car and electric-bus registrations.
- Identification of states with high EV registration totals and trends.

### 2. Charging-infrastructure analysis

- Charger operational status and maximum charging power.
- Fast-charger classification.
- State-level charger distribution and charger density.
- Charger power-band and operator-related exploration.

### 3. Highway coverage analysis

For NH-44, NH-48, NH-16, NH-66, NH-544, and NH-52, the analysis calculates:

- Highway length.
- Fast chargers within 5 km of the highway.
- Chargers per 100 km.
- Percentage of highway classified as covered.
- Uncovered highway length.
- Longest uncovered stretch.
- Highway reliability rating.

### 4. Highway gap analysis

The project identifies uncovered segments using the 50 km rule and summarizes uncovered length and long gaps by highway and by state.

### 5. State-level EV infrastructure gap analysis

The project calculates an **EV Desert Index** using three min-max-scaled components:

| Component | Weight |
|---|---:|
| EVs per charger | 40% |
| Share of highway in a gap | 35% |
| Charger sparsity per area | 25% |

The resulting index is scaled from 0 to 100. The weights are analytical assumptions and are tested through alternative weighting schemes.

### 6. Sensitivity analysis

The analysis tests gap-distance thresholds of **25 km, 50 km, 75 km, and 100 km** and compares fast-charger definitions of **≥22 kW, ≥25 kW, ≥50 kW, and the dataset's fast-charge flag**. Five alternative weighting schemes are also tested for the EV Desert Index.

### 7. K-Means clustering

K-Means clustering with **three clusters** is used as a supporting analysis to group states using the project's EV and charging-infrastructure characteristics.

## Key Findings

The findings below are from the project's documented analysis:

- The Vahan data used contains **10,377,820 EV registrations** across 2017–2026. The 2026 data is partial.
- Electric cars and buses account for approximately **7%** of the EV registrations in the dataset.
- The final charging dataset contains **1,822 operational chargers**, including **1,421 fast chargers** under the ≥25 kW definition.
- Approximately **98,551 km (60%)** of the mapped road network has no fast charger within 50 km.
- **NH-544** has the highest coverage among the six selected highways at **100%**.
- **NH-48** has **97.7%** coverage and is rated **Reliable**.
- **NH-52** has the lowest coverage among the selected highways at **50.0%**.
- The longest uncovered stretch among the selected highways is approximately **427 km on NH-44**.
- **NH-66** has 243 nearby fast chargers but still contains an uncovered stretch of approximately **376 km**, illustrating why charger counts alone do not describe corridor coverage.
- Maharashtra has the largest uncovered length across the selected key highways at approximately **891 km**, followed by Madhya Pradesh at approximately **796 km**.
- The highest overall EV Desert Index scores in the documented analysis are observed for **Bihar, Chandigarh, Assam, Tripura, and Punjab**.
- Highway ratings are broadly stable under the primary fast-charger definitions, while the stricter ≥50 kW definition changes some classifications.

## Highway Coverage Scorecard

| Highway | Length (km) | Fast chargers within 5 km | Chargers / 100 km | Covered (%) | Uncovered (km) | Longest gap (km) | Rating |
|---|---:|---:|---:|---:|---:|---:|---|
| NH-544 | 330.6 | 101 | 30.6 | 100.0 | 0.0 | 0.0 | Reliable |
| NH-48 | 2,542.5 | 191 | 7.5 | 97.7 | 58.2 | 27.1 | Reliable |
| NH-66 | 1,646.0 | 243 | 14.8 | 70.1 | 491.8 | 376.1 | Patchy |
| NH-16 | 1,711.8 | 38 | 2.2 | 64.4 | 609.0 | 126.2 | Poor |
| NH-44 | 3,558.1 | 186 | 5.2 | 56.0 | 1,566.8 | 426.6 | Poor |
| NH-52 | 2,228.7 | 19 | 0.9 | 50.0 | 1,114.7 | 173.4 | Poor |

### Highway rating criteria

- **Reliable:** More than 90% covered.
- **Patchy:** 70%–90% covered.
- **Poor:** 70% or below.

## Recommendations

Based on the documented findings:

- Prioritize specific uncovered highway stretches instead of relying only on state-level charger counts.
- Examine long gaps where a fast charger is not available within the project's 50 km threshold.
- Consider EV demand indicators alongside charger availability when identifying potential infrastructure priorities.
- Use corridor-level coverage metrics for planning, because a high charger count does not guarantee continuous highway coverage.
- Validate mapped gaps against current ground-level information because crowdsourced charger records may be incomplete or outdated.
- Compare results across different gap thresholds and fast-charger definitions rather than relying on a single assumption.
- In future work, incorporate road-network driving distance and traffic data for a more realistic assessment of charging accessibility.

## Visualizations

The following screenshot paths are documented in the project materials. Keep the files at these paths in the GitHub repository for the images to render.

### Fast-charging coverage on six highways

![Fast-Charging Coverage on 6 Highways](outputs/figures/10_gap_map_6_highways.png)

### Highway scorecard

![Highway Scorecard](outputs/figures/09_highway_scorecard.png)

### EV Desert Quadrant

![EV Desert Quadrant](outputs/figures/06_ev_desert_quadrant.png)

### State gap score map

![State Gap Score Map](outputs/figures/18_gap_score_map.png)

Additional project figures are documented in `outputs/figures/`.

## Project Structure

The following structure reflects the uploaded notebooks and the project paths documented in the analysis materials. Keep the directory names and filenames in the README aligned with the files you commit to GitHub.

EV_Highway_Corridor_Reliability_Gap_Analysis/
│
├── README.md
│
├── Dataset/
│   ├── Raw datasets/
│   │   ├── ev_charging_stations.csv
│   │   ├── States_and_Union_Territories.parquet
│   │   ├── National_Highways.parquet
│   │   ├── All_type_EV_registration.csv
│   │   └── Ev_chargers_cleaned.csv
│   │
│   └── Cleaned datasets/
│       ├── States_and_Union_Territories_Cleaned.parquet
│       ├── highways_clean.geojson
│       ├── EV_registration_cleaned.csv
│       └── Ev_chargers_cleaned.csv
│
├── notebooks/
│   └── EV_Highway_Corridor_Reliability_Gap_Analysis.ipynb
│
├── Python/
│   ├── Data_Cleaning.py
│   ├── Data_Loading.py
│   ├── Data_Visualization.py
│   └── Exploratory_Analysis.py
│
├── Visualization/
│   ├── figures/
│   │   ├── 01_ev_growth.png
│   │   ├── 02_top10_state_trend.png
│   │   ├── 03_charger_power.png
│   │   ├── 04_top_operators.png
│   │   ├── 05_charger_speed_mix.png
│   │   ├── 06_ev_desert_quadrant.png
│   │   ├── 07_top_ev_deserts.png
│   │   ├── 08_national_gap_map.png
│   │   ├── 09_highway_scorecard.png
│   │   ├── 10_gap_map_6_highways.png
│   │   ├── 11_gap_map_per_highway.png
│   │   ├── 12_state_priority_6_highways.png
│   │   ├── 13_sensitivity_gap_distance.png
│   │   ├── 14_sensitivity_fast_definition.png
│   │   ├── 15_top_states_chargers_vs_evs.png
│   │   ├── 16_spearman_heatmap.png
│   │   ├── 17_state_clusters.png
│   │   ├── 18_gap_score_map.png
│   │   ├── 19_top10_gap_score.png
│   │   └── 20_score_vs_corridor_gap.png
│   │
│   └── tables/
│       ├── highway_rating_by_definition.csv
│       ├── highway_scorecard.csv
│       ├── state_gap_master.csv
│       ├── state_priority_6_highways.csv
│       ├── national_segments_gap.geojson
│       ├── segments_6_highways_gap.geojson
│       └── chargers_clean.geojson
│
└── Documentation/
    ├── EV_Highway_Corridor_Gap_Analysis.pdf
    └── EV_Highway_Corridor_Gap_Analysis.pptx

The folder tree is a suggested repository layout based on the paths and outputs documented in the supplied notebooks and project notes. Include only files and folders that are actually present in your repository. The detailed report and `requirements.txt` can be added if they are part of the repository.

## Limitations

- Open Charge Map is crowdsourced, so charger records may be incomplete or outdated.
- Operational status is a snapshot and does not measure real-time charger uptime.
- Highway gaps are based on straight-line distance rather than actual driving distance.
- The analysis does not filter highway data by current road status.
- The 5 km corridor buffer, approximately 25 km segmentation, 50 km gap threshold, and ≥25 kW fast-charger definition are project-specific analytical choices.
- Vahan registration data is available at state level and does not directly represent highway traffic or route-level EV demand.
- 2026 registration data is partial.
- EV Desert Index weights are analytical assumptions, and state rankings can change under different weighting schemes.

## Future Work

- Repeat charger data collection over time to study infrastructure changes.
- Use driving distance along the road network instead of straight-line distance.
- Add traffic-volume and highway vehicle-flow data.
- Expand the analysis beyond the six selected highways.
- Compare Open Charge Map data with official state-wise charger counts.
- Estimate the minimum number of new charging locations required to reduce highway gaps below a target distance.
- Build an interactive dashboard or web map for route-level exploration.
- Incorporate charger reliability and uptime information when available.

## Acknowledgements and Data Attribution

This project uses publicly available data and resources from:

- **Open Charge Map contributors** — charging-station data. Open Charge Map data is licensed under **CC BY 4.0**; provide appropriate attribution when reusing it.
- **Vahan Dashboard** — EV registration data.
- **Ministry of Road Transport and Highways (MoRTH) / PM GatiShakti** — national highway data.
- **India Geodata project** — highway and state/Union Territory geospatial datasets.

## Author

- **Name:** Inbavathi Murugan
- **Student ID:** AF05309783
- **Organization:** Anudip Foundation
- **Course:** AIML
- **Batch Code:** ANP-D7444

## Conclusion

This project demonstrates why charger count alone is not sufficient to evaluate EV highway infrastructure. By combining EV registrations, charging-station locations, highway geometry, spatial segmentation, and gap analysis, it provides a corridor-level view of potential fast-charging accessibility.

The central question is not only **“How many chargers does a state have?”**, but also **“Can an EV driver find a fast charger along the highway when needed?”**
