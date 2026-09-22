# Electricity-Price-Forecasting-for-Power-to-X-Off-Take-Agreements
This repository contains the data-processing procedures, forecasting models, parameter specifications, and simulation scripts required to reproduce the main results presented in the paper titled: "Forecast Accuracy vs. Decision Value in Electricity Price Forecasting for Power-to-X Off-Take Agreements"

## Data Sources

The datasets used in this study are publicly available from the following sources.

### 1. Hourly electricity prices and renewable generation

Historical hourly electricity prices for the Danish bidding zone (DK1) can be obtained from the [Energy-Charts platform](https://www.energy-charts.info/charts/power/chart.htm?l=en&c=DK&interval=year&year=2025&legendItems=0xfv2).

The same platform also provides historical electricity generation data by technology, including **wind and solar generation**. These data can be accessed by selecting the relevant country/bidding zone, year, and generation technologies.

> **Note:** The Energy-Charts interface may change over time. The link above provides access to the platform and an example configuration for Denmark (2025). Users may need to adjust the year and data selection to obtain the corresponding historical dataset.

### 2. Year-ahead electricity load forecast

Year-ahead total load forecasts are available through the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/generation/forecast/windAndSolar/onshore).

For each bidding zone and each week of the following year, the year-ahead forecast provides **minimum and maximum total load values**. Thus, the dataset does not provide an hourly load profile; rather, it provides the forecasted weekly load range (minimum and maximum load) for the bidding zone. This is consistent with the ENTSO-E definition of the year-ahead total load forecast.

The ENTSO-E Transparency Platform also provides data through its web interface, API, and downloadable file extracts.
