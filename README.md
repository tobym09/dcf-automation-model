# DCF Automated Valuation Tool

Automated stock valuation in Python. Enter a ticker, and the tool pulls live financials from Yahoo Finance, computes WACC via CAPM, models free cash flow growth with log-linear regression, and returns an intrinsic value, a percent over- or undervaluation, and a verdict, all with a table and graph.

   [![Open In Colab To Run](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/YOUR-NOTEBOOK.ipynb)

## Overview

A Discounted Cash Flow (DCF) model estimates what a company's stock is worth today based on its future cash flows. Building one by hand is slow: you have to gather financial data, build the model, and apply assumptions. This tool automates that process so any public company can be evaluated in seconds, while still letting the user control key assumptions.

The project started as a valuation of FWONK (Formula One Group) and was expanded into a general tool that works for any publicly traded company.

## Features

- **Ticker-based data retrieval:** automatically fetches price, shares outstanding, debt, beta, and free cash flow from Yahoo Finance
- **WACC via CAPM:** full cost-of-capital calculation using real beta, debt/equity weights, and tax-adjusted cost of debt
- **Log-linear regression growth model:** estimates FCF growth (CAGR) from multiple years of historical data using exponential trend fitting, with optional override
- **10-year FCF projection** with discounting of each year's cash flow
- **Gordon Growth terminal value**, discounted to present value
- **Valuation output:** enterprise value, equity value, intrinsic price per share, and % difference from market price
- **Verdict:** overvalued, undervalued, or well-valued
- **Report and visualization:** a formatted results table and a bar chart comparing current price to intrinsic value
- **Input validation:** handles missing prices, missing share counts, negative FCF, and invalid user input

## Inputs

When you run the program, it asks for:

| Input | Purpose |
|---|---|
| **Stock ticker** | The company to value (e.g., `FWONK`, `AAPL`). Re-prompts until a valid Yahoo Finance ticker is entered. |
| **Yearly projections (yes/no)** | Toggles whether the year-by-year projected cash flows are displayed. |
| **Effective tax rate** | Adjusts the cost of debt in WACC. A higher tax rate means a larger tax shield, a lower WACC, and a higher valuation. |

Each input is validated and re-prompted until a proper value is entered.

## Outputs

- A results table with the assumptions and calculated values (WACC, growth rate, enterprise value, equity value, intrinsic price)
- Intrinsic value vs. current market price, with the % over- or undervaluation
- A bar chart comparing current price to intrinsic value, with the verdict in the title

## How to Run

1. Open the notebook in Google Colab using the badge above, or upload it to Colab manually.
2. Install the dependencies (first cell):
   ```
   pip install yfinance tabulate numpy matplotlib
   ```
3. Click **Runtime → Run all**.
4. When prompted, enter a ticker symbol, your yearly-projection choice, and the effective tax rate.
5. Review the table and graph that are generated.

**Tip:** look up the ticker symbol before running so you don't have to guess.

## Code Structure

The code is organized into layers that separate data handling, financial modeling, and presentation.

| Layer | Function | Role |
|---|---|---|
| Helpers | `_find_row()` | Finds financial statement rows even when labels differ across companies |
| | `_valid()` | Checks that a value is usable in calculations |
| | `_fmt()` | Formats large numbers into readable strings |
| Data | `_fetch()` | Pulls company info, balance sheet, and cash flow data from Yahoo Finance |
| Modeling | `_cagr()` | Estimates growth using log-linear regression |
| | `_wacc()` | Computes WACC using CAPM and after-tax cost of debt |
| | `_dcf()` | Projects FCF, discounts cash flows, computes terminal value, and returns intrinsic price |
| Interface | `dcf_model()` | Main function: gathers input, runs the pipeline, and prints the report |
| Visualization | `plot_dcf()` | Bar chart of current price vs. intrinsic value |

## Technical Challenges

- **Non-standardized data:** Yahoo Finance labels the same metric differently across companies. `_find_row()` searches for a list of possible names so the correct data is always extracted.
- **Data quality:** missing prices, missing share counts, and negative FCF are caught by validation checks so the model fails gracefully.
- **Scalability:** switching from link-based input to a ticker-based API made the tool work for any public company.

## Limitations

- Results depend on the quality and availability of Yahoo Finance data.
- DCF valuations are sensitive to assumptions (growth rate, tax rate, discount rate) and are an estimate, not financial advice.
