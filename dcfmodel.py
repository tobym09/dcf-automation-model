"""
================================================================================
= TITLE: Automated DCF (Discounted Cash Flow) Model
=
= FUNCTION:
= Runs an automated DCF model to remove the time it takes to write the equations
= for each company you want to model. Now, you only need to provide the company
= ticker and your assumptions, and the model with provide you with the data and
= its results. This saves hours of time and energy in spreadsheets, calculating
= equations and fetching data
=
= INPUT:
= Call dcf_model() directly with keyword arguments. All parameters have
= defaults so only `ticker` is required.
=
= Usage (in another script or notebook):
=   from dcf_model import dcf_model
=
=   # Input bare minimum example
=   dcf_model(ticker="NVDA")
=
=   # Input custom assumptions example
=   dcf_model(ticker="AAPL", rfr=0.05, mrp=0.06, terminal_gr=0.03)
=
=   # Full override example
=   dcf_model(
=       ticker           = "TSLA", (str)
=       rfr              = 0.045, (float)
=       mrp              = 0.055, (float)
=       implied_spread   = 0.0161, (float) -> Default is BB-rated spread
=       effective_tax    = 0.261, (float)
=       terminal_gr      = 0.025, (float)
=       forecast_horizon = 10, (int)
=       historical_fallback = 0.05, (float)
=       growth_cap_min   = -0.10, (float)
=       growth_cap_max   =  0.15, (float)
=       beta_fallback    = 1.0, (float)
=       show_yearly      = True, (bool)
=       growth_override  = 0.12, (float) -> Manually set FCF growth rate
=   )
=
= OUTPUT:
=   A heading including the company name and ticker, its current share price,
=   beta, WACC, and FCF growth. It also includes an organized dictionary
=   for the above keys, the assumptions made, the fcf for the
=   given forecast horizon, and the DCF summary. The output also includes the
=   evaluation whether the stock is under or over valued and by what percent
=   (as a string). Lastly, this code produces a graph to visually represent
=   the difference between the current and intrinsic values
=
= TERMS AND DEFINITIONS:
=   rfr: Risk-free rate
=   mrp: Market risk premium
=   wacc: Weighted average cost of capital
=   cagr: Compound annual growth rate
=   capm: Capital asset pricing model
=   beta: Coefficient of variation
=   implied_spread: Credit spread above the risk-free rate based on credit rating
=   cod: Pre-tax cost of debt (rfr + implied_spread)
=   after_tax_cod: After-tax cost of debt (cod * (1 - effective_tax))
=   effective_tax: Effective tax rate applied to cost of debt
=   market cap: Total market value of equity
=   market total: sum of net debt and market cap
=   debt_weight: Debt share of capital
=   equity_weight: Equity share of capital
=   terminal_gr: Perpetuity growth rate
=   tax_rate: Corporate tax rate
=   fcf: Free Cash Flow
=   growth: FCF growth rate
=   growth_override: Optional manually set FCF growth rate that bypasses CAGR
=   Terminal value: Value of company's projected fcf at terminal growth rate
=   Intrinsic value: Terminal value discounted to the present
=   Enterprise value: Measure of company's total value
=   Equity value: Enterprise value minus net debt given to shareholders
=   Shares outstanding: Number of shares
=
= NOTE:
=  in order for this code to work, you need to install the following packages
=  (in the terminal if you are in a script, or the previous cell if you are
=  in a notebook):
=     pip install yfinance tabulate numpy or pip3 install yfinance tabulate numpy
================================================================================
"""
# This section of code raises errors if you forget to install these packages
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Run:  pip install yfinance")

try:
    from tabulate import tabulate
except ImportError:
    raise ImportError("Run:  pip install tabulate")

try:
    import numpy as np
except ImportError:
    raise ImportError("Run:  pip install numpy")


# ───────────────────────────── INTERNAL HELPERS ─────────────────────────────

def _find_row(df, candidates):
  '''
  This function looks for rows in yahoo finance that contain financial data. The
  canidates are headings you would find in yahoo fianance such as "Operating
  cash flow." It also uses c.lower() so it is not case sensitive. This function
  returns the index label that contains a canidate, and if there is no row that
  matches the canidates, it returns None.
  '''
  for label in df.index:
      for c in candidates:
          if c.lower() in str(label).lower():
              return label
  return None


def _valid(v):
  '''
  This function takes in a given value v, and determines whether you can perform
  mathematical operations on it. It returns True if it can, and False if it
  cannot (because v is not a float)
  '''
  try:
      return v is not None and float(v) == float(v)
  except (TypeError, ValueError):
      return False


def _fmt(n: float) -> str:
  '''
  This function helps the appearance of our data. It removes the decimals in
  floats and replaces them with T for trillion, B for billion, or M for million.
  This way we dont have huge numbers taking over the screen, and it makes
  numbers look very clean.
  '''
  if abs(n) >= 1e12:
      return f"${n/1e12:.2f}T"
  if abs(n) >= 1e9:
      return f"${n/1e9:.2f}B"
  if abs(n) >= 1e6:
      return f"${n/1e6:.2f}M"
  return f"${n:,.0f}"

# ─────────────────────────────── DATA FETCHING ───────────────────────────────

def _fetch(ticker: str) -> dict:
  '''
  This large function retrieves the raw data from yahoo finance (which the
  library yfinance allows us to do) for a given stock ticker that the user
  will define when they call the main function. The main function then calls
  this function to search for the data for the DCF. The data in this function
  is formatted as a dictionary, hence -> dict. Here is how the function works:
    1) The yahoo finance ticker corresponding to the input ticker (which is
  made uppercase in the main function) is grabbed.
    2) General company (shares, debt, cash, price, etc.) and historical cash
  flow statements are then saved into respective variables for that given
  ticker.
    3) Free cash flow is calculated by iterating through the historical cash
  flow and summing the operating cash flow and capital expenditures for teach
  time period. It works because our helper functions _find_row retrieves the
  data we want, and _valid checks if we can sum the data. It also makes sure
  there is cash flow to look at, and finally each time period of data is
  added to the fcf_history list.
    4) It also fetches total_debt and market_cap directly so that
  debt_weight and equity_weight can be calculated from actual market data
  rather than being passed as a fixed parameter.
    5) Individual keys representing the general data are created by either
  extracting this data from the group of info taken from yahoo finance and put
  into the variabe 'info', or calculated.

  INPUT: Stock ticker (string)

  OUTPUT: A dictionary containing the ticker, stock name, its currency, current
  share price, diluted shares outstanding, calculated net debt, beta, FCF history,
  total_debt (float), and market_cap (float).
  '''
  tk       = yf.Ticker(ticker)
  info     = tk.info
  cashflow = tk.cashflow

  fcf_history = []
  if cashflow is not None and not cashflow.empty:
      ocf_row   = _find_row(cashflow, ["Operating Cash Flow",
                                        "Total Cash From Operating Activities"])
      capex_row = _find_row(cashflow, ["Capital Expenditure",
                                        "Capital Expenditures"])
      if ocf_row and capex_row:
          for col in cashflow.columns:
              ocf   = cashflow.loc[ocf_row,   col]
              capex = cashflow.loc[capex_row, col]
              if _valid(ocf) and _valid(capex):
                  fcf_history.append(float(ocf) + float(capex))

  shares = (
      info.get("impliedSharesOutstanding")
      or info.get("sharesOutstanding")
      or info.get("floatShares")
  )

  total_debt = info.get("totalDebt", 0) or 0
  cash       = info.get("totalCash", 0) or 0

  price = (
      info.get("currentPrice")
      or info.get("regularMarketPrice")
      or info.get("previousClose")
  )

  market_cap = info.get("marketCap", 0) or 0

  return {
      "ticker":        ticker.upper(),
      "name":          info.get("longName", ticker.upper()),
      "currency":      info.get("financialCurrency", "USD"),
      "current_price": price,
      "shares":        shares,
      "net_debt":      total_debt - cash,
      "beta":          info.get("beta"),
      "fcf_history":   fcf_history,
      "total_debt":    total_debt,
      "market_cap":    market_cap,
  }

# ──────────────────────────────────DCF MATH──────────────────────────────────

def _cagr(fcf_history, fallback, cap_min, cap_max):
  '''
  This function calculates how fast the fcf has been growing for the given
  stock. Rather than using only the two endpoints (which is highly sensitive
  to outlier years), it fits a log-linear regression through all available
  positive FCF data points. This uses every data point equally, so one
  unusually high or low year does not distort the result. The slope of the
  regression line represents the continuous growth rate, which is then
  converted to an annual rate. If there are fewer than 2 valid data points,
  the fallback rate is returned. With exactly 2 points, it falls back to the
  standard endpoint CAGR since regression requires at least 3 points to be
  more meaningful than a direct calculation. The result is always clamped
  between cap_min and cap_max.

  INPUT: Free cash flow history (list, most-recent first), fallback (float),
  growth cap min (float), growth cap max (float)

  OUTPUT: The regression-based annual FCF growth rate (float), clamped between
  cap_min and cap_max. Returns fallback if there is insufficient data.
  '''
  # Keep only positive, valid FCF values (negative FCF breaks logarithms)
  clean = [f for f in fcf_history if f and f > 0]

  # Not enough data — return the fallback rate
  if len(clean) < 2:
      return fallback

  # Only 2 points — regression adds no benefit over direct CAGR
  if len(clean) == 2:
      cagr = (clean[0] / clean[1]) ** (1 / 1) - 1
      return max(cap_min, min(cagr, cap_max))

  # 3+ points — fit a log-linear regression through all data points.
  # clean[0] is the most recent year, so reverse to get chronological order
  # (oldest to newest) before fitting.
  chronological = list(reversed(clean))
  log_fcf = np.log(chronological)          # log-linearises exponential growth
  years   = np.arange(len(log_fcf))

  # np.polyfit returns [slope, intercept] for a degree-1 (linear) fit.
  # The slope of log(FCF) vs time equals the continuous growth rate.
  slope, _ = np.polyfit(years, log_fcf, 1)

  # Convert continuous rate back to a discrete annual rate
  cagr = np.exp(slope) - 1

  return max(cap_min, min(cagr, cap_max))

def _wacc(beta, rfr, mrp, debt_weight, equity_weight, after_tax_cod):
  '''
  This function calculates the weighted average cost of capital (WACC) for a
  given stock. It uses known equations for the equity weight and cost of
  equity to calculate the WACC. It calculares after_tax_cod (pre-tax cost of
  debt already multiplied by (1 - effective_tax)) so the tax shield is applied
  before this function is called. debt_weight and equity_weight are  passed in
  as calculated values derived from actual market data (market_cap and total_debt)
  as opposed to a fixed user assumption.This function is called within the main
  function and is not manually called.

  INPUT: Beta (float), rfr (float), mrp (float), debt_weight (float),
  equity_weight (float), after_tax_cod (float)

  OUTPUT: wacc (float) and cost_of_equity (float)
  '''
  cost_of_equity = rfr + beta * mrp
  wacc = round(equity_weight * cost_of_equity + debt_weight * after_tax_cod, 4)
  return wacc, cost_of_equity

def _dcf(base_fcf, growth, wacc, terminal_gr, years, shares, net_debt):
  '''
  This function takes in all of the key metrics and preforms the DCF calculation.
    1) It takes the forecast horizon inputed (represented as years) and projects
  the fcf for that many years by the growth rate each year.
    2) The function then discounts each fcf it just projected back to the
  current year using the WACC
    3) It then calculates the terminal value which is the value of all cash
  flows beyond the forecast horizon and discounts this too.
    4) Calculates the enterprise value from the discounted sum of projected fcfs
  and the terminal value.
    5) Calculates the equity value and price per share using this data.
  This function is called within the main function and is not manually called.

  INPUT: Base FCF (float), growth rate (float), WACC (float), terminal growth
  rate (float), forecast horizon (int), shares outstanding (int), net debt
  (float)

  OUTPUT: A dictionary containing the base fcf, projected fcf, pv of projected
  fcf, terminal value, pv of terminal value, enterprise value, equity value, and
  intrinsic value
  '''
  projected = []
  fcf = base_fcf
  for _ in range(years):
      fcf *= (1 + growth)
      projected.append(fcf)

  pv_fcf = sum(cf / (1 + wacc) ** t for t, cf in enumerate(projected, 1))

  if wacc <= terminal_gr:
      raise ValueError(f"WACC ({wacc:.3f}) must exceed terminal growth rate ({terminal_gr:.3f}).")

  tv    = projected[-1] * (1 + terminal_gr) / (wacc - terminal_gr)
  pv_tv = tv / (1 + wacc) ** years

  enterprise = pv_fcf + pv_tv
  equity     = enterprise - net_debt
  per_share  = equity / shares

  return {
      "base_fcf":         base_fcf,
      "projected_fcf":    projected,
      "pv_fcf":           pv_fcf,
      "terminal_value":   tv,
      "pv_terminal":      pv_tv,
      "enterprise_value": enterprise,
      "equity_value":     equity,
      "intrinsic_value":  per_share,
  }

# ─────────────────────────── MAIN FUNCTION CALLED ───────────────────────────

def dcf_model(
    ticker:              str,
    rfr:                 float = 0.041,
    mrp:                 float = 0.05,
    implied_spread:      float = 0.0161,
    effective_tax:       float = 0.261,
    terminal_gr:         float = 0.025,
    forecast_horizon:    int   = 10,
    historical_fallback: float = 0.05,
    growth_cap_min:      float = -0.10,
    growth_cap_max:      float =  0.15,
    beta_fallback:       float = 1.0,
    show_yearly:         bool  = False,
    growth_override:     float = None,
) -> dict:
    """
    This is the function that you call, which calls all other functions in this
    script. This function runs a full DCF valuation for any publicly traded
    company when you give it a stock ticker. It allows you to cutomize any
    assumption you want, outputs these assumptions along with the DCF summary
    so you can see how changing one assumption affects the model. Each part of
    this function will be annotated under its section below.

    INPUT PARAMETERS as a dictionary. Given below are the keys with associated
    default values:
    ----------
    ticker              : Stock ticker symbol, Ie. "META"
    rfr                 : Risk-free rate                       (default 4.1%)
    mrp                 : Market risk premium                  (default 5.0%)
    implied_spread      : Credit spread above rfr              (default 1.61% — BB-rated)
    effective_tax       : Effective tax rate on cost of debt   (default 26.1%)
    terminal_gr         : Perpetuity growth rate               (default 3%)
    forecast_horizon    : Forecast years                       (default 10)
    historical_fallback : FCF growth if CAGR fails             (default 5.0%)
    growth_cap_min      : Min cap on CAGR                      (default -10%)
    growth_cap_max      : Max cap on CAGR                      (default +15%)
    beta_fallback       : Beta if not in Yahoo data            (default 1.0)
    show_yearly         : Print year-by-year table             (default False)
    growth_override     : Manually set FCF growth rate         (default None — uses historical CAGR if not set)

    OUTPUT: A dictionary containing keys: ticker (string), name (string),
    current_price (float), intrinsic_value (float), pct_difference (float),
    verdict (string), wacc (float), cost_of_equity (float), growth_rate (float),
    dcf_detail (dictionary of the DCF summary).

    """

    ticker = ticker.upper()

    # ────────────────────────── 1. Fetch data ───────────────────────────────
    '''
    This part of the code collects the data from yahoo fincance by calling the
    helper function _fetch() written above. Then, it checks to make sure there
    is sufficient data for the model to actually run. If not, it raises an error.
    Lastly, if there is data inside the fcf_history list, it assigns the most
    recent value to base_fcf, otherwise it raises an error if the value < 0. It
    also unpacks total_debt and market_cap from the fetched data
    to calculate debt_weight and equity_weight from real market values.
    '''
    data = _fetch(ticker)

    if data["current_price"] == None:
        raise ValueError(f"Could not retrieve a market price for {ticker}.")
    if data["shares"] == None or data["shares"] <= 0:
        raise ValueError(f"Could not retrieve shares outstanding for {ticker}.")
    if data["fcf_history"] == None:
        raise ValueError(f"No Free Cash Flow history found for {ticker}.")

    base_fcf = data["fcf_history"][0]
    if base_fcf <= 0:
        raise ValueError(
            f"Most-recent FCF is ${base_fcf:,.0f} (zero or negative)."
            "DCF requires positive FCF."
        )

    # ──────────────────────── 2. Resolve beta & WACC ─────────────────────────
    '''
    This part just sets the beta to what was pulled from the _fetch() function
    and sets the beta to the fallback if there is no beta data. COD is derived
    here as rfr + implied_spread (pre-tax cost of debt). The after-tax cost of debt
    is then calculated as cod * (1 - effective_tax). This after_tax_cod value
    is what gets passed into _wacc() and printed in the report. Debt_weight and
    equity_weight are calculated from actual market data: total_debt
    (book value of debt from Yahoo Finance) and market_cap (current market value
    of equity from Yahoo Finance). Their sum forms total_capital. If
    total_capital is zero or unavailable, the model falls back to equal 50/50
    weights.
    '''
    if data["beta"] and _valid(data["beta"]):
      beta = data["beta"]
    else:
      beta = beta_fallback

    # pre-tax cost of debt
    cod = rfr + implied_spread
    # after-tax cost of debt
    after_tax_cod = cod * (1 - effective_tax)

    total_debt    = data["total_debt"]
    market_cap    = data["market_cap"]
    total_capital = total_debt + market_cap

    if total_capital > 0:
        debt_weight   = total_debt  / total_capital
        equity_weight = market_cap  / total_capital
    else:
        debt_weight   = 0.5
        equity_weight = 0.5

    wacc, cost_of_equity = _wacc(beta, rfr, mrp, debt_weight, equity_weight, after_tax_cod)

    # ──────────────────────── 3. Resolve FCF growth ──────────────────────────
    '''
    This part sets the local growth variable to the result of the _cagr()
    function given the historical data. It also prints this data to the console
    as there are no more calculations to be done. If growth_override is provided
    by the user, it is used directly and the CAGR calculation is skipped entirely.
    A boolean flag (growth_was_overridden) is set so the report can display
    "(Overridden)" next to the growth rate wherever it appears — in both the
    status line and the assumptions table.
    '''
    if growth_override is not None:
        growth_was_overridden = True
    else:
        growth_was_overridden = False

    if growth_was_overridden:
        growth = growth_override
    else:
      growth = _cagr(data["fcf_history"], historical_fallback,
                       growth_cap_min, growth_cap_max)

    if growth_was_overridden:
        growth_label = f"{growth*100:.1f}% (Overridden)"
    else:
        growth_label = f"{growth*100:.1f}%"

    print(f"\n{data['name']}  |  Price: {data['currency']} {data['current_price']:.2f}  "
          f"|  Beta: {beta:.2f}  |  WACC: {wacc*100:.1f}%  "
          f"|  FCF Growth: {growth_label}")

    # ──────────────────────────── 4. Run DCF ──────────────────────────────────
    '''
    This is the part of the function that determines whether the stock provided is
    under or over valued. It does this by running the dcf model given all the
    inputs it calculated from above and the assumptions/parameters it was given.
    It assigns the value of the intrinsic value to iv, and compares it to the
    current stock price.
    '''
    dcf = _dcf(
        base_fcf    = base_fcf,
        growth      = growth,
        wacc        = wacc,
        terminal_gr = terminal_gr,
        years       = forecast_horizon,
        shares      = data["shares"],
        net_debt    = data["net_debt"],
    )

    iv       = dcf["intrinsic_value"]
    price    = data["current_price"]
    pct_diff = (iv - price) / price * 100

    if pct_diff > 0:
        verdict = "UNDERVALUED"
    else:
        verdict = "OVERVALUED"

    # ── 5. Print report ──────────────────────────────────────────────────
    '''
    Finally, the data has been calculated. This section formats the data and
    prints it to the console for a user friendly visualization of the data
    in a table format. Tabulate uses pretty printing for dictionaries so the
    returned data is organized and readable. It is called when printing the
    assumptions, yearly projections, and summary data. If the user decides to
    include the parameter show_yearly = True, then a table of the projected
    yearly values will also print. Lastly, the verdict is printed.
    '''
    currency = data["currency"]

    print("\n" + "═" * 60)
    print(f"  DCF VALUATION REPORT  —  {data['name']} ({ticker})")
    print("═" * 60)

    print("\nASSUMPTIONS AND INPUTS")
    print(tabulate([
        ["Risk-Free Rate",                    f"{rfr*100:.2f}%"],
        ["Market Risk Premium",               f"{mrp*100:.2f}%"],
        ["Implied Credit Spread (BB default)",f"{implied_spread*100:.2f}%"],
        ["Pre-Tax Cost of Debt",              f"{cod*100:.2f}%"],
        ["Effective Tax Rate",                f"{effective_tax*100:.0f}%"],
        ["After-Tax Cost of Debt",            f"{after_tax_cod*100:.2f}%"],
        ["Cost of Equity",                    f"{cost_of_equity*100:.2f}%"],
        ["Debt Weight",                       f"{debt_weight*100:.1f}%"],
        ["Equity Weight",                     f"{equity_weight*100:.1f}%"],
        ["WACC",                              f"{wacc*100:.2f}%"],
        ["FCF Growth Rate",                   growth_label],
        ["Terminal Growth Rate",              f"{terminal_gr*100:.2f}%"],
        ["Forecast Horizon",                  f"{forecast_horizon} years"],
    ], tablefmt="simple"))

    if show_yearly:
        print("\nPROJECTED FREE CASH FLOWS")
        print(tabulate(
            [[f"Year {i+1}", _fmt(v)] for i, v in enumerate(dcf["projected_fcf"])],
            headers=["Year", "FCF"],
            tablefmt="simple"
        ))

    print("\nDCF SUMMARY")
    print(tabulate([
        ["Base FCF (most recent)",  _fmt(dcf["base_fcf"])],
        ["PV of Projected FCFs",   _fmt(dcf["pv_fcf"])],
        ["Terminal Value",         _fmt(dcf["terminal_value"])],
        ["PV of Terminal Value",   _fmt(dcf["pv_terminal"])],
        ["Enterprise Value",       _fmt(dcf["enterprise_value"])],
        ["Net Debt",               _fmt(data["net_debt"])],
        ["Equity Value",           _fmt(dcf["equity_value"])],
        ["Diluted Shares Outstanding", f"{data['shares']/1e9:.3f}B"],
    ], tablefmt="simple"))

    print("\n" + "─" * 60)
    print(f"  Current Market Price : {currency} {price:.2f}")
    print(f"  Intrinsic Value (DCF): {currency} {iv:.2f}")
    print(f"  Difference           : {pct_diff:+.1f}%")
    print(f"\n  ▶  {ticker} appears to be {verdict}")
    print(f"     by approximately {abs(pct_diff):.1f}%")
    print("─" * 60)

    return {
        "ticker":          ticker,
        "name":            data["name"],
        "current_price":   price,
        "intrinsic_value": iv,
        "pct_difference":  pct_diff,
        "verdict":         verdict,
        "wacc":            wacc,
        "cost_of_equity":  cost_of_equity,
        "growth_rate":     growth,
        "dcf_detail":      dcf,
    }

# ─────────────────────────── CHART OUTPUT ───────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_dcf(result: dict):
    '''
    This function takes the dictionary returned by dcf_model() and produces a
    bar chart comparing the current market price to the DCF intrinsic value. It
    does this by setting variables to the respective varibales outputted
    in the dictionary. The current price bar is blue and the intrinsic value
    bar is orange. A horizontal black reference line extends from the top of each
    bar to the y-axis, where the price is labeled. The region between the two
    reference lines on the current price bar is shaded gray to represent the
    difference between the two values.

    INPUT: the dictionary returned by the dcf_model() fucntion (dict)

    OUTPUT: Displays a matplotlib bar chart.
    '''
    price    = result["current_price"]
    iv       = result["intrinsic_value"]
    ticker   = result["ticker"]
    verdict  = result["verdict"]
    pct_diff = result["pct_difference"]

    fig, ax = plt.subplots(figsize=(7, 6))

    bars = ax.bar(
        ["Current Price", "Intrinsic Value"],
        [price, iv],
        color=["Blue", "Orange"],
        width=0.45,
        zorder=3
    )

    y_min = min(price, iv)
    y_max = max(price, iv)

    # Gray shading between the two price levels on the current price bar only
    shade_bottom = min(price, iv)
    shade_top    = max(price, iv)
    ax.bar(["Current Price"], [shade_top - shade_bottom], bottom=shade_bottom,
           color="lightgray", alpha=0.7, width=0.45, zorder=4)

    # Makes the reference lines from each bar top to the side
    ax.hlines(price, xmin=-0.5, xmax=0,  colors="black", linewidths=1.2, zorder=4)
    ax.hlines(iv,    xmin=-0.5, xmax=1,  colors="black", linewidths=1.2, zorder=4)

    # Makes the price labels on the y-axis
    ax.set_yticks([price, iv])
    ax.set_yticklabels([f"${price:.2f}", f"${iv:.2f}"], fontsize=11, fontweight="bold")

    ax.set_xlim(-0.5, 1.5)
    ax.set_title(
        f"{ticker}  —  DCF Valuation\n"
        f"{verdict} by {abs(pct_diff):.1f}%",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False) # gets rid of top and right lines

    plt.show()
