"""
sensor_info.py - documented meanings of the NASA C-MAPSS sensor channels.

Source: NASA Ames Prognostics Data Repository, "Turbofan Engine Degradation
Simulation Data Set" (A. Saxena and K. Goebel, 2008). The symbol / description /
unit table below was cross-checked against several independent reproductions of
that documentation. It describes what each channel MEASURES. It says nothing
about which component is faulty, and AEGIS never infers faults from it.
Channels that are not in the table get no description (nothing is invented).
"""

SOURCE = "NASA C-MAPSS documentation (Saxena & Goebel, 2008)"

# sensor index -> (symbol, description, unit)
_TABLE = {
    1: ("T2", "Total temperature at fan inlet", "°R"),
    2: ("T24", "Total temperature at LPC outlet", "°R"),
    3: ("T30", "Total temperature at HPC outlet", "°R"),
    4: ("T50", "Total temperature at LPT outlet", "°R"),
    5: ("P2", "Pressure at fan inlet", "psia"),
    6: ("P15", "Total pressure in bypass duct", "psia"),
    7: ("P30", "Total pressure at HPC outlet", "psia"),
    8: ("Nf", "Physical fan speed", "rpm"),
    9: ("Nc", "Physical core speed", "rpm"),
    10: ("epr", "Engine pressure ratio (P50/P2)", "-"),
    11: ("Ps30", "Static pressure at HPC outlet", "psia"),
    12: ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    13: ("NRf", "Corrected fan speed", "rpm"),
    14: ("NRc", "Corrected core speed", "rpm"),
    15: ("BPR", "Bypass ratio", "-"),
    16: ("farB", "Burner fuel-air ratio", "-"),
    17: ("htBleed", "Bleed enthalpy", "-"),
    18: ("Nf_dmd", "Demanded fan speed", "rpm"),
    19: ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    20: ("W31", "HPT coolant bleed", "lbm/s"),
    21: ("W32", "LPT coolant bleed", "lbm/s"),
}


def _index(sensor):
    try:
        prefix, num = str(sensor).rsplit("_", 1)
        return int(num) if prefix == "sensor" else None
    except ValueError:
        return None


def sensor_symbol(sensor):
    """Short documented symbol, e.g. 'Ps30', or None if unknown."""
    row = _TABLE.get(_index(sensor))
    return row[0] if row else None


def sensor_description(sensor):
    """Documented description, e.g. 'Static pressure at HPC outlet', or None."""
    row = _TABLE.get(_index(sensor))
    return row[1] if row else None


def sensor_label(sensor):
    """'sensor_11 (Ps30)' - or just 'sensor_11' when undocumented."""
    sym = sensor_symbol(sensor)
    return f"{sensor} ({sym})" if sym else str(sensor)


def sensor_long_label(sensor):
    """'sensor_11 (Ps30: Static pressure at HPC outlet)' or 'sensor_11'."""
    row = _TABLE.get(_index(sensor))
    return f"{sensor} ({row[0]}: {row[1]})" if row else str(sensor)
