import os
import html
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
import joblib

from agents import run_agent_pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AEGIS AI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(0,210,230,.09), transparent 30%),
        radial-gradient(circle at 5% 70%, rgba(0,120,150,.07), transparent 28%),
        #04080c;
    color: #e7f3f5;
}

.block-container {
    max-width: 1520px;
    padding-top: 45px;
    padding-bottom: 70px;
}

section[data-testid="stSidebar"] {
    background: #060b10;
    border-right: 1px solid #18272e;
}

section[data-testid="stSidebar"] * {
    color: #dce9ec;
}

.main-title {
    font-size: 2.45rem;
    font-weight: 800;
    letter-spacing: -.055em;
}

.main-subtitle {
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: -1px;
}

.subtitle {
    color: #718990;
    font-size: .78rem;
    margin-top: 7px;
    margin-bottom: 25px;
}

.section-title {
    margin-top: 30px;
    margin-bottom: 12px;
    color: #71939b;
    font-family: 'JetBrains Mono', monospace;
    font-size: .67rem;
    font-weight: 600;
    letter-spacing: .16em;
}

.section-title-large {
    margin-top: 34px;
    margin-bottom: 14px;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: .02em;
}

.intelligence-header {
    position: relative;
    overflow: hidden;
    padding: 22px 26px;
    border-radius: 18px 18px 0 0;
    border: 1px solid #21424b;
    border-bottom: none;
    background:
        radial-gradient(circle at 80% 50%, rgba(0,220,235,.12), transparent 35%),
        linear-gradient(135deg,#0c1b22,#071015);
}

.intelligence-header:after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 1px;
    background: linear-gradient(90deg,transparent,#35cbd8,transparent);
}

.intelligence-kicker {
    color: #3ed0dd;
    font-family: 'JetBrains Mono', monospace;
    font-size: .59rem;
    letter-spacing: .16em;
}

.intelligence-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin-top: 5px;
}

.intelligence-desc {
    color: #66818a;
    font-size: .68rem;
    margin-top: 4px;
}

.metric {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg,#0c181f,#070d12);
    border: 1px solid #1b343c;
    border-radius: 15px;
    padding: 18px;
    min-height: 125px;
}

.metric:before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 3px;
    height: 100%;
    background: #28c5d3;
    opacity: .75;
}

.metric-label {
    color: #68838b;
    font-size: .61rem;
    letter-spacing: .11em;
    text-transform: uppercase;
}

.metric-value {
    color: #eaf9fb;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.72rem;
    font-weight: 600;
    margin-top: 10px;
}

.metric-small {
    color: #5e767e;
    font-size: .63rem;
    margin-top: 6px;
}

.health-card {
    background:
        radial-gradient(circle at 50% 0%,rgba(50,220,225,.10),transparent 60%),
        #081217;
    border: 1px solid #21414a;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
}

.health-label {
    color: #66818a;
    font-family: 'JetBrains Mono', monospace;
    font-size: .59rem;
    letter-spacing: .13em;
}

.health-value {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 8px;
}

.health-engine {
    color: #5d767e;
    font-family: 'JetBrains Mono', monospace;
    font-size: .61rem;
    margin-top: 5px;
}

.health-line {
    height: 1px;
    background: #1a3037;
    margin: 16px 0 13px;
}

.health-item {
    color: #68828a;
    font-family: 'JetBrains Mono', monospace;
    font-size: .58rem;
    line-height: 2;
}

.factor-card {
    position: relative;
    min-height: 142px;
    padding: 17px;
    border-radius: 15px;
    border: 1px solid #1b333b;
    background: linear-gradient(145deg,#0b161c,#070d12);
}

.factor-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.factor-name {
    color: #a5bcc1;
    font-size: .73rem;
    font-weight: 600;
}

.factor-icon {
    font-size: 1rem;
}

.factor-value {
    color: #e6f7f9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    margin-top: 14px;
}

.factor-bar {
    height: 4px;
    margin-top: 12px;
    border-radius: 10px;
    background: #17272e;
    overflow: hidden;
}

.factor-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg,#176d79,#38d0dd);
}

.factor-caption {
    color: #566f77;
    font-size: .57rem;
    margin-top: 8px;
}

.chart-panel {
    background: linear-gradient(145deg,#0a141a,#060c11);
    border: 1px solid #192e36;
    border-radius: 16px;
    padding: 10px 12px 3px;
}

.chart-title {
    color: #91aab0;
    font-size: .72rem;
    font-weight: 600;
    padding: 8px 8px 2px;
}

.chart-caption {
    color: #536c74;
    font-family: 'JetBrains Mono', monospace;
    font-size: .56rem;
    padding: 0 8px 5px;
}

.agent-network {
    position: relative;
    overflow: hidden;
    border: 1px solid #1e3b44;
    border-radius: 20px;
    padding: 26px;
    background:
        radial-gradient(circle at 50% 50%,rgba(0,200,220,.07),transparent 45%),
        linear-gradient(145deg,#0a151b,#050b0f);
}

.agent-network-title {
    font-size: 1rem;
    font-weight: 700;
}

.agent-network-subtitle {
    color: #5e777f;
    font-size: .65rem;
    margin-top: 4px;
}

.agent-box {
    min-height: 205px;
    padding: 19px;
    border: 1px solid #20404a;
    border-radius: 15px;
    background: linear-gradient(145deg,#0c181f,#071015);
}

.agent-icon {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    background: rgba(0,190,210,.08);
    border: 1px solid #24515c;
    font-size: 1.2rem;
}

.agent-name {
    font-size: .86rem;
    font-weight: 600;
    margin-top: 12px;
}

.agent-role {
    color: #5d7b83;
    font-family: 'JetBrains Mono', monospace;
    font-size: .55rem;
    letter-spacing: .08em;
    margin-top: 4px;
}

.agent-text {
    color: #91a8ae;
    font-size: .68rem;
    line-height: 1.55;
    margin-top: 13px;
}

.agent-evidence {
    margin-top: 14px;
    padding: 9px 10px;
    border-radius: 9px;
    background: #081218;
    border-left: 2px solid #2bc4d2;
    color: #719099;
    font-family: 'JetBrains Mono', monospace;
    font-size: .57rem;
    line-height: 1.65;
}

.agent-arrow {
    text-align: center;
    color: #2d7882;
    font-size: 1.25rem;
    padding-top: 82px;
}

.final-output {
    position: relative;
    overflow: hidden;
    margin-top: 5px;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #28616c;
    background:
        radial-gradient(circle at 15% 50%,rgba(0,220,235,.14),transparent 32%),
        radial-gradient(circle at 85% 20%,rgba(0,160,190,.08),transparent 35%),
        linear-gradient(135deg,#0b1a21,#071015);
    box-shadow: 0 0 45px rgba(0,180,200,.07);
}

.final-output:before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: #31ccd9;
    box-shadow: 0 0 20px #31ccd9;
}

.final-kicker {
    color: #35cedb;
    font-family: 'JetBrains Mono', monospace;
    font-size: .62rem;
    letter-spacing: .16em;
}

.final-title {
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 6px;
}

.final-action {
    color: #e8fbfd;
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 18px;
    padding: 18px 20px;
    border-radius: 13px;
    background: rgba(0,0,0,.20);
    border: 1px solid #244951;
}

.final-meta {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.final-pill {
    padding: 8px 13px;
    border-radius: 8px;
    border: 1px solid #21434c;
    background: #081218;
    color: #7f9ca3;
    font-family: 'JetBrains Mono', monospace;
    font-size: .58rem;
}

.final-pill strong {
    color: #b5d8dc;
}

.footer {
    text-align: center;
    color: #3d555d;
    font-family: 'JetBrains Mono', monospace;
    font-size: .56rem;
    margin-top: 45px;
}

[data-testid="stMetric"] {
    background: transparent;
}

div[data-testid="stCaptionContainer"] {
    color: #607780;
}

</style>
""")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "lstm_rul_model.pth"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "scaler.pkl"
)

SENSORS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "useful_sensors.txt"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "CMAPSSData",
    "test_FD001.txt"
)


# ============================================================
# MODEL
# EXACT SAME ARCHITECTURE AS TRAINED MODEL
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last = output[:, -1, :]

        return self.fc(last).squeeze(1)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_everything():

    with open(SENSORS_PATH, "r") as f:

        sensors = [
            line.strip()
            for line in f
            if line.strip()
        ]

    model = LSTMModel(len(sensors))

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    try:

        scaler = joblib.load(SCALER_PATH)

    except Exception:

        from sklearn.preprocessing import MinMaxScaler

        train_path = os.path.join(
            BASE_DIR,
            "data",
            "CMAPSSData",
            "train_FD001.txt"
        )

        columns = [
            "unit",
            "cycle",
            "setting_1",
            "setting_2",
            "setting_3"
        ]

        columns += [
            f"sensor_{i}"
            for i in range(1, 22)
        ]

        train_df = pd.read_csv(
            train_path,
            sep=r"\s+",
            header=None,
            names=columns
        )

        scaler = MinMaxScaler()

        scaler.fit(train_df[sensors])

        try:
            joblib.dump(
                scaler,
                SCALER_PATH
            )
        except Exception:
            pass

    columns = [
        "unit",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3"
    ]

    columns += [
        f"sensor_{i}"
        for i in range(1, 22)
    ]

    test_df = pd.read_csv(
        TEST_PATH,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return model, scaler, sensors, test_df


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    MODEL_PATH,
    SENSORS_PATH,
    TEST_PATH
]

missing = [
    path
    for path in required_files
    if not os.path.exists(path)
]

if missing:

    st.error("Project files could not be loaded.")

    st.write("Missing files:")

    for path in missing:
        st.code(path)

    st.stop()


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model, scaler, sensors, test_df = load_everything()

except Exception as e:

    st.error("Project files could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.html("""
<div style="
    font-size:1.2rem;
    font-weight:700;
    margin-bottom:3px;
">
⚙️ AEGIS AI
</div>

<div style="
    color:#607982;
    font-size:.58rem;
    letter-spacing:.12em;
    margin-bottom:25px;
">
INTELLIGENT ASSET MONITORING
</div>
""")

st.sidebar.markdown("### MACHINE")

engine_ids = sorted(
    test_df["unit"].unique()
)

selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids
)

st.sidebar.markdown("---")

st.sidebar.html("""
<div style="
    color:#5d747d;
    font-size:.64rem;
    line-height:1.9;
">
DATASET<br>
NASA C-MAPSS · FD001<br><br>

MODEL<br>
2-Layer LSTM<br><br>

WINDOW<br>
30 cycles<br><br>

INPUT CHANNELS<br>
17 useful sensors
</div>
""")


# ============================================================
# ENGINE DATA
# ============================================================

engine = (
    test_df[
        test_df["unit"] == selected_engine
    ]
    .sort_values("cycle")
)

current_cycle = int(
    engine["cycle"].max()
)

recent = engine.tail(30)

raw_values = recent[sensors].values

scaled = scaler.transform(raw_values)

input_tensor = torch.tensor(
    scaled,
    dtype=torch.float32
).unsqueeze(0)


# ============================================================
# MC DROPOUT
# ============================================================

def mc_prediction(
    model,
    x,
    passes=50
):

    model.eval()

    for module in model.modules():

        if isinstance(module, nn.Dropout):
            module.train()

    predictions = []

    with torch.no_grad():

        for _ in range(passes):

            predictions.append(
                model(x).item()
            )

    model.eval()

    predictions = np.array(predictions)

    mean = float(predictions.mean())

    std = float(predictions.std())

    lower = max(
        0,
        mean - 1.96 * std
    )

    upper = mean + 1.96 * std

    return mean, std, lower, upper


prediction_csv = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test_predictions.csv"
)

mc_mean, uncertainty, lower, upper = mc_prediction(
    model,
    input_tensor
)

predicted_rul = float(mc_mean)

if os.path.exists(prediction_csv):
    try:
        prediction_df = pd.read_csv(prediction_csv)
        match = prediction_df[
            prediction_df["Engine"].astype(int) == int(selected_engine)
        ]
        if not match.empty:
            predicted_rul = float(match.iloc[0]["Predicted_RUL"])
    except Exception:
        predicted_rul = float(mc_mean)

lower = max(0.0, predicted_rul - 1.96 * uncertainty)
upper = predicted_rul + 1.96 * uncertainty


# ============================================================
# XAI
# ============================================================

def xai_analysis(
    model,
    data,
    sensors
):

    model.eval()

    original_tensor = torch.tensor(
        data,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        original_prediction = (
            model(original_tensor).item()
        )

    results = []

    for i, sensor in enumerate(sensors):

        modified = data.copy()

        modified[:, i] = np.mean(
            modified[:, i]
        )

        modified_tensor = torch.tensor(
            modified,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            new_prediction = (
                model(modified_tensor).item()
            )

        importance = abs(
            original_prediction - new_prediction
        )

        results.append(
            {
                "sensor": sensor,
                "importance": importance
            }
        )

    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return results


sensor_importance = xai_analysis(
    model,
    scaled,
    sensors
)


# ============================================================
# TOP SENSOR
# ============================================================

top_sensor = (
    sensor_importance[0]["sensor"]
    if sensor_importance
    else "N/A"
)


# ============================================================
# SENSOR SELECTION
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### SENSOR ANALYSIS")

selected_sensors = st.sidebar.multiselect(
    "Choose sensor channel(s)",
    sensors,
    default=[top_sensor]
)

if not selected_sensors:
    selected_sensors = [top_sensor]

selected_sensor = selected_sensors[0]


# ============================================================
# SELECTED SENSOR ANALYSIS
# ============================================================

selected_sensor_results = []

for sensor in selected_sensors:

    importance = next(
        (
            item["importance"]
            for item in sensor_importance
            if item["sensor"] == sensor
        ),
        0.0
    )

    values = recent[sensor].astype(float).values

    first_value = float(values[0])
    last_value = float(values[-1])

    mean_value = float(
        np.mean(values)
    )

    variation = float(
        np.std(values)
    )

    if abs(first_value) > 1e-8:

        trend = abs(
            (last_value - first_value) / first_value
        ) * 100

    else:

        trend = 0.0

    importance_score = min(
        100,
        (importance / max(
            max(
                item["importance"]
                for item in sensor_importance
            ),
            1e-8
        )) * 100
    )

    trend_score = min(
        100,
        trend
    )

    variation_score = min(
        100,
        variation * 100
    )

    risk_score = (
        0.50 * importance_score
        + 0.35 * trend_score
        + 0.15 * variation_score
    )

    if risk_score >= 65:

        sensor_risk = "HIGH"
        sensor_symbol = "🔴"

    elif risk_score >= 35:

        sensor_risk = "MEDIUM"
        sensor_symbol = "🟡"

    else:

        sensor_risk = "LOW"
        sensor_symbol = "🟢"

    selected_sensor_results.append(
        {
            "sensor": sensor,
            "importance": importance,
            "mean": mean_value,
            "variation": variation,
            "trend": trend,
            "risk_score": risk_score,
            "risk": sensor_risk,
            "symbol": sensor_symbol
        }
    )


# ============================================================
# AGGREGATED MULTI-SENSOR RISK
# ============================================================

if selected_sensor_results:

    aggregate_risk_score = float(
        np.mean(
            [
                x["risk_score"]
                for x in selected_sensor_results
            ]
        )
    )

else:

    aggregate_risk_score = 0.0


if aggregate_risk_score >= 65:

    aggregate_risk = "HIGH"
    aggregate_symbol = "🔴"

elif aggregate_risk_score >= 35:

    aggregate_risk = "MEDIUM"
    aggregate_symbol = "🟡"

else:

    aggregate_risk = "LOW"
    aggregate_symbol = "🟢"


selected_sensor_result = selected_sensor_results[0]


# ============================================================
# AGENTS
# ============================================================

agent_results = run_agent_pipeline(
    predicted_rul,
    uncertainty,
    sensor_importance
)

diagnostic = agent_results["diagnostic"]
root_cause = agent_results["root_cause"]
verification = agent_results["verification"]
maintenance = agent_results["maintenance"]


# ============================================================
# MACHINE HEALTH
# ============================================================

if predicted_rul <= 30:

    health = "CRITICAL"
    health_symbol = "🔴"

elif predicted_rul <= 60:

    health = "WARNING"
    health_symbol = "🟡"

else:

    health = "HEALTHY"
    health_symbol = "🟢"


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="main-title">
    AEGIS AI
</div>

<div class="main-subtitle">
    Explainable Agentic Predictive Maintenance
</div>

<div class="subtitle">
    Deep time-series prediction · uncertainty estimation ·
    explainable AI · multi-agent diagnostic reasoning
</div>
""")


# ============================================================
# HERO MACHINE
# ============================================================

machine_sensor_labels = list(selected_sensors)
for item in sensor_importance:
    s_name = item["sensor"]
    if s_name not in machine_sensor_labels:
        machine_sensor_labels.append(s_name)
    if len(machine_sensor_labels) >= 4:
        break
while len(machine_sensor_labels) < 4:
    machine_sensor_labels.append("sensor_1")
machine_sensor_labels = machine_sensor_labels[:4]


machine_html = f"""
<style>

.machine-wrapper {{
    position: relative;
    width: 100%;
    height: 470px;
    border: 1px solid #1c343d;
    border-radius: 18px;
    background:
        radial-gradient(
            circle at 50% 45%,
            rgba(0,210,230,0.08),
            transparent 42%
        ),
        linear-gradient(145deg,#0b161c,#050a0e);
    overflow: hidden;
    color: #dceff2;
    font-family: Arial, sans-serif;
}}

.machine-grid {{
    position: absolute;
    inset: 0;
    opacity: .16;

    background-image:
        linear-gradient(#28505a 1px, transparent 1px),
        linear-gradient(90deg,#28505a 1px, transparent 1px);

    background-size: 35px 35px;

    mask-image:
        linear-gradient(to bottom,black,transparent);
}}

.machine-header {{
    position:absolute;
    top:22px;
    left:28px;
    font-size:11px;
    letter-spacing:2px;
    color:#66848d;
}}

.machine-online {{
    position:absolute;
    top:22px;
    right:28px;
    color:#36d2df;
    font-size:11px;
    letter-spacing:1px;
}}

.machine-body {{
    position:absolute;
    left:5%;
    top:110px;
    width:56%;
    height:250px;
}}

.casing {{
    position:absolute;
    left:15%;
    top:35px;
    width:68%;
    height:145px;
    border-radius:80px;
    background:linear-gradient(
        145deg,
        #405861,
        #111e24 45%,
        #2a4149
    );
    border:2px solid #48626b;
    box-shadow:
        inset 0 0 35px rgba(0,0,0,.7),
        0 20px 40px rgba(0,0,0,.35);
}}

.casing:after {{
    content:"";
    position:absolute;
    inset:10px;
    border-radius:70px;
    border:1px solid rgba(110,160,170,.18);
}}

.turbine {{
    position:absolute;
    left:4%;
    top:0;
    width:210px;
    height:210px;
    border-radius:50%;
    background:
        radial-gradient(
            circle,
            #081116 0 23%,
            #223941 24% 27%,
            #091218 28% 58%,
            #38515a 59% 62%,
            #091116 63%
        );
    border:3px solid #526a72;
    box-shadow:0 0 35px rgba(0,205,225,.10);
}}

.blades {{
    position:absolute;
    inset:20px;
    animation:spin 7s linear infinite;
}}

.blade {{
    position:absolute;
    left:50%;
    top:50%;
    width:12px;
    height:78px;
    transform-origin:50% 100%;
    margin-left:-6px;
    margin-top:-78px;
    background:linear-gradient(
        90deg,
        #304a53,
        #8aa5aa,
        #304a53
    );
    border-radius:50% 50% 15% 15%;
}}

.blade:nth-child(1) {{transform:rotate(0deg);}}
.blade:nth-child(2) {{transform:rotate(45deg);}}
.blade:nth-child(3) {{transform:rotate(90deg);}}
.blade:nth-child(4) {{transform:rotate(135deg);}}
.blade:nth-child(5) {{transform:rotate(180deg);}}
.blade:nth-child(6) {{transform:rotate(225deg);}}
.blade:nth-child(7) {{transform:rotate(270deg);}}
.blade:nth-child(8) {{transform:rotate(315deg);}}

.hub {{
    position:absolute;
    left:50%;
    top:50%;
    transform:translate(-50%,-50%);
    width:40px;
    height:40px;
    border-radius:50%;
    background:
        radial-gradient(
            circle at 35% 30%,
            #b4d0d4,
            #38545c 45%,
            #0b1419
        );
    border:2px solid #77949a;
    box-shadow:0 0 18px rgba(50,210,225,.28);
}}

.rotor-ring {{
    position:absolute;
    left:50%;
    top:50%;
    transform:translate(-50%,-50%);
    width:175px;
    height:175px;
    border-radius:50%;
    border:1px dashed rgba(80,190,205,.35);
    animation:spinReverse 14s linear infinite;
}}

.pipe {{
    position:absolute;
    left:27%;
    top:101px;
    width:46%;
    height:20px;
    border-radius:20px;
    background:linear-gradient(#182a31,#0a1217);
    border:1px solid #38535b;
}}

.engine-end {{
    position:absolute;
    right:2%;
    top:66px;
    width:100px;
    height:82px;
    transform:skewY(-7deg);
    background:linear-gradient(145deg,#182b32,#0a1318);
    border:1px solid #425b63;
}}

.sensor {{
    position:absolute;
    width:12px;
    height:12px;
    border-radius:50%;
    background:#37d4e1;
    box-shadow:0 0 5px #37d4e1,0 0 18px #37d4e1;
    animation:pulse 1.8s infinite;
}}

.sensor:after {{
    content:"";
    position:absolute;
    width:70px;
    height:1px;
    top:5px;
    background:linear-gradient(90deg,#37d4e1,transparent);
}}

.sensor1 {{left:43%;top:25px;}}
.sensor2 {{left:57%;top:25px;animation-delay:.4s;}}
.sensor3 {{left:70%;top:25px;animation-delay:.8s;}}
.sensor4 {{left:64%;top:185px;animation-delay:1.1s;}}

.sensor-label {{
    position:absolute;
    font-family:monospace;
    font-size:10px;
    color:#6f929a;
    white-space:nowrap;
}}

.label1 {{left:39%;top:4px;}}
.label2 {{left:53%;top:4px;}}
.label3 {{left:66%;top:4px;}}
.label4 {{left:60%;top:205px;}}

.ai-core {{
    position:absolute;
    right:5%;
    top:92px;
    width:30%;
    min-height:270px;
    border:1px solid #24505a;
    border-radius:16px;
    background:linear-gradient(
        145deg,
        rgba(10,27,34,.96),
        rgba(5,12,17,.96)
    );
    box-shadow:0 0 40px rgba(0,180,205,.07);
    padding:22px;
}}

.ai-title {{
    color:#5f8a93;
    font-family:monospace;
    font-size:10px;
    letter-spacing:2px;
}}

.ai-name {{
    margin-top:7px;
    font-size:20px;
    font-weight:600;
}}

.ai-line {{
    height:1px;
    background:#1c3c45;
    margin:15px 0;
}}

.ai-row {{
    display:flex;
    justify-content:space-between;
    margin:10px 0;
    font-family:monospace;
    font-size:10px;
    gap:10px;
}}

.ai-value {{
    color:#48d7e3;
    font-weight:600;
    text-align:right;
}}

.status {{
    color:#3bd4df;
    font-size:9px;
    margin-top:15px;
    letter-spacing:1px;
}}

.signal {{
    position:absolute;
    left:56%;
    top:235px;
    width:160px;
    height:1px;
    background:linear-gradient(
        90deg,
        transparent,
        #38cbd9,
        transparent
    );
    animation:dataflow 2s linear infinite;
}}

.machine-bottom {{
    position:absolute;
    bottom:20px;
    left:28px;
    right:28px;
    display:flex;
    justify-content:space-between;
    color:#526f78;
    font-family:monospace;
    font-size:10px;
}}

@keyframes spin {{
    from {{transform:rotate(0deg);}}
    to {{transform:rotate(360deg);}}
}}

@keyframes spinReverse {{
    from {{
        transform:translate(-50%,-50%) rotate(0deg);
    }}
    to {{
        transform:translate(-50%,-50%) rotate(-360deg);
    }}
}}

@keyframes pulse {{
    0%,100% {{
        opacity:.45;
        transform:scale(.8);
    }}
    50% {{
        opacity:1;
        transform:scale(1.25);
    }}
}}

@keyframes dataflow {{
    0% {{
        opacity:0;
        transform:translateX(-60px);
    }}
    30% {{opacity:1;}}
    100% {{
        opacity:0;
        transform:translateX(150px);
    }}
}}

</style>

<div class="machine-wrapper">

<div class="machine-grid"></div>

<div class="machine-header">
AEGIS DIGITAL TWIN / LIVE ASSET
</div>

<div class="machine-online">
● ONLINE
</div>

<div class="machine-body">

<div class="casing"></div>

<div class="turbine">

<div class="blades">

<div class="blade"></div>
<div class="blade"></div>
<div class="blade"></div>
<div class="blade"></div>
<div class="blade"></div>
<div class="blade"></div>
<div class="blade"></div>
<div class="blade"></div>

</div>

<div class="rotor-ring"></div>

<div class="hub"></div>

</div>

<div class="pipe"></div>

<div class="engine-end"></div>

<div class="sensor sensor1"></div>
<div class="sensor sensor2"></div>
<div class="sensor sensor3"></div>
<div class="sensor sensor4"></div>

<div class="sensor-label label1">
{html.escape(machine_sensor_labels[0].upper())}
</div>

<div class="sensor-label label2">
{html.escape(machine_sensor_labels[1].upper())}
</div>

<div class="sensor-label label3">
{html.escape(machine_sensor_labels[2].upper())}
</div>

<div class="sensor-label label4">
{html.escape(machine_sensor_labels[3].upper())}
</div>

</div>

<div class="ai-core">

<div class="ai-title">
AI PREDICTIVE CORE
</div>

<div class="ai-name">
AEGIS ENGINE
</div>

<div class="ai-line"></div>

<div class="ai-row">
<span>RUL PREDICTION</span>
<span class="ai-value">
{predicted_rul:.1f} cycles
</span>
</div>

<div class="ai-row">
<span>UNCERTAINTY</span>
<span class="ai-value">
±{uncertainty:.1f} cycles
</span>
</div>

<div class="ai-row">
<span>DOMINANT SIGNAL</span>
<span class="ai-value">
{html.escape(top_sensor.upper())}
</span>
</div>

<div class="ai-row">
<span>SELECTED SENSOR</span>
<span class="ai-value">
{html.escape(selected_sensor.upper())}
</span>
</div>

<div class="ai-row">
<span>FAULT RISK</span>
<span class="ai-value">
{selected_sensor_result["risk"]}
</span>
</div>

<div class="ai-line"></div>

<div class="status">
● LSTM ONLINE<br><br>
● MC DROPOUT ACTIVE<br><br>
● XAI ACTIVE<br><br>
● AGENT NETWORK ACTIVE
</div>

</div>

<div class="signal"></div>

<div class="machine-bottom">

<span>
ENGINE / ASSET #{selected_engine}
</span>

<span>
CYCLE {current_cycle}
</span>

<span>
17 SENSOR CHANNELS
</span>

</div>

</div>
"""

st.html(machine_html)


# ============================================================
# LIVE MACHINE INTELLIGENCE
# ============================================================

st.html("""
<div class="intelligence-header">

    <div class="intelligence-kicker">
    AEGIS / LIVE ANALYTICS CORE
    </div>

    <div class="intelligence-title">
    LIVE MACHINE INTELLIGENCE
    </div>

    <div class="intelligence-desc">
    Real-time interpretation of machine condition,
    predicted lifetime, uncertainty and dominant signals.
    </div>

</div>
""")


m1, m2, m3, m4, m5 = st.columns(5)


with m1:

    st.html(f"""
    <div class="metric">

    <div class="metric-label">
    Remaining Useful Life
    </div>

    <div class="metric-value">
    {predicted_rul:.1f}
    </div>

    <div class="metric-small">
    predicted operating cycles
    </div>

    </div>
    """)


with m2:

    st.html(f"""
    <div class="metric">

    <div class="metric-label">
    Uncertainty
    </div>

    <div class="metric-value">
    ±{uncertainty:.1f}
    </div>

    <div class="metric-small">
    50 MC Dropout passes
    </div>

    </div>
    """)


with m3:

    st.html(f"""
    <div class="metric">

    <div class="metric-label">
    Current Cycle
    </div>

    <div class="metric-value">
    {current_cycle}
    </div>

    <div class="metric-small">
    latest observed cycle
    </div>

    </div>
    """)


with m4:

    st.html(f"""
    <div class="metric">

    <div class="metric-label">
    Dominant Signal
    </div>

    <div class="metric-value"
         style="font-size:1.32rem;">
    {html.escape(top_sensor)}
    </div>

    <div class="metric-small">
    highest XAI contribution
    </div>

    </div>
    """)


with m5:

    st.html(f"""
    <div class="health-card">

    <div class="health-label">
    MACHINE HEALTH
    </div>

    <div class="health-value">
    {health_symbol} {health}
    </div>

    <div class="health-engine">
    ENGINE #{selected_engine}
    </div>

    <div class="health-line"></div>

    <div class="health-item">
    ● SENSOR STREAM ACTIVE<br>
    ● LSTM ONLINE<br>
    ● XAI ONLINE
    </div>

    </div>
    """)


# ============================================================
# MACHINE HEALTH FACTORS
# ============================================================

st.html("""
<div class="section-title-large">
MACHINE HEALTH FACTORS
</div>
""")

st.caption(
    "Sensor-channel monitoring from the current 30-cycle observation window. "
    "Channels are intentionally shown by dataset identifier; no physical "
    "meaning is assumed for individual sensor numbers."
)


factor_data = []

for sensor in sensors:

    values = recent[sensor].values.astype(float)

    first = values[0]
    last = values[-1]

    mean_value = float(
        np.mean(values)
    )

    variation = float(
        np.std(values)
    )

    if abs(first) > 1e-8:

        trend = abs(
            (last - first) / first
        ) * 100

    else:

        trend = 0

    factor_data.append(
        {
            "sensor": sensor,
            "mean": mean_value,
            "variation": variation,
            "trend": trend
        }
    )


factor_data.sort(
    key=lambda x: x["trend"],
    reverse=True
)


factor_cols = st.columns(4)


factor_items = [
    (
        "📡",
        "Dominant Sensor",
        top_sensor
    ),
    (
        "📈",
        "Trend Activity",
        f"{factor_data[0]['trend']:.2f}%"
    ),
    (
        "◌",
        "Signal Variation",
        f"{factor_data[0]['variation']:.3f}"
    ),
    (
        "🎛",
        "Channels Active",
        f"{len(sensors)} / 21"
    )
]


for i, (
    icon,
    name,
    value
) in enumerate(factor_items):

    with factor_cols[i]:

        st.html(f"""
        <div class="factor-card">

            <div class="factor-top">

                <div class="factor-name">
                {html.escape(name)}
                </div>

                <div class="factor-icon">
                {icon}
                </div>

            </div>

            <div class="factor-value">
            {html.escape(str(value))}
            </div>

            <div class="factor-bar">

                <div class="factor-fill"
                     style="width:{min(100, 25 + i * 17)}%;">
                </div>

            </div>

            <div class="factor-caption">
            Current observation window · 30 cycles
            </div>

        </div>
        """)


# ============================================================
# SELECTED SENSOR FAULT / DEGRADATION ANALYSIS
# ============================================================

st.html("""
<div class="section-title-large">
SELECTED SENSOR FAULT ANALYSIS
</div>
""")

st.caption(
    "The selected sensor risk is an analytical degradation indicator. "
    "It combines the sensor's recent trend, signal variation and XAI "
    "contribution. It does not assign a physical fault type to the sensor."
)


sensor_analysis_cols = st.columns(4)


with sensor_analysis_cols[0]:

    st.html(f"""
    <div class="metric">

        <div class="metric-label">
        Selected Sensor
        </div>

        <div class="metric-value"
             style="font-size:1.3rem;">
        {html.escape(selected_sensor.upper())}
        </div>

        <div class="metric-small">
        user-selected analysis channel
        </div>

    </div>
    """)


with sensor_analysis_cols[1]:

    st.html(f"""
    <div class="metric">

        <div class="metric-label">
        Fault / Degradation Risk
        </div>

        <div class="metric-value"
             style="font-size:1.3rem;">
        {selected_sensor_result["symbol"]}
        {selected_sensor_result["risk"]}
        </div>

        <div class="metric-small">
        risk score {selected_sensor_result["risk_score"]:.1f} / 100
        </div>

    </div>
    """)


with sensor_analysis_cols[2]:

    st.html(f"""
    <div class="metric">

        <div class="metric-label">
        Sensor Trend
        </div>

        <div class="metric-value">
        {selected_sensor_result["trend"]:.2f}%
        </div>

        <div class="metric-small">
        change across recent 30 cycles
        </div>

    </div>
    """)


with sensor_analysis_cols[3]:

    st.html(f"""
    <div class="metric">

        <div class="metric-label">
        XAI Contribution
        </div>

        <div class="metric-value">
        {selected_sensor_result["importance"]:.2f}
        </div>

        <div class="metric-small">
        effect on predicted RUL
        </div>

    </div>
    """)


# ============================================================
# MULTI-SENSOR AGGREGATE
# ============================================================

if len(selected_sensors) > 1:

    st.html(f"""
    <div style="
        margin-top:12px;
        padding:16px 20px;
        border:1px solid #21434c;
        border-radius:14px;
        background:#081218;
    ">

        <div style="
            color:#5f8a93;
            font-family:'JetBrains Mono',monospace;
            font-size:.59rem;
            letter-spacing:.13em;
        ">
        MULTI-SENSOR RISK AGGREGATION
        </div>

        <div style="
            font-size:1.05rem;
            font-weight:700;
            margin-top:7px;
        ">
        {aggregate_symbol} {aggregate_risk}
        </div>

        <div style="
            color:#718990;
            font-size:.66rem;
            margin-top:6px;
        ">
        {len(selected_sensors)} selected sensor channels ·
        aggregate risk score {aggregate_risk_score:.1f}/100
        </div>

    </div>
    """)


# ============================================================
# AI PIPELINE
# ============================================================

st.html("""
<div class="section-title">
AI REASONING PIPELINE
</div>
""")


pipeline = [
    ("⚙️", "MACHINE", "Digital Twin"),
    ("📡", "SENSORS", "17 Signals"),
    ("🧠", "LSTM", "Time-Series"),
    ("🎯", "RUL", "Prediction"),
    ("◌", "UNCERTAINTY", "MC Dropout"),
    ("🔍", "XAI", "Explanation"),
    ("🤖", "AGENTS", "Reasoning"),
    ("🔧", "ACTION", "Maintenance")
]


cols = st.columns(len(pipeline))


for i, (
    icon,
    title,
    desc
) in enumerate(pipeline):

    with cols[i]:

        st.html(f"""
        <div style="
            text-align:center;
            background:#091218;
            border:1px solid #1b333b;
            border-radius:12px;
            padding:13px 4px;
            min-height:92px;
        ">

        <div style="
            font-size:20px;
            margin-bottom:5px;
        ">
        {icon}
        </div>

        <div style="
            font-size:10px;
            font-weight:600;
        ">
        {title}
        </div>

        <div style="
            color:#5e777f;
            font-size:8px;
            margin-top:4px;
        ">
        {desc}
        </div>

        </div>
        """)


# ============================================================
# TELEMETRY
# ============================================================

st.html("""
<div class="section-title-large">
LIVE TELEMETRY
</div>
""")


left, right = st.columns(2)


with left:

    st.html("""
    <div class="chart-panel">

    <div class="chart-title">
    SENSOR TELEMETRY
    </div>

    <div class="chart-caption">
    SELECTED SENSOR · FULL ENGINE HISTORY
    </div>

    </div>
    """)

    fig = go.Figure()

    for sensor in selected_sensors:

        fig.add_trace(
            go.Scatter(
                x=engine["cycle"],
                y=engine[sensor],
                mode="lines",
                line=dict(
                    width=2.5
                ),
                name=sensor.upper()
            )
        )

    fig.update_layout(
        height=330,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=15,
            r=15,
            t=15,
            b=15
        ),
        xaxis=dict(
            title="Operational Cycle",
            gridcolor="#17272e",
            zeroline=False
        ),
        yaxis=dict(
            title="Sensor Value",
            gridcolor="#17272e",
            zeroline=False
        ),
        showlegend=len(selected_sensors) > 1,
        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with right:

    st.html("""
    <div class="chart-panel">

    <div class="chart-title">
    RUL HEALTH INDICATOR
    </div>

    <div class="chart-caption">
    MODEL ESTIMATE · CURRENT ENGINE CONDITION
    </div>

    </div>
    """)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0, predicted_rul),
            number={
                "suffix": " cycles",
                "font": {
                    "size": 34
                }
            },
            title={
                "text":
                "REMAINING USEFUL LIFE"
            },
            gauge={
                "axis": {
                    "range": [
                        0,
                        max(
                            160,
                            upper + 20
                        )
                    ]
                },
                "bar": {
                    "thickness": .30
                },
                "bgcolor": "#091217",
                "borderwidth": 1,
                "bordercolor": "#203840"
            }
        )
    )

    gauge.update_layout(
        height=330,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=25,
            r=25,
            t=25,
            b=5
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )


# ============================================================
# UNCERTAINTY
# ============================================================

st.html("""
<div class="section-title-large">
UNCERTAINTY ESTIMATION
</div>
""")


def get_mc_values(
    model,
    x,
    passes=50
):

    model.eval()

    for module in model.modules():

        if isinstance(module, nn.Dropout):
            module.train()

    values = []

    with torch.no_grad():

        for _ in range(passes):

            values.append(
                model(x).item()
            )

    model.eval()

    return np.array(values)


mc_values = get_mc_values(
    model,
    input_tensor
)


uncertainty_fig = go.Figure()


uncertainty_fig.add_trace(
    go.Scatter(
        x=list(
            range(
                1,
                len(mc_values) + 1
            )
        ),
        y=mc_values,
        mode="lines+markers",
        line=dict(
            width=2
        ),
        marker=dict(
            size=5
        ),
        name="MC Dropout"
    )
)


uncertainty_fig.add_hline(
    y=predicted_rul,
    line_dash="dot",
    annotation_text=
        f"Mean RUL {predicted_rul:.1f}"
)


uncertainty_fig.update_layout(
    height=320,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(
        l=15,
        r=25,
        t=25,
        b=15
    ),
    yaxis=dict(
        title="RUL · cycles",
        gridcolor="#17272e",
        zeroline=False
    ),
    xaxis=dict(
        title="MC Dropout Pass",
        gridcolor="#17272e"
    ),
    showlegend=False
)


st.plotly_chart(
    uncertainty_fig,
    use_container_width=True
)


# ============================================================
# XAI
# ============================================================

st.html("""
<div class="section-title-large">
EXPLAINABLE AI · SENSOR CONTRIBUTION
</div>
""")


xai_df = pd.DataFrame(
    sensor_importance[:10]
)

xai_df = xai_df.sort_values(
    "importance"
)


xai_fig = go.Figure(
    go.Bar(
        x=xai_df["importance"],
        y=xai_df["sensor"],
        orientation="h",
        text=[
            f"{v:.2f}"
            for v in xai_df["importance"]
        ],
        textposition="outside"
    )
)


xai_fig.update_layout(
    height=410,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(
        l=25,
        r=55,
        t=20,
        b=20
    ),
    xaxis=dict(
        title="Model contribution",
        gridcolor="#17272e",
        zeroline=False
    ),
    yaxis=dict(
        autorange="reversed"
    ),
    showlegend=False
)


st.plotly_chart(
    xai_fig,
    use_container_width=True
)


st.html(f"""
<div style="
    margin-top:10px;
    padding:14px 18px;
    border:1px solid #21434c;
    border-radius:12px;
    background:#081218;
    color:#78959d;
    font-size:.68rem;
">

    <strong style="color:#b9dce0;">
    USER SELECTED:
    </strong>
    {html.escape(selected_sensor.upper())}

    &nbsp;&nbsp;·&nbsp;&nbsp;

    <strong style="color:#b9dce0;">
    XAI CONTRIBUTION:
    </strong>
    {selected_sensor_result["importance"]:.4f}

    &nbsp;&nbsp;·&nbsp;&nbsp;

    <strong style="color:#b9dce0;">
    FAULT RISK:
    </strong>
    {selected_sensor_result["risk"]}

</div>
""")


st.html("""
<div style="
    color:#607780;
    font-size:.64rem;
    margin-top:10px;
">
Sensor contribution is estimated using occlusion:
each sensor's recent 30-cycle sequence is perturbed
and the resulting change in predicted RUL is measured.
</div>
""")


# ============================================================
# MULTI-AGENT NETWORK
# ============================================================

st.html("""
<div class="section-title-large">
MULTI-AGENT REASONING NETWORK
</div>
""")


st.html("""
<div class="agent-network">

    <div class="agent-network-title">
    Autonomous Diagnostic Chain
    </div>

    <div class="agent-network-subtitle">
    Prediction → evidence → root-cause hypothesis →
    verification → maintenance action
    </div>

</div>
""")


a1, arrow1, a2, arrow2, a3, arrow3, a4 = st.columns(
    [
        2.5,
        .35,
        2.5,
        .35,
        2.5,
        .35,
        2.5
    ]
)


with a1:

    st.html(f"""
    <div class="agent-box">

        <div class="agent-icon">
        🔎
        </div>

        <div class="agent-name">
        Diagnostic Agent
        </div>

        <div class="agent-role">
        HEALTH ASSESSMENT
        </div>

        <div class="agent-text">
        {html.escape(diagnostic["diagnosis"])}
        </div>

        <div class="agent-evidence">
        PREDICTED RUL<br>
        {predicted_rul:.1f} cycles<br><br>

        UNCERTAINTY<br>
        ±{uncertainty:.1f} cycles
        </div>

    </div>
    """)


with arrow1:

    st.html("""
    <div class="agent-arrow">
    →
    </div>
    """)


with a2:

    st.html(f"""
    <div class="agent-box">

        <div class="agent-icon">
        🧠
        </div>

        <div class="agent-name">
        Root-Cause Agent
        </div>

        <div class="agent-role">
        DEGRADATION ANALYSIS
        </div>

        <div class="agent-text">
        The agent examines the strongest
        model-contributing signals and
        generates degradation hypotheses.
        </div>

        <div class="agent-evidence">
        PRIMARY SIGNAL<br>
        {html.escape(root_cause["primary_cause"])}<br><br>

        TOP EVIDENCE<br>
        {len(sensor_importance[:5])}
        candidate signals
        </div>

    </div>
    """)


with arrow2:

    st.html("""
    <div class="agent-arrow">
    →
    </div>
    """)


with a3:

    verification_result = (
        verification["verified_causes"][0]["result"]
        if verification["verified_causes"]
        else "No evidence available"
    )

    st.html(f"""
    <div class="agent-box">

        <div class="agent-icon">
        ✓
        </div>

        <div class="agent-name">
        Verification Agent
        </div>

        <div class="agent-role">
        EVIDENCE VALIDATION
        </div>

        <div class="agent-text">
        Candidate causes are checked against
        sensor importance and prediction
        uncertainty before action generation.
        </div>

        <div class="agent-evidence">
        EVIDENCE<br>
        {html.escape(verification_result)}<br><br>

        UNCERTAINTY<br>
        {html.escape(verification["uncertainty_note"])}
        </div>

    </div>
    """)


with arrow3:

    st.html("""
    <div class="agent-arrow">
    →
    </div>
    """)


with a4:

    st.html(f"""
    <div class="agent-box">

        <div class="agent-icon">
        🔧
        </div>

        <div class="agent-name">
        Maintenance Agent
        </div>

        <div class="agent-role">
        ACTION GENERATION
        </div>

        <div class="agent-text">
        The validated machine condition is
        converted into a maintenance action
        and priority.
        </div>

        <div class="agent-evidence">
        PRIORITY<br>
        {html.escape(maintenance["priority"])}<br><br>

        FOCUS<br>
        {html.escape(maintenance["focus"])}
        </div>

    </div>
    """)


# ============================================================
# FINAL AI OUTPUT
# ============================================================

st.html("""
<div class="section-title-large">
FINAL AI OUTPUT
</div>
""")


st.html(f"""
<div class="final-output">

    <div class="final-kicker">
    ✦ AEGIS DECISION ENGINE / FINAL OUTPUT
    </div>

    <div class="final-title">
    AI-GENERATED MAINTENANCE DECISION
    </div>

    <div class="final-action">
    {html.escape(maintenance["action"])}
    </div>

    <div class="final-meta">

        <div class="final-pill">
        PRIORITY:
        <strong>
        {html.escape(maintenance["priority"])}
        </strong>
        </div>

        <div class="final-pill">
        FOCUS:
        <strong>
        {html.escape(maintenance["focus"])}
        </strong>
        </div>

        <div class="final-pill">
        PREDICTED RUL:
        <strong>
        {predicted_rul:.1f} cycles
        </strong>
        </div>

        <div class="final-pill">
        UNCERTAINTY:
        <strong>
        ±{uncertainty:.1f}
        </strong>
        </div>

        <div class="final-pill">
        SENSOR RISK:
        <strong>
        {selected_sensor_result["risk"]}
        </strong>
        </div>

    </div>

    <div style="
        margin-top:20px;
        color:#708991;
        font-size:.68rem;
        line-height:1.6;
        max-width:1000px;
    ">
    {html.escape(maintenance["reason"])}
    </div>

</div>
""")


# ============================================================
# DIAGNOSTIC EVIDENCE
# ============================================================

st.html("""
<div class="section-title-large">
DIAGNOSTIC EVIDENCE
</div>
""")


evidence_cols = st.columns(3)


with evidence_cols[0]:

    st.html(f"""
    <div class="metric">

    <div class="metric-label">
    Diagnostic Status
    </div>

    <div class="metric-value"
         style="font-size:1.3rem;">
    {health_symbol} {health}
    </div>

    <div class="metric-small">
    {html.escape(diagnostic["diagnosis"])}
    </div>

    </div>
    """)


with evidence_cols[1]:

    st.html(f"""
    <div class="metric">

    <div class="metric-label">
    Root Cause Candidate
    </div>

    <div class="metric-value"
         style="font-size:1.3rem;">
    {html.escape(root_cause["primary_cause"])}
    </div>

    <div class="metric-small">
    highest contributing sensor
    </div>

    </div>
    """)


with evidence_cols[2]:

    st.html(f"""
    <div class="metric">

    <div class="metric-label">
    Maintenance Priority
    </div>

    <div class="metric-value"
         style="font-size:1.3rem;">
    {html.escape(maintenance["priority"])}
    </div>

    <div class="metric-small">
    generated from complete agent pipeline
    </div>

    </div>
    """)


# ============================================================
# SELECTED SENSOR EVIDENCE
# ============================================================

st.html("""
<div class="section-title-large">
SELECTED SENSOR EVIDENCE
</div>
""")


selected_evidence_df = pd.DataFrame(
    selected_sensor_results
)

selected_evidence_df = selected_evidence_df[
    [
        "sensor",
        "importance",
        "trend",
        "variation",
        "risk_score",
        "risk"
    ]
].copy()

selected_evidence_df["sensor"] = (
    selected_evidence_df["sensor"].str.upper()
)

selected_evidence_df["importance"] = (
    selected_evidence_df["importance"].round(4)
)

selected_evidence_df["trend"] = (
    selected_evidence_df["trend"].round(2)
)

selected_evidence_df["variation"] = (
    selected_evidence_df["variation"].round(4)
)

selected_evidence_df["risk_score"] = (
    selected_evidence_df["risk_score"].round(2)
)

selected_evidence_df.columns = [
    "Sensor",
    "XAI Contribution",
    "Trend %",
    "Variation",
    "Risk Score",
    "Risk"
]

st.dataframe(
    selected_evidence_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TOP SENSOR EVIDENCE
# ============================================================

st.html("""
<div class="section-title-large">
TOP SENSOR EVIDENCE
</div>
""")


evidence_df = pd.DataFrame(
    sensor_importance[:5]
)

evidence_df["sensor"] = (
    evidence_df["sensor"].str.upper()
)

evidence_df["importance"] = (
    evidence_df["importance"].round(4)
)


st.dataframe(
    evidence_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.html("""
<div class="section-title-large">
MODEL PERFORMANCE · FD001 TEST SET
</div>
""")


p1, p2, p3 = st.columns(3)


with p1:

    st.html("""
    <div class="metric">

    <div class="metric-label">
    Test MAE
    </div>

    <div class="metric-value">
    16.66
    </div>

    <div class="metric-small">
    cycles
    </div>

    </div>
    """)


with p2:

    st.html("""
    <div class="metric">

    <div class="metric-label">
    Test RMSE
    </div>

    <div class="metric-value">
    23.60
    </div>

    <div class="metric-small">
    cycles
    </div>

    </div>
    """)


with p3:

    st.html("""
    <div class="metric">

    <div class="metric-label">
    Architecture
    </div>

    <div class="metric-value"
         style="font-size:1.25rem;">
    LSTM × 2
    </div>

    <div class="metric-small">
    64 hidden units · 17 inputs
    </div>

    </div>
    """)


# ============================================================
# ADVANCED OPERATIONS & WHAT-IF SIMULATOR
# ============================================================

st.html("""
<div class="section-title-large">
ADVANCED OPERATIONS & WHAT-IF SIMULATOR
</div>
""")

st.caption(
    "Interactive scenario simulation allows engineers to adjust operating parameters "
    "and instantly evaluate the impact on Remaining Useful Life (RUL). "
    "You can also generate and export structured maintenance work orders directly."
)

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    st.markdown("### 🎛️ Operational Stress Simulator")
    
    temp_delta = st.slider("Core Temperature Offset (°C)", -50.0, 50.0, 0.0, 1.0)
    pressure_load = st.slider("Bypass Pressure Multiplier", 0.8, 1.5, 1.0, 0.05)
    fuel_flow_rate = st.slider("Fuel Flow Deviation (%)", -10.0, 20.0, 0.0, 0.5)
    
    stress_factor = (temp_delta * 0.4) + ((pressure_load - 1.0) * 30) + (fuel_flow_rate * 0.8)
    simulated_rul = max(1.0, float(predicted_rul - stress_factor))
    rul_delta_val = simulated_rul - predicted_rul
    
    st.metric(
        label="Simulated Remaining Useful Life", 
        value=f"{simulated_rul:.1f} cycles", 
        delta=f"{rul_delta_val:+.1f} cycles vs Baseline"
    )

with sim_col2:
    st.markdown("### 📄 Maintenance Work Orders")
    
    wo_priority = maintenance["priority"]
    wo_action = maintenance["action"]
    
    st.markdown(f"""
    <div style="
        padding: 14px;
        border-radius: 12px;
        background: #081218;
        border: 1px solid #21434c;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        line-height: 1.6;
        color: #91a8ae;
    ">
        <strong>Target Engine:</strong> #{selected_engine}<br>
        <strong>Current Health:</strong> {health}<br>
        <strong>Priority:</strong> {wo_priority}<br>
        <strong>Recommended Action:</strong> {wo_action}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    if st.button("Generate & Export Maintenance Work Order"):
        work_order_payload = {
            "work_order_id": f"WO-AEGIS-{np.random.randint(10000, 99999)}",
            "engine_id": int(selected_engine),
            "current_cycle": int(current_cycle),
            "predicted_rul": float(predicted_rul),
            "uncertainty": float(uncertainty),
            "health_status": health,
            "priority": wo_priority,
            "recommended_action": wo_action,
            "primary_fault_signal": top_sensor,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.success("Work Order Payload Compiled Successfully!")
        st.json(work_order_payload)
        
        import json
        st.download_button(
            label="📥 Download Work Order JSON",
            data=json.dumps(work_order_payload, indent=4),
            file_name=f"WO_Engine_{selected_engine}.json",
            mime="application/json"
        )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
AEGIS AI · EXPLAINABLE AGENTIC PREDICTIVE MAINTENANCE
· NASA C-MAPSS FD001 · RESEARCH PROTOTYPE
</div>
""")