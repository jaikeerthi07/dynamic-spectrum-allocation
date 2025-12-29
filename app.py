# g_noc_allin_viz_fixed.py
"""
Multi-Cell 4G/5G NOC — ALL-IN with Advanced Visuals (Fixed)
Fix: unique keys for all Plotly charts to avoid StreamlitDuplicateElementId.

Save & run:
    streamlit run g_noc_allin_viz_fixed.py
"""

# ----------------------------
# Standard imports
# ----------------------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from plotly.subplots import make_subplots
import time, random, json, html, math, os, re
from io import BytesIO
from fpdf import FPDF

# ----------------------------
# Optional heavy imports wrapped
# ----------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, MetaData, Table
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False

try:
    import requests
    REQ_AVAILABLE = True
except Exception:
    REQ_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# ----------------------------
# Helpers & session init
# ----------------------------
def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            raise AttributeError
    except Exception:
        st.stop()

# unique UID for this session (used to make keys stable)
if "uid" not in st.session_state:
    st.session_state.uid = f"{int(time.time()*1000)}_{random.randint(1000,9999)}"
UID = st.session_state.uid

# show_plotly helper to avoid duplicate element ids
def show_plotly(fig, name="plot", use_container_width=True, extra=None):
    """
    Display a Plotly figure with a stable unique Streamlit key.
    name: short descriptive identifier for the chart (e.g., "topology", "heatmap")
    extra: optional additional string/number to make key unique (e.g., cell index or step)
    """
    step = st.session_state.get("step", 0)
    # build descriptive key
    key_parts = [name, str(UID), str(step)]
    if extra is not None:
        key_parts.append(str(extra))
    # If this is rendered multiple times in same step, add a small index
    # Determine count for this name in the session (to keep deterministic)
    counter_key = f"_plot_count_{name}"
    count = st.session_state.get(counter_key, 0) + 1
    st.session_state[counter_key] = count
    key_parts.append(str(count))
    uniq = "_".join(key_parts)
    st.plotly_chart(fig, use_container_width=use_container_width, key=uniq)

# reset per-render counters for keys (so keys are deterministic per rerun)
if "_plot_count_reset_ts" not in st.session_state:
    st.session_state["_plot_count_reset_ts"] = time.time()
# simple mechanism: reset counters if step changed or after a few seconds
_last_reset_step = st.session_state.get("_plot_count_step", None)
if _last_reset_step != st.session_state.get("step", None):
    # reset any _plot_count_ keys
    for k in list(st.session_state.keys()):
        if k.startswith("_plot_count_"):
            del st.session_state[k]
    st.session_state["_plot_count_step"] = st.session_state.get("step", None)

# session state defaults
_defaults = {
    "running": False,
    "paused": False,
    "step": 0,
    "alert_log": [],
    "cell_kpis": {},  # per cell list of tuples (util_array, latency, jitter, throughput, anomalous)
    "model": None,
    "anomaly_model": None,
    "topology": None,
    "run_history": [],  # list of snapshots per step
    "dqn_agent": None,
    "q_table": {},
    "optimizer_q": {},
    "chat_history": [],
    "offline_faq": [
        {"q":"what is this dashboard","a":"Simulates multi-cell 4G/5G with prediction, optimization, explainability and voice."},
        {"q":"how to run","a":"Set parameters and press Run Simulation. Use AI Insights to train models."}
    ],
    "last_bot_text": ""
}
for k,v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# Page config & sidebar
# ----------------------------
st.set_page_config(page_title="Multi-Cell 4G/5G NOC", layout="wide")
st.title("Multi-Cell 4G/5G NOC")

with st.sidebar:
    st.header("Simulation Settings")
    num_cells = st.slider("Number of Cells", 1, 8, 4)
    num_bands = st.slider("Bands per Cell", 1, 8, 3)
    num_users = st.slider("Users per Cell", 5, 500, 80)
    interference_level = st.slider("Interference Probability", 0.0, 1.0, 0.15, 0.01)
    steps = st.number_input("Simulation Steps (per run)", 1, 1000, 50)
    speed = st.number_input("Step Delay (s)", 0.0, 1.0, 0.05, step=0.01)

    st.markdown("---")
    st.header("AI & Optimizer")
    enable_ml = st.checkbox("Enable Predictor (RandomForest)", value=True)
    enable_anomaly = st.checkbox("Enable Anomaly (IsolationForest)", value=True)
    enable_dqn = st.checkbox("Enable DQN Optimizer (PyTorch if installed)", value=True)
    dqn_train_epochs = st.number_input("DQN train epochs (on-demand)", 1, 200, 20)
    st.markdown("---")
    st.header("Persistence & Alerts")
    enable_persistence = st.checkbox("Enable SQLite persistence", value=False)
    enable_webhook = st.checkbox("Enable webhook alerts", value=False)
    webhook_url = st.text_input("Webhook URL (optional)", value="")
    st.markdown("---")
    st.header("Run Control")
    start = st.button("🚀 Run Simulation")
    pause = st.button("⏸ Pause")
    resume = st.button("▶ Resume")
    stop = st.button("⏹ Stop")
    reset = st.button("🔄 Reset All")
    st.markdown("---")
    st.header("Demo & Visuals")
    guided_demo = st.button("▶ Start Guided Demo (auto)")
    show_advanced = st.checkbox("Show Advanced Visuals", value=True)
    st.markdown("---")
    st.header("Export")
    export_csv = st.button("📥 Download Alerts CSV")
    export_model = st.button("📦 Export Predictor (.joblib)")
    export_report = st.button("📝 Generate PDF Report")

# ----------------------------
# Optional persistence (SQLite)
# ----------------------------
DB_PATH = os.path.join(os.getcwd(), "noc_runs_allin_viz_fixed.db")
if enable_persistence and SQLALCHEMY_AVAILABLE:
    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    metadata = MetaData()
    runs_table = Table('runs', metadata,
                       Column('id', Integer, primary_key=True, autoincrement=True),
                       Column('timestamp', String),
                       Column('params', JSON),
                       Column('summary', JSON))
    alerts_table = Table('alerts', metadata,
                         Column('id', Integer, primary_key=True, autoincrement=True),
                         Column('run_id', Integer),
                         Column('step', Integer),
                         Column('cell', Integer),
                         Column('band', Integer),
                         Column('util', Float))
    metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    dbs = DBSession()
else:
    dbs = None

# ----------------------------
# Topology generator
# ----------------------------
def gen_topology(num_cells):
    centers = []
    for i in range(num_cells):
        lon = random.uniform(-0.5,0.5) + 77.6 + random.uniform(-0.12,0.12)
        lat = random.uniform(-0.5,0.5) + 12.9 + random.uniform(-0.12,0.12)
        centers.append({"cell": i, "lon": lon, "lat": lat})
    return centers

if st.session_state.topology is None or len(st.session_state.topology) != num_cells:
    st.session_state.topology = gen_topology(num_cells)

# ----------------------------
# Simulation core
# ----------------------------
def generate_cell_data(num_bands, num_users, interference_level, slice_ratios=None):
    if slice_ratios is None:
        slice_ratios = np.array([0.6, 0.25, 0.15])
    user_bands = np.random.randint(0, num_bands, num_users)
    traffic = np.random.exponential(scale=3.0, size=num_users) + np.random.uniform(0.1, 6.0, num_users)
    band_loads = np.array([traffic[user_bands == b].sum() for b in range(num_bands)])
    capacity = np.full(num_bands, 100.0)
    interference = np.random.rand(num_bands) < interference_level
    if interference.any():
        multipliers = np.random.uniform(0.4, 0.8, size=int(interference.sum()))
        capacity[interference] = capacity[interference] * multipliers
    slice_loads = np.zeros((3, num_bands))
    for s_idx in range(3):
        slice_loads[s_idx, :] = band_loads * slice_ratios[s_idx]
    congested = int((band_loads / np.maximum(capacity, 1e-9) > 0.85).any())
    return band_loads, capacity, interference, congested, slice_loads

def compute_kpis(band_loads, capacity):
    util = np.clip((band_loads / np.maximum(capacity, 1e-9)) * 100, 0, 200)
    latency = 5 + np.mean(util)/3 + np.random.normal(0,1)
    jitter = 0.5 + np.std(util)/4 + np.random.normal(0,0.2)
    throughput = np.sum(np.minimum(band_loads, capacity))
    return util, latency, jitter, throughput

def reallocate(band_loads, capacity, mode="balanced"):
    ratios = band_loads / np.maximum(capacity, 1e-9)
    if mode == "balanced":
        idx_high = np.argmax(ratios)
        idx_low = np.argmin(ratios)
        shift = 0.12 * band_loads[idx_high]
        band_loads[idx_high] -= shift
        band_loads[idx_low] += shift
    elif mode == "aggressive":
        idx_high = np.argmax(ratios)
        idx_low = np.argmin(ratios)
        shift = 0.30 * band_loads[idx_high]
        band_loads[idx_high] -= shift
        band_loads[idx_low] += shift
    return np.clip(band_loads, 0, None)

# ----------------------------
# ML helpers: predictor & anomaly
# ----------------------------
from sklearn.ensemble import RandomForestClassifier, IsolationForest

def build_dataset(samples, num_bands, num_users, interference_level):
    X, y = [], []
    for _ in range(samples):
        bl, cap, _, congested, _ = generate_cell_data(num_bands, num_users, interference_level)
        X.append(np.concatenate([bl, cap]))
        y.append(congested)
    return np.array(X), np.array(y)

def train_predictor(num_samples=400):
    X, y = build_dataset(num_samples, num_bands, num_users, interference_level)
    if len(np.unique(y)) < 2:
        return None
    clf = RandomForestClassifier(n_estimators=80, random_state=42)
    clf.fit(X, y)
    return clf

def train_anomaly(num_samples=300):
    X = []
    for _ in range(num_samples):
        bl, cap, _, _, _ = generate_cell_data(num_bands, num_users, interference_level)
        feats = (bl/np.maximum(cap,1e-9))
        X.append(feats)
    X = np.array(X)
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    return iso

# ----------------------------
# DQN / Q-table fallback
# ----------------------------
if TORCH_AVAILABLE:
    class DQNNet(nn.Module):
        def __init__(self, input_dim, output_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim)
            )
        def forward(self, x):
            return self.net(x)

    class DQNAgent:
        def __init__(self, input_dim, action_dim=3, lr=1e-3, gamma=0.98, eps=0.15):
            self.device = torch.device('cpu')
            self.net = DQNNet(input_dim, action_dim).to(self.device)
            self.target = DQNNet(input_dim, action_dim).to(self.device)
            self.target.load_state_dict(self.net.state_dict())
            self.opt = optim.Adam(self.net.parameters(), lr=lr)
            self.gamma = gamma
            self.eps = eps
            from collections import deque, namedtuple
            self.memory = deque(maxlen=2000)
            self.Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))
            self.batch_size = 64
            self.steps_done = 0

        def policy(self, state):
            if random.random() < self.eps:
                return random.randrange(0,3)
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.net(s)
            return int(torch.argmax(q).item())

        def push(self, *args):
            self.memory.append(self.Transition(*args))

        def train_step(self):
            if len(self.memory) < self.batch_size:
                return
            batch = random.sample(self.memory, self.batch_size)
            batch_t = self.Transition(*zip(*batch))
            s = torch.tensor(np.vstack(batch_t.state), dtype=torch.float32).to(self.device)
            a = torch.tensor(batch_t.action, dtype=torch.long).unsqueeze(1).to(self.device)
            r = torch.tensor(batch_t.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
            ns = torch.tensor(np.vstack(batch_t.next_state), dtype=torch.float32).to(self.device)
            done = torch.tensor(batch_t.done, dtype=torch.float32).unsqueeze(1).to(self.device)
            qvals = self.net(s).gather(1, a)
            next_q = self.target(ns).max(1)[0].detach().unsqueeze(1)
            target = r + self.gamma * next_q * (1-done)
            loss = nn.MSELoss()(qvals, target)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.steps_done += 1
            if self.steps_done % 200 == 0:
                self.target.load_state_dict(self.net.state_dict())
else:
    class SimpleQAgent:
        def __init__(self):
            self.q = {}
        def policy(self, state_key, epsilon=0.15):
            if random.random() < epsilon or state_key not in self.q:
                return random.randrange(0,3)
            return int(np.argmax(self.q[state_key]))
        def update(self, state_key, action, reward, alpha=0.2):
            if state_key not in self.q:
                self.q[state_key] = np.zeros(3)
            self.q[state_key][action] = (1-alpha)*self.q[state_key][action] + alpha*(reward)

# ----------------------------
# Advanced visual helpers
# ----------------------------
def animated_utilization_heatmap(run_history, interval=500, max_frames=120):
    if not run_history:
        return go.Figure()
    total = len(run_history)
    stride = max(1, total // max_frames)
    C = len(run_history[0]["grid_utils"])
    B = len(run_history[0]["grid_utils"][0])
    z0 = np.array(run_history[0]["grid_utils"])
    fig = go.Figure(
        data=go.Heatmap(z=z0, x=[f"B{b+1}" for b in range(B)], y=[f"C{c+1}" for c in range(C)],
                        colorscale="Turbo", colorbar=dict(title="Util (%)")),
        layout=go.Layout(
            title=f"Utilization animation (step {run_history[0]['step']})",
            updatemenus=[{
                "type":"buttons",
                "buttons":[{"label":"Play",
                            "method":"animate",
                            "args":[None, {"frame": {"duration": interval, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 0}}]}]
            }]
        )
    )
    frames = []
    for r in run_history[::stride]:
        z = np.array(r["grid_utils"])
        frames.append(go.Frame(data=[go.Heatmap(z=z)], name=str(r["step"]),
                               layout=go.Layout(title_text=f"Utilization (step {r['step']})")))
    fig.frames = frames
    return fig

def sankey_reallocation(prev_bl, post_bl, cell_idx=0):
    prev = np.array(prev_bl, dtype=float)
    post = np.array(post_bl, dtype=float)
    B = len(prev)
    sources, targets, values = [], [], []
    labels = [f"Prev B{b+1}" for b in range(B)] + [f"Post B{b+1}" for b in range(B)]
    if post.sum() == 0:
        for i in range(B):
            sources.append(i); targets.append(B+i); values.append(prev[i])
    else:
        post_prop = post / (post.sum() + 1e-9)
        for i in range(B):
            for j in range(B):
                amount = prev[i] * post_prop[j]
                if amount > 0.0001:
                    sources.append(i); targets.append(B + j); values.append(amount)
    sank = dict(
        type='sankey',
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
        link=dict(source=sources, target=targets, value=values)
    )
    fig = go.Figure(data=[sank])
    fig.update_layout(title_text=f"Traffic reallocation — Cell {cell_idx+1}", font_size=12)
    return fig

def stacked_area_slices_over_time(run_history, cell_idx=0):
    rows = []
    for r in run_history:
        step = r["step"]
        if "slice_loads" in r and cell_idx in r["slice_loads"]:
            sl = np.sum(np.array(r["slice_loads"][cell_idx]), axis=1)
        else:
            util = np.array(r["grid_utils"][cell_idx])
            total = util.sum()
            sl = np.array([0.6, 0.25, 0.15]) * total
        rows.append({"step": step, "eMBB": float(sl[0]), "URLLC": float(sl[1]), "mMTC": float(sl[2])})
    df = pd.DataFrame(rows).sort_values("step")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["step"], y=df["eMBB"], stackgroup='one', name='eMBB'))
    fig.add_trace(go.Scatter(x=df["step"], y=df["URLLC"], stackgroup='one', name='URLLC'))
    fig.add_trace(go.Scatter(x=df["step"], y=df["mMTC"], stackgroup='one', name='mMTC'))
    fig.update_layout(title=f"Slice-level traffic over time — Cell {cell_idx+1}", xaxis_title="step", yaxis_title="Offered traffic")
    return fig

def kpi_correlation_matrix(run_history, last_n=50):
    records = []
    seq = run_history[-last_n:] if len(run_history) >= last_n else run_history
    for r in seq:
        stp = r["step"]
        for c, k in r["kpis"].items():
            util = np.mean(k["util"]) if isinstance(k["util"], (list, np.ndarray)) else k["util"]
            records.append({"step": stp, "cell": int(c), "util": util, "lat": float(k["lat"]), "jit": float(k["jit"]), "thr": float(k["thr"])})
    df = pd.DataFrame(records)
    if df.empty:
        return go.Figure()
    corr = df[["util","lat","jit","thr"]].corr()
    fig = px.imshow(corr, text_auto=True, title="KPI Correlation Matrix")
    return fig

def latency_violin_plot(run_history):
    recs = []
    for r in run_history:
        for c, k in r["kpis"].items():
            recs.append({"cell": f"C{int(c)+1}", "latency": float(k["lat"])})
    df = pd.DataFrame(recs)
    if df.empty:
        return go.Figure()
    fig = px.violin(df, x="cell", y="latency", box=True, points="all", title="Latency distribution per cell")
    return fig

def network_topology_graph(topology, cell_kpis):
    G = nx.Graph()
    for t in topology:
        idx = int(t["cell"])
        util = float(np.mean(cell_kpis[idx][-1][0])) if idx in cell_kpis and len(cell_kpis[idx])>0 else 0.0
        G.add_node(idx, pos=(t["lon"], t["lat"]), util=util)
    nodes = list(G.nodes(data=True))
    for i, (n, d) in enumerate(nodes):
        posi = d['pos']
        dists = []
        for j, (m, md) in enumerate(nodes):
            if n==m: continue
            posj = md['pos']
            dist = (posi[0]-posj[0])**2 + (posi[1]-posj[1])**2
            dists.append((dist, m))
        dists = sorted(dists, key=lambda x:x[0])[:2]
        for dist, m in dists:
            if not G.has_edge(n, m):
                G.add_edge(n, m, weight=max(0.1, 1.0/(math.sqrt(dist)+1e-6)))
    pos = nx.get_node_attributes(G, 'pos')
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    node_utils = [G.nodes[n]['util'] for n in G.nodes()]
    edge_x, edge_y = [], []
    for e in G.edges(data=True):
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_trace = go.Scatter(x=x_nodes, y=y_nodes, mode='markers+text', text=[f"C{n+1}" for n in G.nodes()],
                            marker=dict(size=[8 + u*0.35 for u in node_utils], color=node_utils, colorscale='Viridis', colorbar=dict(title="Util %")),
                            textposition="top center")
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='gray'), hoverinfo='none')
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Network Topology Graph (node size ~ util)", showlegend=False)
    return fig

# ----------------------------
# Recorder HTML for voice
# ----------------------------
recorder_html = f"""
<div style='font-family: Arial, sans-serif; font-size:13px;'>
  <div>
    <button id='recordBtn_{UID}'>🔴 Record</button>
    <button id='stopBtn_{UID}' disabled>⏹ Stop</button>
    <span id='status_{UID}' style='margin-left:8px;'>Idle</span>
  </div>
  <div style='margin-top:8px;'>
    <label><strong>Transcript (copy to chat):</strong></label>
    <div id='transcript_{UID}' style='border:1px solid #ddd; padding:8px; min-height:48px; background:#fafafa'></div>
  </div>
  <div style='margin-top:8px;'>
    <button id='speakBtn_{UID}'>🔊 Speak Last Reply</button>
    <span id='tts_status_{UID}' style='margin-left:8px;color:#555;'></span>
  </div>

<script>
(function() {{
  const recordBtn = document.getElementById('recordBtn_{UID}');
  const stopBtn = document.getElementById('stopBtn_{UID}');
  const status = document.getElementById('status_{UID}');
  const transcript = document.getElementById('transcript_{UID}');
  const speakBtn = document.getElementById('speakBtn_{UID}');
  const ttsStatus = document.getElementById('tts_status_{UID}');
  let recognition = null;
  try {{
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {{
      status.innerText = 'SpeechRecognition not available (use Chrome/Edge).';
      recordBtn.disabled = true;
    }} else {{
      recognition = new SR();
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.onstart = () => {{ status.innerText = 'Recording...'; recordBtn.disabled=true; stopBtn.disabled=false; }};
      recognition.onend = () => {{ status.innerText = 'Idle'; recordBtn.disabled=false; stopBtn.disabled=true; }};
      recognition.onerror = (e) => {{ status.innerText = 'STT error: ' + (e && e.error ? e.error : e); console.error('STT', e); }};
      recognition.onresult = (event) => {{
        let interim = '';
        let final = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {{
          if (event.results[i].isFinal) final += event.results[i][0].transcript;
          else interim += event.results[i][0].transcript;
        }}
        transcript.innerText = (final + ' ' + interim).trim();
      }};
    }}
  }} catch(err) {{
    console.error('STT init', err); status.innerText = 'STT init error'; recordBtn.disabled = true;
  }}

  recordBtn.onclick = () => {{ try {{ if (recognition) recognition.start(); }} catch(e){{ console.error(e); }} }};
  stopBtn.onclick = () => {{ try {{ if (recognition) recognition.stop(); }} catch(e){{ console.error(e); }} }};

  function speakText(text) {{
    if (!text) {{ ttsStatus.innerText = 'No reply.'; return; }}
    if (!('speechSynthesis' in window)) {{ ttsStatus.innerText = 'No TTS support.'; return; }}
    window.speechSynthesis.cancel();
    const ut = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    if (voices && voices.length) {{
      const v = voices.find(v => /en/i.test(v.lang)) || voices[0];
      ut.voice = v;
    }}
    ut.rate = 1.0; ut.pitch = 1.0; ut.lang = 'en-US';
    ttsStatus.innerText = 'Speaking...';
    ut.onend = () => {{ ttsStatus.innerText = 'Done.'; }};
    ut.onerror = (e) => {{ console.error('TTS err', e); ttsStatus.innerText = 'TTS error'; }};
    window.speechSynthesis.speak(ut);
  }}

  speakBtn.onclick = () => {{
    try {{
      const lastDiv = document.getElementById('last_bot_reply_{UID}');
      const text = lastDiv ? lastDiv.innerText : (window._last_bot_reply || '');
      speakText(text);
    }} catch(e){{ console.error('Speak click', e); ttsStatus.innerText = 'Speak error'; }}
  }};
}})();
</script>
"""

with st.sidebar.expander("🤖 Voice Controls"):
    try:
        st.components.v1.html(recorder_html, height=260)
    except Exception:
        st.sidebar.markdown(recorder_html, unsafe_allow_html=True)

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs(["Overview","Advanced Visuals","AI Insights","DQN Trainer","Voice Assistant","Replay & Demo","Reports","Control"])
tab_overview, tab_adv, tab_ai, tab_dqn, tab_voice, tab_replay, tab_reports, tab_ctrl = tabs

# ----------------------------
# Overview tab
# ----------------------------
with tab_overview:
    st.subheader("Overview & Live Metrics")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🌐 Mean Net Util (%)", "0%")
    c2.metric("⚡ Avg Throughput (Mbps)", "0")
    c3.metric("📶 Avg Latency (ms)", "0")
    c4.metric("📉 Avg Jitter (ms)", "0")

    left,right = st.columns([1.6,1])
    with left:
        st.markdown("### Topology")
        topo_fig = network_topology_graph(st.session_state.topology, st.session_state.cell_kpis)
        show_plotly(topo_fig, name="topology")
        st.markdown("### Heatmap (latest)")
        if st.session_state.step == 0:
            empty_utils = np.zeros((num_cells, num_bands))
            fig = animated_utilization_heatmap([{"step":0,"grid_utils":empty_utils}], interval=400)
            show_plotly(fig, name="heatmap_init", extra=0)
        else:
            grid_utils = np.zeros((num_cells, num_bands))
            for c in range(num_cells):
                if c in st.session_state.cell_kpis and st.session_state.cell_kpis[c]:
                    grid_utils[c,:] = st.session_state.cell_kpis[c][-1][0]
            fig = animated_utilization_heatmap([{"step":st.session_state.step,"grid_utils":grid_utils}], interval=400)
            show_plotly(fig, name="heatmap_live", extra=st.session_state.step)
    with right:
        st.markdown("### Recent Alerts")
        if st.session_state.alert_log:
            st.dataframe(pd.DataFrame(st.session_state.alert_log[:50]))
        else:
            st.info("No alerts yet. Run the simulation.")

# ----------------------------
# Advanced Visuals tab
# ----------------------------
with tab_adv:
    st.subheader("Advanced Visualizations")
    if not st.session_state.run_history:
        st.info("Run simulation to populate run_history. Or run the Guided Demo (sidebar) for an example.")
    else:
        if st.checkbox("Show animated utilization (heatmap)"):
            fig_anim = animated_utilization_heatmap(st.session_state.run_history, interval=400)
            show_plotly(fig_anim, name="anim_heatmap", extra=len(st.session_state.run_history))
        st.markdown("---")
        st.markdown("### Sankey reallocation (compare last two steps)")
        if len(st.session_state.run_history) >= 2:
            prev = np.array(st.session_state.run_history[-2]["grid_utils"])
            post = np.array(st.session_state.run_history[-1]["grid_utils"])
            sel_cell = st.selectbox("Select cell for Sankey", options=list(range(1, num_cells+1)), index=0) - 1
            sank = sankey_reallocation(prev[sel_cell], post[sel_cell], cell_idx=sel_cell)
            show_plotly(sank, name="sankey", extra=f"{sel_cell}_{st.session_state.step}")
        else:
            st.info("Need at least 2 steps in run history for Sankey.")
        st.markdown("---")
        st.markdown("### Stacked area: slice traffic over time (per cell)")
        sel_cell2 = st.selectbox("Choose cell for slices", options=list(range(1, num_cells+1)), index=0) - 1
        fig_slices = stacked_area_slices_over_time(st.session_state.run_history, cell_idx=sel_cell2)
        show_plotly(fig_slices, name="slices", extra=sel_cell2)
        st.markdown("---")
        st.markdown("### KPI correlation matrix")
        fig_corr = kpi_correlation_matrix(st.session_state.run_history)
        show_plotly(fig_corr, name="kpi_corr")
        st.markdown("---")
        st.markdown("### Latency distributions (violin)")
        fig_v = latency_violin_plot(st.session_state.run_history)
        show_plotly(fig_v, name="lat_violin")
        st.markdown("---")
        st.markdown("### Network topology (graph)")
        topo_graph_fig = network_topology_graph(st.session_state.topology, st.session_state.cell_kpis)
        show_plotly(topo_graph_fig, name="topo_graph")

# ----------------------------
# AI Insights tab
# ----------------------------
with tab_ai:
    st.subheader("AI Insights & Models")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Predictor (RandomForest)")
        if enable_ml and st.session_state.model is None:
            with st.spinner("Training predictor on synthetic data..."):
                st.session_state.model = train_predictor(num_samples=600)
                if st.session_state.model is not None:
                    st.success("Predictor trained.")
                else:
                    st.warning("Could not train predictor (insufficient label variety).")
        if st.session_state.model is not None:
            st.write("Predictor ready.")
            if st.button("Retrain Predictor"):
                st.session_state.model = train_predictor(num_samples=800)
                st.success("Retrained predictor.")
            try:
                fi = st.session_state.model.feature_importances_
                n = len(fi)//2
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[f"band{i+1}" for i in range(n)], y=fi[:n], name="band_load_importance"))
                show_plotly(fig, name="feature_importance")
            except Exception:
                st.write("Feature importance unavailable.")
    with colB:
        st.markdown("#### Anomaly detection")
        if enable_anomaly and st.session_state.anomaly_model is None:
            with st.spinner("Training anomaly detector..."):
                try:
                    st.session_state.anomaly_model = train_anomaly(num_samples=400)
                    st.success("Anomaly detector trained.")
                except Exception:
                    st.warning("Anomaly train failed.")
        if st.session_state.anomaly_model is not None:
            st.write("Anomaly detector ready.")
            if st.button("Retrain Anomaly Detector"):
                st.session_state.anomaly_model = train_anomaly(num_samples=500)
                st.success("Retrained anomaly detector.")
    st.markdown("#### SHAP explainability")
    if not SHAP_AVAILABLE:
        st.info("SHAP not installed. Install `shap` to enable feature explanations.")
    else:
        if st.session_state.model is not None and st.session_state.step>0:
            sample_cell = 0
            for c in range(num_cells):
                if c in st.session_state.cell_kpis and st.session_state.cell_kpis[c]:
                    sample_cell = c; break
            last_util = st.session_state.cell_kpis[sample_cell][-1][0]
            last_cap = np.full(len(last_util), 100.0)
            feat = np.concatenate([last_util, last_cap]).reshape(1,-1)
            try:
                expl = shap.TreeExplainer(st.session_state.model)
                sv = expl.shap_values(feat)
                vals = sv[1].reshape(-1)
                feat_names = [f"bl{i+1}" for i in range(len(last_util))] + [f"cap{i+1}" for i in range(len(last_cap))]
                shap_df = pd.DataFrame({"feature":feat_names, "shap":vals})
                shap_df = shap_df.sort_values("shap", key=abs, ascending=False).head(12)
                st.dataframe(shap_df)
            except Exception as e:
                st.write("SHAP explain failed:", e)
        else:
            st.info("Run simulation and train model to show SHAP.")

# ----------------------------
# DQN Trainer tab
# ----------------------------
with tab_dqn:
    st.subheader("DQN Trainer & Agent")
    if TORCH_AVAILABLE:
        st.success("PyTorch available — full DQN enabled.")
    else:
        st.warning("PyTorch not available — using Q-table fallback.")
    if st.session_state.dqn_agent is None:
        if st.button("Initialize Agent"):
            state_dim = num_bands*2
            if TORCH_AVAILABLE:
                st.session_state.dqn_agent = DQNAgent(input_dim=state_dim, action_dim=3)
            else:
                st.session_state.dqn_agent = SimpleQAgent()
            st.success("Agent initialized.")
    else:
        st.info("Agent present.")
    if st.session_state.dqn_agent is not None and TORCH_AVAILABLE:
        if st.button("Run short DQN training step"):
            with st.spinner("Training..."):
                for _ in range(max(10, dqn_train_epochs)):
                    st.session_state.dqn_agent.train_step()
                st.success("DQN training executed.")
    st.markdown("Agent Q-table / memory summary:")
    if TORCH_AVAILABLE and st.session_state.dqn_agent is not None:
        mem_len = len(st.session_state.dqn_agent.memory) if hasattr(st.session_state.dqn_agent, "memory") else 0
        st.write("DQN memory size:", mem_len)
    elif st.session_state.dqn_agent is not None:
        st.write("Q-table entries:", len(st.session_state.dqn_agent.q))

# ----------------------------
# Voice Assistant tab
# ----------------------------
with tab_voice:
    st.subheader("Voice Assistant (Recorder + Chat)")
    st.markdown("Use Recorder in sidebar. Copy transcript into the chat box, Ask Bot, then Speak Last Reply.")
    with st.expander("Edit FAQ"):
        q = st.text_input("FAQ question (short)")
        a = st.text_area("FAQ answer")
        if st.button("Add FAQ"):
            if q.strip() and a.strip():
                st.session_state.offline_faq.insert(0, {"q":q.strip().lower(),"a":a.strip()})
                st.success("FAQ added.")
                safe_rerun()
    st.markdown("### Chat history")
    for u,b in st.session_state.chat_history[-8:]:
        st.markdown(f"**You:** {html.escape(u)}")
        st.markdown(f"**Bot:** {html.escape(b)}")
    user_q = st.text_area("Type question or paste transcript", height=100)
    if st.button("Ask Bot"):
        if user_q.strip():
            def bot_answer(qs):
                ql = qs.lower()
                for it in st.session_state.offline_faq:
                    if it["q"] in ql:
                        return it["a"]
                if "explain" in ql and "cell" in ql:
                    m = re.search(r'cell\s*(\d+)', ql)
                    if m:
                        ci = int(m.group(1))-1
                        if ci in st.session_state.cell_kpis and st.session_state.cell_kpis[ci]:
                            last = st.session_state.cell_kpis[ci][-1]
                            return f"Cell {ci+1} latest util: {np.mean(last[0]):.1f}% latency {last[1]:.1f}ms throughput {last[3]:.1f}Mbps"
                        return f"No state for cell {ci+1} yet."
                return "I can explain KPIs, alerts, and optimizer actions. Try: 'Explain cell 1'."
            ans = bot_answer(user_q.strip())
            st.session_state.chat_history.append((user_q.strip(), ans))
            st.session_state.last_bot_text = ans
            safe_html = html.escape(ans)
            st.markdown(f"<div id='last_bot_reply_{UID}' style='display:none'>{safe_html}</div>", unsafe_allow_html=True)
            st.markdown(f"<script>window._last_bot_reply = {json.dumps(ans)};</script>", unsafe_allow_html=True)
            safe_rerun()

# ----------------------------
# Replay & Guided Demo tab
# ----------------------------
with tab_replay:
    st.subheader("Replay & Guided Demo")
    if not st.session_state.run_history:
        st.info("No run history yet. Run Simulation or click Guided Demo (sidebar).")
    else:
        max_step = len(st.session_state.run_history) - 1
        step_idx = st.slider("Replay step", 0, max(0, max_step), 0)
        if st.button("Show step"):
            s = st.session_state.run_history[step_idx]
            gu = np.array(s["grid_utils"])
            fig_single = animated_utilization_heatmap([s], interval=400)
            show_plotly(fig_single, name="replay_step", extra=step_idx)
            kpis = s["kpis"]
            rows = []
            for c, k in kpis.items():
                rows.append({"Cell": f"C{int(c)+1}", "Mean Util (%)": float(np.mean(k["util"])), "Latency": float(k["lat"]), "Jitter": float(k["jit"]), "Throughput": float(k["thr"])})
            st.table(pd.DataFrame(rows))
    if guided_demo:
        st.session_state.run_history = []
        st.session_state.alert_log = []
        st.session_state.cell_kpis = {}
        st.session_state.step = 0
        st.session_state.running = True
        demo_steps = 8
        for si in range(demo_steps):
            grid_utils = np.zeros((num_cells, num_bands))
            slice_store = {}
            kpis_snapshot = {}
            for c in range(num_cells):
                bl, cap, interf, congested, sl = generate_cell_data(num_bands, num_users, interference_level)
                util, lat, jit, thr = compute_kpis(bl, cap)
                grid_utils[c,:] = util
                st.session_state.cell_kpis.setdefault(c, []).append((util, lat, jit, thr, False))
                slice_store[c] = sl.tolist()
                kpis_snapshot[c] = {"util": util.tolist(), "lat":lat, "jit":jit, "thr":thr}
                for b_i, v in enumerate(util):
                    if v > 92:
                        st.session_state.alert_log.insert(0, {"Step": st.session_state.step+1, "Cell": c+1, "Band": b_i+1, "Util": f"{v:.1f}%"})
            st.session_state.run_history.append({"step": st.session_state.step, "grid_utils": grid_utils.tolist(), "kpis": kpis_snapshot, "slice_loads": slice_store})
            st.session_state.step += 1
            time.sleep(0.15)
        st.success("Guided demo run complete (short). Use Replay controls to inspect steps.")
        narration = "Guided demo recorded. Use the replay controls to inspect alerts, reallocation events, and SHAP explanations."
        st.session_state.last_bot_text = narration
        st.markdown(f"<div id='last_bot_reply_{UID}' style='display:none'>{html.escape(narration)}</div>", unsafe_allow_html=True)
        st.markdown(f"<script>window._last_bot_reply = {json.dumps(narration)};</script>", unsafe_allow_html=True)

# ----------------------------
# Reports tab
# ----------------------------
with tab_reports:
    st.subheader("Reports & Exports")
    if export_csv:
        if st.session_state.alert_log:
            csv_bytes = pd.DataFrame(st.session_state.alert_log).to_csv(index=False).encode('utf-8')
            st.download_button("Download Alerts CSV", csv_bytes, file_name="alerts.csv", mime="text/csv")
        else:
            st.info("No alerts to export.")
    if export_model:
        if JOBLIB_AVAILABLE and st.session_state.model is not None:
            buf = BytesIO()
            joblib.dump(st.session_state.model, buf)
            buf.seek(0)
            st.download_button("Download Predictor Model (.joblib)", buf, file_name="predictor.joblib")
        else:
            st.info("Model not available or joblib not installed.")
    if export_report:
        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Arial", size=12)
        pdf.cell(0,10,"NOC Simulation Report (ALL-IN Viz Fixed)", ln=True, align="C")
        pdf.ln(4)
        pdf.cell(0,8, f"Steps recorded: {len(st.session_state.run_history)}", ln=True)
        pdf.cell(0,8, f"Cells: {num_cells}, Bands: {num_bands}", ln=True)
        pdf.ln(6)
        pdf.cell(0,8, "Recent alerts:", ln=True)
        if st.session_state.alert_log:
            for al in st.session_state.alert_log[:20]:
                pdf.multi_cell(0,6, f"Step {al['Step']} - Cell {al['Cell']} Band {al['Band']} Util {al['Util']}")
        else:
            pdf.cell(0,6,"No alerts.", ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("Download PDF Report", pdf_bytes, file_name="noc_report_allin_viz_fixed.pdf", mime="application/pdf")

# ----------------------------
# Control tab
# ----------------------------
with tab_ctrl:
    st.subheader("Control & Run")
    st.write("Run history length:", len(st.session_state.run_history))
    if st.button("Clear History & Alerts"):
        st.session_state.run_history = []
        st.session_state.alert_log = []
        st.success("Cleared.")
    st.markdown("### Manual Reallocation Preview")
    demo_cell = st.number_input("Cell (1-based)", min_value=1, max_value=num_cells, value=1) - 1
    demo_mode = st.selectbox("Mode", ["none","balanced","aggressive"])
    if st.button("Apply Manual Realloc (preview)"):
        if demo_cell in st.session_state.cell_kpis and st.session_state.cell_kpis[demo_cell]:
            last = st.session_state.cell_kpis[demo_cell][-1][0]
            cap = np.full(len(last), 100.0)
            bl = (last/100.0) * cap
            new_bl = reallocate(bl, cap, mode=demo_mode) if demo_mode!="none" else bl
            new_util = np.clip((new_bl/cap)*100,0,200)
            st.table(pd.DataFrame({"Band":[f"B{i+1}" for i in range(len(new_util))],"Util":new_util}))
        else:
            st.warning("No state for this cell yet.")

# ----------------------------
# Simulation runner
# ----------------------------
if reset:
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    st.experimental_rerun()

if start:
    st.session_state.running = True; st.session_state.paused = False
if pause:
    st.session_state.paused = True
if resume:
    st.session_state.paused = False
if stop:
    st.session_state.running = False

# initialize agent
if st.session_state.dqn_agent is None and enable_dqn:
    if TORCH_AVAILABLE:
        st.session_state.dqn_agent = DQNAgent(input_dim=num_bands*2, action_dim=3)
    else:
        st.session_state.dqn_agent = SimpleQAgent()

if st.session_state.running:
    prog = st.progress(0.0)
    total_utils = []
    for s in range(steps):
        if st.session_state.paused:
            st.info("Paused — click Resume to continue.")
            time.sleep(0.25); continue

        grid_utils = np.zeros((num_cells, num_bands))
        snap_kpis = {}
        slice_store = {}
        for c in range(num_cells):
            bl, cap, interf, congested, sl = generate_cell_data(num_bands, num_users, interference_level)
            predicted = False
            if enable_ml and st.session_state.model is not None:
                feat = np.concatenate([bl, cap]).reshape(1,-1)
                try:
                    predicted = bool(st.session_state.model.predict(feat)[0])
                except Exception:
                    predicted = False
            anomalous = False
            if enable_anomaly and st.session_state.anomaly_model is not None:
                try:
                    an = st.session_state.anomaly_model.predict((bl/np.maximum(cap,1e-9)).reshape(1,-1))[0]
                    anomalous = (an == -1)
                except Exception:
                    anomalous = False
            action = 0
            if enable_dqn and st.session_state.dqn_agent is not None:
                state = np.concatenate([bl/np.maximum(cap,1e-9), cap/100.0])
                if TORCH_AVAILABLE:
                    action = st.session_state.dqn_agent.policy(state)
                else:
                    state_key = str(int(np.max((bl/np.maximum(cap,1e-9))*100)//10))
                    action = st.session_state.dqn_agent.policy(state_key)
                if action == 1:
                    bl = reallocate(bl, cap, mode="balanced")
                elif action == 2:
                    bl = reallocate(bl, cap, mode="aggressive")
            else:
                if predicted or congested:
                    bl = reallocate(bl, cap, mode="balanced")
            util, lat, jit, thr = compute_kpis(bl, cap)
            grid_utils[c,:] = util
            st.session_state.cell_kpis.setdefault(c, []).append((util, lat, jit, thr, anomalous))
            slice_store[c] = sl.tolist()
            snap_kpis[c] = {"util": util.tolist(), "lat":lat, "jit":jit, "thr":thr}
            for b_i, v in enumerate(util):
                if v > 92:
                    alert = {"Step": st.session_state.step+1, "Cell": c+1, "Band": b_i+1, "Util": f"{v:.1f}%"}
                    st.session_state.alert_log.insert(0, alert)
                    if enable_webhook and REQ_AVAILABLE and webhook_url:
                        try:
                            requests.post(webhook_url, json={"text": f"ALERT: Step {alert['Step']} Cell {alert['Cell']} Band {alert['Band']} Util {alert['Util']}"}, timeout=2)
                        except Exception:
                            pass
        st.session_state.run_history.append({"step": st.session_state.step, "grid_utils": grid_utils.tolist(), "kpis": snap_kpis, "slice_loads": slice_store})
        total_utils.append(float(np.mean(grid_utils)))
        st.session_state.step += 1
        prog.progress((s+1)/steps)
        time.sleep(max(0.0, float(speed)))
        if not st.session_state.running:
            break
    prog.empty()
    st.success(f"Run finished. Avg util: {np.mean(total_utils):.1f}%")
    if enable_persistence and SQLALCHEMY_AVAILABLE and dbs is not None:
        try:
            summary = {"avg_util": float(np.mean(total_utils)), "steps": len(st.session_state.run_history)}
            dbs.execute(runs_table.insert().values(timestamp=str(time.time()), params={"cells":num_cells,"bands":num_bands}, summary=summary))
            dbs.commit()
        except Exception:
            pass

st.markdown("---")
