"""Interactive, offline dashboard for the shared DoH test set."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "XGBoost": ("xgboost", "#5144d5", "Phân loại có giám sát trên 29 đặc trưng flow."),
    "Isolation Forest": ("isolation_forest", "#087f70", "Phát hiện bất thường từ các flow benign."),
    "1D-CNN": ("cnn1d", "#c55432", "Học từ kích thước, hướng và thời gian giữa các packet."),
}
DATASET = "CIRA-CIC-DoHBrw-2020"
INK = "#172447"
CSS = """
<style>
.stApp {background:#f3f6ff;color:#172447}
.stApp header[data-testid="stHeader"] {background:#f3f6ff}
.block-container {max-width:1240px;padding-top:2.5rem;padding-bottom:3rem}
.hero {position:relative;overflow:hidden;background:linear-gradient(116deg,#252464,#4836ae 60%,#1b7799);
       border-radius:26px;padding:clamp(1.5rem,4vw,3.5rem);color:white;margin-bottom:1.4rem}
.hero::after {content:"";position:absolute;width:370px;height:370px;right:-60px;top:-150px;
              border:1px solid #ffffff50;border-radius:50%;box-shadow:0 0 0 36px #ffffff12,0 0 0 78px #ffffff0c;pointer-events:none}
.hero small {font-weight:700;letter-spacing:.11em;color:#d7e8ff;font-size:.85rem}
.hero h1 {font-size:clamp(2rem,5vw,3.4rem);font-weight:800;line-height:1.12;margin:.5rem 0;color:white;padding:0}
.hero p {color:#eff2ff;margin:.75rem 0 0;font-size:1.05rem;position:relative;z-index:1}
.kpi {border:1px solid #dce4f7;border-top:4px solid var(--accent);background:white;
      border-radius:16px;padding:1rem 1.1rem;min-height:112px;box-shadow:0 4px 15px #26346c08;
      transition:transform .18s ease,box-shadow .18s ease}
.kpi:hover {transform:translateY(-3px);box-shadow:0 9px 22px #26346c15}
.kpi span {display:block;color:#4b5d7f;font-weight:600;font-size:.9rem}
.kpi strong {display:block;color:#172447;font-size:clamp(1.55rem,3vw,2.3rem);letter-spacing:-.055em}
.kpi small {color:#4b5d7f;font-size:.83rem}
[data-testid="stRadio"] label {font-weight:650}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px}
</style>
"""


@st.cache_data
def read_parquet(path: Path, modified: int) -> pd.DataFrame:
    """Use the file timestamp as part of the cache key when a run is replaced."""
    return pd.read_parquet(path)


def load_models():
    available, problems = {}, []
    for name, (key, _, _) in MODELS.items():
        path = ROOT / "outputs" / key / "predictions.parquet"
        try:
            frame = read_parquet(path, path.stat().st_mtime_ns)
            required = {"sample_id", "y_true", "y_pred", "y_score", "threshold", "split"}
            if not required.issubset(frame) or frame.empty or not frame.split.eq("test").all():
                raise ValueError("file cần chứa dự đoán của tập test")
            if frame.sample_id.isna().any() or not frame.sample_id.is_unique:
                raise ValueError("sample_id bị thiếu hoặc trùng")
            if not all(frame[c].isin([0, 1]).all() for c in ("y_true", "y_pred")):
                raise ValueError("nhãn phải là 0 hoặc 1")
            if not np.isfinite(frame[["y_score", "threshold"]]).all().all():
                raise ValueError("điểm hoặc ngưỡng không hợp lệ")
            if frame.threshold.nunique() != 1 or not frame.y_pred.eq(frame.y_score.ge(frame.threshold)).all():
                raise ValueError("nhãn dự đoán không khớp ngưỡng")
            available[name] = frame
        except FileNotFoundError:
            problems.append(f"{name}: thiếu `outputs/{key}/predictions.parquet`")
        except (ValueError, TypeError, OSError) as error:
            problems.append(f"{name}: {error}")
    return available, problems


def metrics(frame):
    tn, fp, fn, tp = confusion_matrix(frame.y_true, frame.y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    both = frame.y_true.nunique() == 2
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "Accuracy": (tn + tp) / len(frame),
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "FPR": fp / (fp + tn) if fp + tn else np.nan,
        "AP": average_precision_score(frame.y_true, frame.y_score) if both else np.nan,
        "ROC-AUC": roc_auc_score(frame.y_true, frame.y_score) if both else np.nan,
    }


def percent(value):
    return f"{value:.2%}" if np.isfinite(value) else "—"


def style_chart(figure, height=370, *, left=76, right=22, top=18):
    figure.update_layout(
        height=height, margin=dict(l=left, r=right, t=top, b=72),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=14, color=INK),
        hoverlabel=dict(bgcolor="white"),
    )
    figure.update_xaxes(automargin=True, ticks="outside", ticklen=5, title_standoff=16)
    figure.update_yaxes(automargin=True, ticks="outside", ticklen=5, title_standoff=16)
    return figure


def matrix_chart(m, color):
    rgb = tuple(int(color[start:start + 2], 16) for start in (1, 3, 5))
    tint = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.27)"
    figure = go.Figure(go.Heatmap(
        z=[[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        x=["Benign", "Tunnel"], y=["Benign", "Tunnel"],
        texttemplate="%{z:,}", textfont=dict(size=21, color=INK),
        colorscale=[[0, "#f4f6fc"], [1, tint]], showscale=False,
        xgap=6, ygap=6,
        hovertemplate="Thực tế: %{y}<br>Dự đoán: %{x}<br>Flow: %{z:,}<extra></extra>",
    ))
    figure.update_xaxes(title_text="Nhãn dự đoán", side="bottom", showgrid=False)
    figure.update_yaxes(title_text="Nhãn thực tế", autorange="reversed", showgrid=False)
    return style_chart(figure, left=104)


def pr_chart(frames):
    figure = go.Figure()
    for name, frame in frames.items():
        if frame.y_true.nunique() != 2:
            continue
        precision, recall, _ = precision_recall_curve(frame.y_true, frame.y_score)
        figure.add_scatter(
            x=recall, y=precision, name=name, mode="lines", line=dict(color=MODELS[name][1], width=3),
            hovertemplate="Recall %{x:.1%}<br>Precision %{y:.1%}<extra>%{fullData.name}</extra>",
        )
    show_legend = len(figure.data) > 1
    figure.update_layout(
        showlegend=show_legend,
        legend=dict(orientation="h", x=0, y=1.06, xanchor="left", yanchor="bottom"),
    )
    percent_axis = dict(tickformat=".0%", tickvals=[0, .25, .5, .75, 1], gridcolor="#e7ecf8")
    figure.update_xaxes(title_text="Recall", range=[0, 1.03], **percent_axis)
    figure.update_yaxes(title_text="Precision", range=[0, 1.03], **percent_axis)
    return style_chart(figure, top=70 if show_legend else 18)


def chart(figure):
    st.plotly_chart(figure, width="stretch", theme=None, config={"displaylogo": False})


def show_sources(frame, color):
    path = ROOT / "data/processed/cnn1d/test_metadata.parquet"
    if not path.exists():
        st.caption("Cần test_metadata.parquet để xem kết quả theo trình duyệt và công cụ.")
        return
    metadata = read_parquet(path, path.stat().st_mtime_ns)
    joined = frame.merge(metadata[["sample_id", "source", "label"]], on="sample_id", validate="one_to_one")
    if len(joined) != len(frame) or not joined.label.eq(joined.y_true).all():
        st.warning("Metadata nguồn không khớp ID hoặc nhãn test.")
        return
    grouped = joined.groupby(["y_true", "source"]).y_pred.agg(["sum", "size"]).reset_index()
    panels = st.columns(2)
    for label, title, panel in zip((0, 1), ("Báo nhầm theo trình duyệt", "Phát hiện theo công cụ"), panels):
        rows = grouped[grouped.y_true.eq(label)]
        rate = rows["sum"] / rows["size"]
        with panel:
            st.subheader(title)
            fig = go.Figure(go.Bar(
                x=rate, y=rows.source, orientation="h", marker_color=color,
                text=[percent(value) for value in rate], textposition="outside", cliponaxis=False,
                textfont=dict(color=INK, size=14), constraintext="none",
                customdata=rows[["sum", "size"]],
                hovertemplate="%{y}<br>%{customdata[0]:,} / %{customdata[1]:,} flow<extra></extra>",
            ))
            fig.update_xaxes(
                title_text="Tỷ lệ báo nhầm" if label == 0 else "Tỷ lệ phát hiện",
                tickformat=".0%", tickvals=[0, .25, .5, .75, 1],
                range=[0, 1], gridcolor="#e7ecf8",
            )
            fig.update_yaxes(autorange="reversed")
            chart(style_chart(fig, 275, left=92, right=72))


def main():
    st.set_page_config(page_title="DoH Model Lab", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero"><small>{DATASET}</small><h1>DoH Model Lab</h1>'
        '<p>Khám phá kết quả phát hiện DNS tunneling trên tập test chung.</p></div>',
        unsafe_allow_html=True,
    )
    frames, problems = load_models()
    for message in problems:
        st.warning(message)
    if not frames:
        st.stop()

    selected = st.radio("Mô hình", list(frames), horizontal=True)
    frame = frames[selected]
    _, color, description = MODELS[selected]
    st.subheader(selected)
    st.caption(f"{description} · {len(frame):,} flow test · Ngưỡng {frame.threshold.iloc[0]:.6f}")
    selected_metrics = metrics(frame)
    for column, (name, detail) in zip(st.columns(4), (
        ("F1", "Cân bằng Precision và Recall"),
        ("Recall", f"{selected_metrics['tp']:,} tunnel phát hiện đúng"),
        ("FPR", f"{selected_metrics['fp']:,} benign bị báo nhầm"),
        ("AP", "Khả năng xếp hạng tunnel"),
    )):
        with column:
            st.markdown(
                f'<div class="kpi" style="--accent:{color}"><span>{name}</span>'
                f"<strong>{percent(selected_metrics[name])}</strong><small>{detail}</small></div>",
                unsafe_allow_html=True,
            )

    detail, sources, compare = st.tabs(["Tổng quan", "Theo nguồn", "So sánh mô hình"])
    with detail:
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.subheader("Ma trận nhầm lẫn")
                chart(matrix_chart(selected_metrics, color))
        with right:
            with st.container(border=True):
                st.subheader("Đường Precision–Recall")
                chart(pr_chart({selected: frame}))
        st.table(pd.DataFrame([{
            key: percent(selected_metrics[key]) for key in ("Accuracy", "Precision", "ROC-AUC")
        }]), hide_index=True)
    with sources:
        show_sources(frame, color)
    with compare:
        reference = frame.set_index("sample_id").y_true.astype("int64").sort_index()
        if not all(other.set_index("sample_id").y_true.astype("int64").sort_index().equals(reference)
                   for other in frames.values()):
            st.warning("Chỉ so sánh trực tiếp khi các mô hình có cùng ID và nhãn test.")
        else:
            st.subheader("Kết quả trên cùng tập test")
            rows = pd.DataFrame({name: metrics(other) for name, other in frames.items()}).T
            columns = ["F1", "Precision", "Recall", "FPR", "AP", "ROC-AUC"]
            st.dataframe(rows[columns].apply(lambda col: col.map(percent)), width="stretch")
            chart(pr_chart(frames))
            st.caption("Tỷ lệ báo nhầm càng thấp càng tốt. Mỗi mô hình dùng ngưỡng đã khóa của lần chạy.")


if __name__ == "__main__":
    main()
