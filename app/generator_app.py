from __future__ import annotations

import base64
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run.generation_config import DEFAULT_CONFIG, WEATHER_TYPES, discover_scene_map
from utils.project_paths import resolve_project_path

BRAND_ICON = PROJECT_ROOT / "app" / "assets" / "v2x-sensecomm-icon.png"
GENERATED_DIR = PROJECT_ROOT / "app" / "generated"
GENERATED_CONFIG = GENERATED_DIR / "generation_config.yaml"

st.set_page_config(
    page_title="V2X-SenseComm Data Generation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


SECTION_META = {
    "run_settings": ("Run Settings", "settings", "#0f62fe"),
    "scene_conditions": ("Scene Conditions", "cloud", "#06b6d4"),
    "traffic_geometry": ("Traffic Geometry", "map", "#0f62fe"),
    "sensor_modules": ("Sensor Modules", "sensor", "#06b6d4"),
    "communication": ("Communication", "radio", "#6366f1"),
    "output": ("Output", "folder", "#6366f1"),
    "run": ("Run", "play", "#12b76a"),
    "preview": ("Preview", "eye", "#4f46e5"),
}


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def svg_icon(name: str) -> str:
    icons = {
        "settings": '<path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"/><path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.05.05a2.1 2.1 0 0 1-2.97 2.97l-.05-.05a1.8 1.8 0 0 0-1.98-.36 1.8 1.8 0 0 0-1.09 1.65V21a2.1 2.1 0 0 1-4.2 0v-.08a1.8 1.8 0 0 0-1.09-1.65 1.8 1.8 0 0 0-1.98.36l-.05.05a2.1 2.1 0 1 1-2.97-2.97l.05-.05a1.8 1.8 0 0 0 .36-1.98 1.8 1.8 0 0 0-1.65-1.09H3a2.1 2.1 0 1 1 0-4.2h.08a1.8 1.8 0 0 0 1.65-1.09 1.8 1.8 0 0 0-.36-1.98l-.05-.05a2.1 2.1 0 0 1 2.97-2.97l.05.05a1.8 1.8 0 0 0 1.98.36 1.8 1.8 0 0 0 1.09-1.65V3a2.1 2.1 0 1 1 4.2 0v.08a1.8 1.8 0 0 0 1.09 1.65 1.8 1.8 0 0 0 1.98-.36l.05-.05a2.1 2.1 0 0 1 2.97 2.97l-.05.05a1.8 1.8 0 0 0-.36 1.98 1.8 1.8 0 0 0 1.65 1.09H21a2.1 2.1 0 1 1 0 4.2h-.08A1.8 1.8 0 0 0 19.4 15Z"/>',
        "cloud": '<path d="M17.5 18H7a4 4 0 1 1 .9-7.9A5.5 5.5 0 0 1 18.4 12H19a3 3 0 0 1-1.5 6Z"/><path d="M8 21h8"/><path d="M10 3v2"/><path d="m4 5 1.5 1.5"/><path d="M20 5 18.5 6.5"/>',
        "map": '<path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3V6Z"/><path d="M9 3v15"/><path d="M15 6v15"/>',
        "sensor": '<path d="M12 3v3"/><path d="M12 18v3"/><path d="M3 12h3"/><path d="M18 12h3"/><circle cx="12" cy="12" r="4"/><path d="m5.6 5.6 2.1 2.1"/><path d="m16.3 16.3 2.1 2.1"/><path d="m18.4 5.6-2.1 2.1"/><path d="m7.7 16.3-2.1 2.1"/>',
        "radio": '<path d="M4.9 19.1a10 10 0 0 1 0-14.2"/><path d="M19.1 4.9a10 10 0 0 1 0 14.2"/><path d="M8.5 15.5a5 5 0 0 1 0-7"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><circle cx="12" cy="12" r="1.8"/>',
        "note": '<path d="M5 4h10l4 4v12H5V4Z"/><path d="M15 4v5h5"/><path d="M8 13h8"/><path d="M8 17h6"/>',
        "folder": '<path d="M3 6.5h6l2 2h10v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6.5Z"/><path d="M3 10h18"/>',
        "play": '<circle cx="12" cy="12" r="9"/><path d="m10 8 6 4-6 4V8Z"/>',
        "eye": '<path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="3"/>',
    }
    body = icons.get(name, icons["settings"])
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #10233f;
            --muted: #667085;
            --blue: #0f62fe;
            --line: #dbe5f2;
            --soft: #f8fbff;
            --shadow: 0 6px 18px rgba(16, 35, 63, 0.045);
        }

        .stApp {
            background: #ffffff;
            color: var(--ink);
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        header[data-testid="stHeader"],
        .stAppToolbar,
        .stAppHeader {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        .block-container {
            max-width: 1240px;
            padding: 8px 22px 28px;
        }

        section[data-testid="stMain"] .block-container > div[data-testid="stVerticalBlock"] {
            gap: 8px !important;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #edf2f7;
            min-width: 218px !important;
            max-width: 218px !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 4px 4px 16px;
        }

        .sidebar-brand img {
            width: 38px;
            height: 38px;
            object-fit: contain;
        }

        .sidebar-brand strong {
            display: block;
            font-size: 24.5px;
            line-height: 1.15;
            color: var(--ink);
        }

        .sidebar-brand span {
            color: var(--muted);
            font-size: 21.5px;
        }

        .sidebar-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 11px;
            margin: 6px 0;
            border-radius: 12px;
            color: #1f2937;
            font-weight: 560;
            font-size: 23.5px;
            border: 1px solid transparent;
        }

        .sidebar-item.active {
            background: #eef5ff;
            color: #0f62fe;
            border-color: #d7e8ff;
        }

        .sidebar-item svg {
            width: 18px;
            height: 18px;
            color: currentColor;
        }

        .top-header {
            display: grid;
            grid-template-columns: minmax(420px, 1fr) minmax(420px, 560px);
            align-items: center;
            gap: 24px;
            padding-bottom: 16px;
            margin-bottom: 18px;
            border-bottom: 1px solid #edf2f7;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand-row img {
            width: 34px;
            height: 34px;
            object-fit: contain;
        }

        .page-title {
            margin: 0;
            color: var(--ink);
            font-size: clamp(30px, 2.1vw, 37px);
            line-height: 1.06;
            font-weight: 620;
            letter-spacing: 0;
        }

        .page-subtitle {
            display: none;
            margin: 0;
            color: var(--muted);
            font-size: 22px;
        }

        .header-rule {
            height: 1px;
            background: #edf2f7;
            margin: 8px 0 12px;
        }

        .stButtonGroup {
            display: flex;
            justify-content: flex-end;
        }

        .stButtonGroup button {
            min-width: 132px !important;
            height: 36px !important;
            border-radius: 11px !important;
            border-width: 1px !important;
            font-size: 22.5px !important;
            font-weight: 560 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 7px !important;
        }

        .stButtonGroup button:nth-of-type(1)::before {
            content: "";
            width: 14px;
            height: 14px;
            display: inline-block;
            background: currentColor;
            -webkit-mask: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M4 5h7v7H4V5Zm9 0h7v7h-7V5ZM4 14h7v5H4v-5Zm9 0h7v5h-7v-5Z'/%3E%3C/svg%3E") center / contain no-repeat;
            mask: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M4 5h7v7H4V5Zm9 0h7v7h-7V5ZM4 14h7v5H4v-5Zm9 0h7v5h-7v-5Z'/%3E%3C/svg%3E") center / contain no-repeat;
        }

        .stButtonGroup button:nth-of-type(2)::before {
            content: "";
            width: 14px;
            height: 14px;
            display: inline-block;
            background: currentColor;
            -webkit-mask: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M4 7h10v2H4V7Zm0 8h16v2H4v-2Zm14-10a2 2 0 1 1 0 4 2 2 0 0 1 0-4ZM8 13a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z'/%3E%3C/svg%3E") center / contain no-repeat;
            mask: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M4 7h10v2H4V7Zm0 8h16v2H4v-2Zm14-10a2 2 0 1 1 0 4 2 2 0 0 1 0-4ZM8 13a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z'/%3E%3C/svg%3E") center / contain no-repeat;
        }

        div[data-testid="stVerticalBlock"][class*="st-key-sec_"] {
            background: #ffffff;
            border: 1px solid #dbe5f2 !important;
            border-left-width: 3px !important;
            border-radius: 16px !important;
            box-shadow: var(--shadow);
            overflow: hidden;
            padding: 14px !important;
            margin-bottom: 8px;
        }

        div[data-testid="stElementContainer"]:has(> div[data-testid="stVerticalBlock"][class*="st-key-sec_"]) {
            margin-bottom: 0 !important;
        }

        div[data-testid="stVerticalBlock"].st-key-sec_run_settings,
        div[data-testid="stVerticalBlock"].st-key-sec_traffic_geometry {
            border-left-color: #0f62fe !important;
        }

        div[data-testid="stVerticalBlock"].st-key-sec_scene_conditions,
        div[data-testid="stVerticalBlock"].st-key-sec_sensor_modules {
            border-left-color: #06b6d4 !important;
        }

        div[data-testid="stVerticalBlock"].st-key-sec_output,
        div[data-testid="stVerticalBlock"].st-key-sec_communication {
            border-left-color: #6366f1 !important;
        }

        div[data-testid="stVerticalBlock"].st-key-sec_run {
            border-left-color: #12b76a !important;
        }

        div[data-testid="stVerticalBlock"].st-key-sec_preview {
            border-left-color: #4f46e5 !important;
        }

        .section-shell {
            margin: -14px -14px 8px -14px;
            overflow: hidden;
        }

        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-bottom: 1px solid #edf2f7;
            border-left: 3px solid currentColor;
            height: 46px;
            min-height: 46px;
            max-height: 46px;
            overflow: hidden;
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 9px;
            height: 46px;
        }

        .section-icon {
            width: 23px;
            height: 23px;
            border-radius: 8px;
            background: #eef5ff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .section-icon svg {
            width: 14px;
            height: 14px;
        }

        .section-title h3 {
            margin: 0;
            font-size: 24.5px;
            font-weight: 560;
            color: #0b1f3a;
            letter-spacing: 0;
            line-height: 46px;
            display: flex;
            align-items: center;
        }

        .section-chevron {
            color: #667085;
            font-size: 23px;
        }

        .section-caption {
            color: var(--muted);
            font-size: 22.5px;
            margin: -4px 0 16px;
        }

        .inline-toggle-label {
            min-height: 39px;
            display: flex;
            align-items: center;
            color: #1f2937;
            font-size: 22.5px;
            font-weight: 520;
            line-height: 1.2;
        }

        .inline-field-label {
            min-height: 39px;
            display: flex;
            align-items: center;
            color: #1f2937;
            font-size: 22.5px;
            font-weight: 520;
            white-space: nowrap;
        }

        .field-label-compact {
            color: #1f2937;
            font-size: 22.5px;
            font-weight: 520;
            line-height: 1;
            margin-bottom: 6px;
            text-align: center;
            height: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        div[data-testid="stVerticalBlock"].st-key-sec_scene_conditions
        > div:nth-of-type(2)
        > div[data-testid="stHorizontalBlock"]
        > div:nth-of-type(2) {
            transform: translateX(18px);
        }

        .index-pill {
            min-height: 39px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 0 12px;
            border: 1px solid #dbe5f2;
            border-radius: 999px;
            background: #f8fbff;
            color: #1f2937;
            font-size: 22.5px;
            font-weight: 560;
            white-space: nowrap;
        }

        .loaded-pill {
            min-height: 39px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            padding: 0 12px;
            border: 1px solid #cfeedd;
            border-radius: 999px;
            background: #f0fdf5;
            color: #087443;
            font-size: 22.5px;
            font-weight: 560;
            white-space: nowrap;
        }

        .run-status {
            border-radius: 12px;
            border: 1px solid #dbe5f2;
            background: #f8fbff;
            padding: 0 14px;
            min-height: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            color: #1f2937;
            font-size: 23px;
            font-weight: 560;
        }

        .metric-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 14px;
            padding: 8px 0;
            border-bottom: 1px dashed #e4eaf3;
            font-size: 22px;
        }

        div[data-testid="stVerticalBlock"][class*="st-key-preview_panel_"] {
            border: 1px solid #e3eaf3 !important;
            border-radius: 14px !important;
            padding: 8px !important;
            box-shadow: none;
            background: #ffffff;
            min-height: 210px;
        }

        .preview-panel-title {
            color: #0b1f3a;
            font-size: 23px;
            font-weight: 560;
            text-align: center;
            padding-bottom: 6px;
            margin-bottom: 6px;
            border-bottom: 1px solid #edf2f7;
        }

        div[data-testid="stVerticalBlock"][class*="st-key-preview_panel_"] img,
        div[data-testid="stVerticalBlock"][class*="st-key-preview_panel_"] canvas {
            width: 100% !important;
            max-height: 190px;
            object-fit: contain;
        }

        div[data-testid="stVerticalBlock"][class*="st-key-preview_panel_"] [data-testid="stImage"],
        div[data-testid="stVerticalBlock"][class*="st-key-preview_panel_"] [data-testid="stPyplot"] {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        div[data-testid="stVerticalBlock"].st-key-preview_panel_image img {
            width: 100% !important;
            height: 178px !important;
            object-fit: cover !important;
            border-radius: 8px;
        }

        .metric-row span:first-child {
            color: var(--muted);
        }

        .metric-row span:last-child {
            color: #0f62fe;
            font-weight: 560;
            text-align: right;
        }

        .preview-label-card {
            border: 1px solid #dbe5f2;
            border-radius: 14px;
            background: #ffffff;
            padding: 8px;
            min-height: 0;
            box-sizing: border-box;
            width: 100%;
            max-width: 100%;
            overflow: hidden;
        }

        div[data-testid="stVerticalBlock"].st-key-preview_panel_label {
            min-height: 0 !important;
        }

        div[data-testid="stVerticalBlock"].st-key-sec_preview div[data-testid="stHorizontalBlock"] {
            gap: 0 !important;
        }

        .label-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            width: 100%;
            box-sizing: border-box;
        }

        .label-value {
            margin-top: 0;
            padding: 10px;
            border-radius: 12px;
            background: #f8fbff;
            border: 1px solid #e5edf7;
            color: #0b1f3a;
            font-size: 22.5px;
            line-height: 1.55;
            text-align: center;
            min-width: 0;
            overflow-wrap: anywhere;
            box-sizing: border-box;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f62fe, #0950d9);
            border: 1px solid #0f62fe;
            border-radius: 10px;
            color: #ffffff;
            font-weight: 560;
            min-height: 42px;
            box-shadow: 0 6px 14px rgba(15, 98, 254, 0.18);
        }

        .stButton > button[kind="secondary"],
        .stDownloadButton > button {
            background: #ffffff;
            border: 1px solid #c9d7ec;
            border-radius: 10px;
            color: #0f62fe;
            font-weight: 560;
            min-height: 42px;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #0f62fe;
            box-shadow: 0 8px 18px rgba(15, 98, 254, 0.14);
        }

        label, .stTextInput label, .stNumberInput label, .stSelectbox label,
        .stMultiSelect label, .stRadio label {
            color: #1f2937 !important;
            font-size: 21px !important;
            font-weight: 760 !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] > div {
            min-height: 41px;
            border-radius: 9px;
            background: #ffffff !important;
            border-color: #dbe5f2 !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span,
        .stRadio div[role="radiogroup"] label span,
        .stCheckbox label span,
        [data-testid="stMarkdownContainer"] p {
            font-size: 21.5px !important;
        }

        .stNumberInput button {
            font-size: 23px !important;
        }

        div[data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid #edf2f7;
            margin-bottom: 12px;
        }

        button[data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 8px 16px;
            font-weight: 780;
        }

        .stSlider [data-testid="stTickBar"] {
            display: none;
        }

        @media (max-width: 1100px) {
            .top-header {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 18px;
                padding-right: 18px;
            }

            [data-testid="stSidebar"] {
                min-width: 190px !important;
                max-width: 190px !important;
            }

            .page-title {
                font-size: 28px;
            }

        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(meta_key: str, title: str | None = None) -> None:
    default_title, icon_name, accent = SECTION_META[meta_key]
    title = title or default_title
    st.markdown(
        f"""
        <div class="section-shell" style="color:{accent};">
          <div class="section-header">
            <div class="section-title">
              <span class="section-icon" style="color:{accent};">{svg_icon(icon_name)}</span>
              <h3>{title}</h3>
            </div>
            <span class="section-chevron">⌄</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def section(meta_key: str, title: str | None = None):
    with st.container(border=True, key=f"sec_{meta_key}"):
        section_header(meta_key, title)
        yield


def render_sidebar_nav(advanced: bool) -> None:
    brand_uri = image_data_uri(BRAND_ICON)
    brand_img = f'<img src="{brand_uri}" alt="">' if brand_uri else svg_icon("radio")
    if advanced:
        items = [
            ("Run Settings", "settings"),
            ("Traffic Geometry", "map"),
            ("Sensor Modules", "sensor"),
            ("Communication", "radio"),
            ("Output", "folder"),
            ("Run", "play"),
            ("Preview", "eye"),
        ]
    else:
        items = [
            ("Run Settings", "settings"),
            ("Scene Conditions", "cloud"),
            ("Output", "folder"),
            ("Run", "play"),
            ("Preview", "eye"),
        ]
    item_html = "".join(
        f'<div class="sidebar-item {"active" if i == 0 else ""}">{svg_icon(icon)}<span>{label}</span></div>'
        for i, (label, icon) in enumerate(items)
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            {brand_img}
            <div><strong>V2X-SenseComm</strong><span>Local Generator</span></div>
        </div>
        {item_html}
        """,
        unsafe_allow_html=True,
    )


def render_header() -> str:
    if "generation_state" not in st.session_state:
        st.session_state["generation_state"] = "Idle"

    brand_uri = image_data_uri(BRAND_ICON)
    brand_img = f'<img src="{brand_uri}" alt="V2X-SenseComm logo">' if brand_uri else svg_icon("radio")
    left, mode_col = st.columns([0.58, 0.42], gap="small")
    with left:
        st.markdown(
            f"""
            <div class="brand-row">
                {brand_img}
                <div>
                    <div class="page-title">V2X-SenseComm Data Generation Platform</div>
                    <p class="page-subtitle">Local platform for synchronized V2X sensing-communication data generation.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mode_col:
        if hasattr(st, "segmented_control"):
            mode = st.segmented_control(
                "Configuration Mode",
                ["Basic Mode", "Advanced Mode"],
                default="Basic Mode",
                label_visibility="collapsed",
            )
        else:
            mode = st.radio(
                "Configuration Mode",
                ["Basic Mode", "Advanced Mode"],
                horizontal=True,
                label_visibility="collapsed",
            )
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)
    return mode


def clone_defaults() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def parse_speed_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values or [25.0]


def estimate_frames(config: dict) -> int:
    run = config["run"]
    speed = abs(float(run["speed_list"][0])) if run["speed_list"] else 0.0
    interval = float(run["frame_interval_s"])
    if speed <= 0 or interval <= 0:
        return 0
    return max(1, int(float(run["total_distance"]) / (speed * interval)) + 1)


def get_actual_frame_count(config: dict) -> int | None:
    output_root = resolve_project_path(config["run"]["output_root"])
    index_path = output_root / "index.csv"
    if not index_path.exists():
        return None
    try:
        df = pd.read_csv(index_path)
    except Exception:
        return None
    if df.empty:
        return None
    if config["run"]["scene_ids"]:
        df = df[df["scene_id"] == config["run"]["scene_ids"][0]]
    if config["run"]["speed_list"]:
        run_id = f"speed_{float(config['run']['speed_list'][0]):+.0f}"
        df = df[df["run_id"] == run_id]
    return None if df.empty else len(df)


def write_config(config: dict) -> str:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
    GENERATED_CONFIG.write_text(text, encoding="utf-8")
    return text


def run_generation_subprocess(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run" / "run_from_config.py"), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def render_metrics(config: dict, actual_frames: int | None = None) -> None:
    run = config["run"]
    frames = actual_frames if actual_frames is not None else estimate_frames(config)
    rows = [
        ("Selected scenes", ", ".join(run["scene_ids"])),
        ("Speed list", ", ".join(f"{float(v):g}" for v in run["speed_list"])),
        ("Estimated frames", f"{frames:,}"),
        ("Output root", str(run["output_root"])),
        ("Weather", str(run["weather_type"])),
    ]
    html = "".join(f'<div class="metric-row"><span>{k}</span><span>{v}</span></div>' for k, v in rows)
    st.markdown(html, unsafe_allow_html=True)


def render_run_settings(config: dict, scene_options: list[str]) -> tuple[list[str], str, float, float]:
    default_scenes = [s for s in config["run"]["scene_ids"] if s in scene_options] or scene_options[:1]
    with section("run_settings"):
        cols = st.columns(4, gap="medium")
        scene_ids = cols[0].multiselect("Scene", scene_options, default=default_scenes)
        speed_raw = cols[1].text_input(
            "Speed List (m/s)",
            value=", ".join(f"{v:g}" for v in config["run"]["speed_list"]),
            help="Use commas for multiple runs, e.g. 5, 15, 25.",
        )
        total_distance = cols[2].number_input(
            "Total Distance (m)",
            min_value=1.0,
            value=float(config["run"]["total_distance"]),
            step=10.0,
        )
        frame_interval = cols[3].number_input(
            "Frame Interval (s)",
            min_value=0.01,
            value=float(config["run"]["frame_interval_s"]),
            step=0.01,
            format="%.2f",
        )
    return scene_ids or default_scenes, speed_raw, float(total_distance), float(frame_interval)


def render_scene_conditions(config: dict) -> None:
    with section("scene_conditions"):
        cols = st.columns(3, gap="medium")
        config["run"]["weather_type"] = cols[0].selectbox(
            "Weather Type",
            list(WEATHER_TYPES),
            index=list(WEATHER_TYPES).index(config["run"]["weather_type"]),
        )
        config["run"]["motion_axis"] = cols[1].radio(
            "Motion Axis",
            ["x", "y"],
            index=1 if config["run"]["motion_axis"] == "y" else 0,
            horizontal=True,
        )
        with cols[2]:
            st.markdown(
                '<div class="field-label-compact">Occluder Cars</div>',
                unsafe_allow_html=True,
            )
            toggle_cols = st.columns([0.38, 0.24, 0.38], gap="small")
            config["run"]["enable_occluder_cars"] = toggle_cols[1].toggle(
                "Occluder Cars",
                value=bool(config["run"]["enable_occluder_cars"]),
                label_visibility="collapsed",
            )


def render_output(config: dict, compact: bool = False) -> None:
    with section("output"):
        if compact:
            root_cols = st.columns([0.16, 0.84], gap="small")
            root_cols[0].markdown(
                '<div class="inline-field-label">Output Root</div>',
                unsafe_allow_html=True,
            )
            config["run"]["output_root"] = root_cols[1].text_input(
                "Output Root",
                value=str(config["run"]["output_root"]),
                label_visibility="collapsed",
            )
        else:
            config["run"]["output_root"] = st.text_input(
                "Output Root",
                value=str(config["run"]["output_root"]),
            )


def render_traffic_geometry(config: dict) -> None:
    with section("traffic_geometry"):
        cols = st.columns(5, gap="medium")
        config["traffic"]["road_center_x"] = cols[0].number_input(
            "road_center_x (m)", value=float(config["traffic"]["road_center_x"]), step=1.0
        )
        config["traffic"]["road_center_y"] = cols[1].number_input(
            "road_center_y (m)", value=float(config["traffic"]["road_center_y"]), step=1.0
        )
        config["traffic"]["y_extent"] = cols[2].number_input(
            "y_extent (m)", value=float(config["traffic"]["y_extent"]), step=5.0
        )
        config["traffic"]["x_min"] = cols[3].number_input(
            "x_min (m)", value=float(config["traffic"]["x_min"]), step=5.0
        )
        config["traffic"]["x_max"] = cols[4].number_input(
            "x_max (m)", value=float(config["traffic"]["x_max"]), step=5.0
        )


def render_sensor_modules(config: dict) -> None:
    with section("sensor_modules"):
        active_sensor = st.radio(
            "Sensor Type",
            ["Camera", "LiDAR", "Radar", "GPS"],
            index=2,
            horizontal=True,
            label_visibility="collapsed",
        )
        if active_sensor == "Camera":
            cols = st.columns(4, gap="medium")
            config["sensors"]["camera"]["fov"] = cols[0].number_input(
                "FOV", value=float(config["sensors"]["camera"]["fov"]), step=5.0
            )
            width = cols[1].number_input("Width", value=int(config["sensors"]["camera"]["resolution"][0]), step=64)
            height = cols[2].number_input("Height", value=int(config["sensors"]["camera"]["resolution"][1]), step=64)
            config["sensors"]["camera"]["num_samples"] = cols[3].number_input(
                "Render Samples", value=int(config["sensors"]["camera"]["num_samples"]), step=16
            )
            config["sensors"]["camera"]["resolution"] = [int(width), int(height)]
        elif active_sensor == "LiDAR":
            cols = st.columns(3, gap="medium")
            config["sensors"]["lidar"]["num_channels"] = cols[0].number_input(
                "Channels", value=int(config["sensors"]["lidar"]["num_channels"]), step=16
            )
            config["sensors"]["lidar"]["h_resolution"] = cols[1].number_input(
                "Horizontal Resolution", value=int(config["sensors"]["lidar"]["h_resolution"]), step=32
            )
            config["sensors"]["lidar"]["max_range"] = cols[2].number_input(
                "Max Range (m)", value=float(config["sensors"]["lidar"]["max_range"]), step=10.0
            )
        elif active_sensor == "Radar":
            cols = st.columns(7, gap="medium")
            config["sensors"]["radar"]["array_rows"] = cols[0].number_input(
                "Array Rows", value=int(config["sensors"]["radar"]["array_rows"]), step=1
            )
            config["sensors"]["radar"]["array_cols"] = cols[1].number_input(
                "Array Cols", value=int(config["sensors"]["radar"]["array_cols"]), step=1
            )
            carrier_ghz = cols[2].number_input(
                "Carrier (GHz)", value=float(config["sensors"]["radar"]["carrier_frequency"]) / 1e9, step=1.0
            )
            bandwidth_mhz = cols[3].number_input(
                "Bandwidth (MHz)", value=float(config["sensors"]["radar"]["bandwidth"]) / 1e6, step=10.0
            )
            config["sensors"]["radar"]["num_chirps"] = cols[4].number_input(
                "Num Chirps", value=int(config["sensors"]["radar"]["num_chirps"]), step=1
            )
            config["sensors"]["radar"]["num_samples"] = cols[5].number_input(
                "Num Samples", value=int(config["sensors"]["radar"]["num_samples"]), step=32
            )
            noise_options = ["None"] + [str(value) for value in range(-20, 25, 5)]
            current_noise = config["sensors"]["radar"].get("noise_snr_db")
            current_noise_label = "None" if current_noise is None else f"{float(current_noise):g}"
            if current_noise_label not in noise_options:
                noise_options.append(current_noise_label)
            noise_label = cols[6].selectbox(
                "Noise SNR (dB)",
                noise_options,
                index=noise_options.index(current_noise_label),
            )
            config["sensors"]["radar"]["carrier_frequency"] = float(carrier_ghz) * 1e9
            config["sensors"]["radar"]["bandwidth"] = float(bandwidth_mhz) * 1e6
            config["sensors"]["radar"]["noise_snr_db"] = None if noise_label == "None" else float(noise_label)
        else:
            cols = st.columns(3, gap="medium")
            noise = config["sensors"]["gps"]["noise_std"]
            config["sensors"]["gps"]["noise_std"] = [
                cols[0].number_input("Noise X (m)", value=float(noise[0]), step=0.1),
                cols[1].number_input("Noise Y (m)", value=float(noise[1]), step=0.1),
                cols[2].number_input("Noise Z (m)", value=float(noise[2]), step=0.1),
            ]


def render_communication(config: dict) -> None:
    with section("communication"):
        comm = config["communication"]
        cols = st.columns(5, gap="medium")
        carrier_ghz = cols[0].number_input("Carrier (GHz)", value=float(comm["carrier_frequency"]) / 1e9, step=0.5)
        comm["bs_tx_num_rows"] = cols[1].number_input("BS TX Rows", value=int(comm["bs_tx_num_rows"]), step=1)
        comm["bs_tx_num_cols"] = cols[2].number_input("BS TX Cols", value=int(comm["bs_tx_num_cols"]), step=1)
        comm["num_subcarriers"] = cols[3].number_input("Subcarriers", value=int(comm["num_subcarriers"]), step=16)
        comm["top_k"] = cols[4].number_input("Top-k", value=int(comm["top_k"]), step=1)
        comm["carrier_frequency"] = float(carrier_ghz) * 1e9
        comm["codebook_file"] = st.text_input("Codebook File", value=str(comm["codebook_file"]))


def render_run_section(config: dict, config_yaml: str, actual_frames: int | None) -> bool:
    with section("run"):
        cols = st.columns([0.22, 0.22, 0.56], gap="medium")
        run_clicked = cols[0].button("Run Generation", type="primary", use_container_width=True)
        cols[1].download_button(
            "Export YAML",
            data=config_yaml,
            file_name="generation_config.yaml",
            mime="text/yaml",
            use_container_width=True,
        )
        state = st.session_state.get("generation_state", "Idle")
        status_text = {
            "Idle": "Ready to generate",
            "Running": "Running",
            "Done": "Completed",
            "Failed": "Failed",
        }.get(state, state)
        cols[2].markdown(f'<div class="run-status">● &nbsp; {status_text}</div>', unsafe_allow_html=True)
    return run_clicked


def build_config_from_ui(advanced: bool) -> dict:
    config = clone_defaults()
    scene_options = sorted(discover_scene_map().keys())

    scene_ids, speed_raw, total_distance, frame_interval = render_run_settings(config, scene_options)
    config["run"]["scene_ids"] = scene_ids
    config["run"]["speed_list"] = parse_speed_list(speed_raw)
    config["run"]["total_distance"] = total_distance
    config["run"]["frame_interval_s"] = frame_interval

    if advanced:
        render_traffic_geometry(config)
        render_sensor_modules(config)
        render_communication(config)
        render_output(config, compact=True)
    else:
        cond_col, out_col = st.columns([0.58, 0.42], gap="medium")
        with cond_col:
            render_scene_conditions(config)
        with out_col:
            render_output(config)
    return config


def make_lidar_figure(path: Path):
    data = np.load(path)
    xyz = data["xyz"]
    distance = data["distance"] if "distance" in data else np.linalg.norm(xyz, axis=1)
    fig, ax = plt.subplots(figsize=(5.0, 3.25), dpi=120)
    ax.scatter(xyz[:, 1], xyz[:, 0], c=distance, s=3.2, cmap="viridis", linewidths=0)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-25, 100)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def build_radar_fft_angle_spectrum(radar_cube, num_rows=4, num_cols=8, fft_az=128, fft_el=32):
    radar_cube = np.asarray(radar_cube)
    num_chirps, num_antennas, num_samples = radar_cube.shape
    if num_antennas != num_rows * num_cols:
        side = int(np.sqrt(num_antennas))
        num_rows = max(1, side)
        num_cols = max(1, num_antennas // num_rows)
        radar_cube = radar_cube[:, : num_rows * num_cols, :]
    radar_4d = radar_cube.reshape(num_chirps, num_cols, num_rows, num_samples)
    radar_avg_over_range = np.mean(radar_4d, axis=3)
    radar_snapshot = np.mean(radar_avg_over_range, axis=0)
    angle_spectrum = np.fft.fftshift(np.fft.fft2(radar_snapshot, s=(fft_az, fft_el)))
    angle_spectrum = np.abs(angle_spectrum)
    angle_spectrum = angle_spectrum / (np.max(angle_spectrum) + 1e-12)
    angle_spectrum = np.flip(angle_spectrum, axis=1)
    return angle_spectrum


def make_radar_figure(path: Path):
    data = np.load(path)
    radar_cube = data["radar_cube"]
    spectrum = build_radar_fft_angle_spectrum(radar_cube)
    fig, ax = plt.subplots(figsize=(5.0, 3.25), dpi=120)
    ax.imshow(
        20 * np.log10(spectrum.T + 1e-6),
        extent=[-1.0, 1.0, -1.0, 1.0],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=-60,
        vmax=0,
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def make_channel_figure(channel_path: Path):
    data = np.load(channel_path)
    H = data["H"]
    flat = np.asarray(H).reshape(-1)
    target = 64 * 16 * 128
    if flat.size >= target:
        channel = flat[:target].reshape(64, 16, 128)
        image = np.abs(np.sum(channel, axis=2)).T
    else:
        image = np.abs(np.asarray(H).squeeze())
        if image.ndim > 2:
            image = image.reshape(image.shape[0], -1)
    fig, ax = plt.subplots(figsize=(5.0, 3.25), dpi=120)
    ax.imshow(image, origin="lower", aspect="auto", cmap="viridis")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def make_camera_preview_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).copy()
    dark_mask = np.all(arr < 8, axis=2)
    arr[dark_mask] = 255
    image = Image.fromarray(arr)

    width, height = image.size
    target_ratio = 1.55
    target_height = min(height, int(width / target_ratio))
    overflow = max(0, height - target_height)
    top = int(overflow * 0.58)
    return image.crop((0, top, width, top + target_height))


def resolve_sample_path(output_root: Path, path_value: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    dataset_relative = output_root / path
    if dataset_relative.exists():
        return dataset_relative
    return resolve_project_path(path)


def read_label_top1(row: pd.Series, output_root: Path) -> tuple[str, str]:
    label_path = resolve_sample_path(output_root, str(row.get("beam_label_path", "")))
    if not label_path.exists():
        return "N/A", "N/A"
    try:
        label = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception:
        return "N/A", "N/A"

    top_indices = label.get("topk_indices")
    top_powers = label.get("topk_powers")
    if top_indices and top_powers:
        return str(top_indices[0]), f"{float(top_powers[0]):.3e}"
    return str(label.get("best_beam_index_3d", "N/A")), f"{float(label.get('best_power', 0.0)):.3e}"


def read_gps_measured(row: pd.Series, output_root: Path) -> str:
    gps_path = resolve_sample_path(output_root, str(row.get("gps_path", "")))
    if not gps_path.exists():
        return "N/A"
    try:
        gps = np.load(gps_path)
        measured = gps.get("measured_position")
        if measured is None:
            measured = gps.get("true_position")
        if measured is None:
            return "N/A"
        values = np.asarray(measured).reshape(-1)[:3]
        return "[" + ", ".join(f"{float(v):.2f}" for v in values) + "] m"
    except Exception:
        return "N/A"


def render_preview(config: dict) -> None:
    output_root = resolve_project_path(config["run"]["output_root"])
    index_path = output_root / "index.csv"

    with section("preview"):
        if not index_path.exists():
            st.info("No preview data available. Run generation first.")
            return

        df = pd.read_csv(index_path)
        if df.empty:
            st.info("No preview data available. Run generation first.")
            return

        control_cols = st.columns([0.18, 0.18, 0.39, 0.13, 0.12], gap="medium")
        scenes = sorted(df["scene_id"].dropna().unique().tolist())
        scene = control_cols[0].selectbox("Scene", scenes, index=0)
        scene_df = df[df["scene_id"] == scene]
        runs = sorted(scene_df["run_id"].dropna().unique().tolist())
        run_id = control_cols[1].selectbox("Run ID", runs, index=0)
        run_df = scene_df[scene_df["run_id"] == run_id].reset_index(drop=True)
        total = len(run_df)
        frame_pos = control_cols[2].slider("Sample Index", 0, max(total - 1, 0), 0, format="%d")
        control_cols[3].markdown(
            f'<div class="index-pill">Sample {frame_pos + 1:,} / {total:,}</div>',
            unsafe_allow_html=True,
        )
        control_cols[4].markdown(
            '<div class="loaded-pill">● Loaded</div>',
            unsafe_allow_html=True,
        )

        row = run_df.iloc[frame_pos]
        panels = st.columns(4, gap="small")

        with panels[0].container(border=True, key="preview_panel_image"):
            st.markdown('<div class="preview-panel-title">Image</div>', unsafe_allow_html=True)
            img_path = resolve_sample_path(output_root, str(row["camera_path"]))
            if img_path.exists():
                st.image(make_camera_preview_image(img_path), use_container_width=True)
            else:
                st.warning("Image file not found.")

        with panels[1].container(border=True, key="preview_panel_lidar"):
            st.markdown('<div class="preview-panel-title">LiDAR</div>', unsafe_allow_html=True)
            lidar_path = resolve_sample_path(output_root, str(row["lidar_path"]))
            if lidar_path.exists():
                fig = make_lidar_figure(lidar_path)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning("LiDAR file not found.")

        with panels[2].container(border=True, key="preview_panel_radar"):
            st.markdown('<div class="preview-panel-title">Radar</div>', unsafe_allow_html=True)
            radar_path = resolve_sample_path(output_root, str(row["radar_path"]))
            if radar_path.exists():
                fig = make_radar_figure(radar_path)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning("Radar file not found.")

        with panels[3].container(border=True, key="preview_panel_channel"):
            st.markdown('<div class="preview-panel-title">Channel</div>', unsafe_allow_html=True)
            channel_path = resolve_sample_path(output_root, str(row["channel_path"]))
            if channel_path.exists():
                fig = make_channel_figure(channel_path)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning("Channel file not found.")

        top1_index, top1_power = read_label_top1(row, output_root)
        gps_measured = read_gps_measured(row, output_root)
        st.markdown(
            f"""
            <div class="preview-label-card">
                <div class="label-grid">
                    <div class="label-value">
                        <strong>GPS</strong><br>{gps_measured}
                    </div>
                    <div class="label-value">
                        <strong>Top-1 Index</strong><br>{top1_index}
                    </div>
                    <div class="label-value">
                        <strong>Top-1 Power</strong><br>{top1_power}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_theme()
    mode = render_header()
    advanced = mode == "Advanced Mode"
    render_sidebar_nav(advanced)

    config = build_config_from_ui(advanced)
    config_yaml = write_config(config)
    actual_frames = get_actual_frame_count(config)

    run_clicked = render_run_section(config, config_yaml, actual_frames)

    if run_clicked:
        st.session_state["generation_state"] = "Running"
        with st.status("Running generation pipeline...", expanded=True) as status:
            result = run_generation_subprocess(GENERATED_CONFIG)
            if result.stdout:
                st.code(result.stdout[-6000:], language="text")
            if result.stderr:
                st.code(result.stderr[-6000:], language="text")
            if result.returncode == 0:
                st.session_state["generation_state"] = "Done"
                status.update(label="Generation finished", state="complete")
            else:
                st.session_state["generation_state"] = "Failed"
                status.update(label=f"Generation failed: exit code {result.returncode}", state="error")

    render_preview(config)


if __name__ == "__main__":
    main()
