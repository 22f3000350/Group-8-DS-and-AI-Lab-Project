"""
theme.py
--------
Custom Gradio theme for the Crop Disease AI Assistant.
Compatible with Gradio 3.x and 4.x.

Primary brand color : #1D9E75  (teal-green)
Accent / dark       : #0F6E56
Surface             : #F5F5F3  (warm off-white)
Border              : #E0E0DA
Text primary        : #1A1A18
Text secondary      : #80807A
"""

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes


# ── Custom colour ramps ────────────────────────────────────────────────────────

_green = colors.Color(
    name="green",
    c50="#E1F5EE",
    c100="#9FE1CB",
    c200="#5DCAA5",
    c300="#2DB88A",
    c400="#1D9E75",
    c500="#189E71",
    c600="#0F6E56",
    c700="#085041",
    c800="#04342C",
    c900="#022018",
    c950="#011510",
)

_neutral = colors.Color(
    name="neutral",
    c50="#F5F5F3",
    c100="#EAEAE6",
    c200="#CFCFC8",
    c300="#B4B2A9",
    c400="#888780",
    c500="#6E6D67",
    c600="#5F5E5A",
    c700="#444441",
    c800="#2C2C2A",
    c900="#1A1A18",
    c950="#0D0D0C",
)


# ── Theme builder ──────────────────────────────────────────────────────────────

def get_theme() -> gr.themes.Base:
    """
    Returns a Gradio Base theme styled for the Crop Disease AI app.

    Usage:
        from theme import get_theme
        demo = gr.Blocks(theme=get_theme(), ...)
    """

    theme = gr.themes.Base(
        font=[
            fonts.GoogleFont("DM Sans"),
            fonts.Font("ui-sans-serif"),
            fonts.Font("system-ui"),
            fonts.Font("sans-serif"),
        ],
        font_mono=[
            fonts.GoogleFont("DM Mono"),
            fonts.Font("ui-monospace"),
            fonts.Font("monospace"),
        ],
        primary_hue=_green,
        secondary_hue=_green,
        neutral_hue=_neutral,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_md,
        text_size=sizes.text_md,
    )

    # Build kwargs carefully — only include a property if it exists in this
    # Gradio installation to avoid "unexpected keyword argument" crashes.
    safe_kwargs = {
        # Page
        "body_background_fill":            "#F5F5F3",
        "body_background_fill_dark":       "#1A1A18",
        "body_text_color":                 "#1A1A18",
        "body_text_color_dark":            "#E8E8E2",
        "body_text_size":                  "14px",
        "body_text_weight":                "400",

        # Blocks / panels
        "block_background_fill":           "#FFFFFF",
        "block_background_fill_dark":      "#242422",
        "block_border_color":              "#E0E0DA",
        "block_border_color_dark":         "#3A3A38",
        "block_border_width":              "0.5px",
        "block_label_background_fill":     "#F5F5F3",
        "block_label_background_fill_dark":"#2C2C2A",
        "block_label_text_color":          "#80807A",
        "block_label_text_color_dark":     "#A0A099",
        "block_label_text_size":           "11px",
        "block_label_text_weight":         "600",
        "block_padding":                   "12px 14px",
        "block_radius":                    "12px",
        "block_shadow":                    "none",
        "block_title_text_color":          "#1A1A18",
        "block_title_text_color_dark":     "#E8E8E2",
        "block_title_text_size":           "13px",
        "block_title_text_weight":         "500",

        # Inputs
        "input_background_fill":           "#F5F5F3",
        "input_background_fill_dark":      "#2C2C2A",
        "input_background_fill_focus":     "#FFFFFF",
        "input_background_fill_focus_dark":"#333330",
        "input_border_color":              "#CFCFC8",
        "input_border_color_dark":         "#4A4A47",
        "input_border_color_focus":        "#1D9E75",
        "input_border_color_focus_dark":   "#2DB88A",
        "input_border_color_hover":        "#B4B2A9",
        "input_border_color_hover_dark":   "#5F5E5A",
        "input_border_width":              "0.5px",
        "input_placeholder_color":         "#B4B2A9",
        "input_placeholder_color_dark":    "#5F5E5A",
        "input_radius":                    "10px",
        "input_shadow":                    "none",
        "input_shadow_focus":              "0 0 0 2px #E1F5EE",
        "input_text_size":                 "13px",
        "input_text_weight":               "400",

        # Buttons — primary
        "button_primary_background_fill":             "#1D9E75",
        "button_primary_background_fill_dark":        "#1D9E75",
        "button_primary_background_fill_hover":       "#0F6E56",
        "button_primary_background_fill_hover_dark":  "#0F6E56",
        "button_primary_border_color":                "#1D9E75",
        "button_primary_border_color_dark":           "#1D9E75",
        "button_primary_border_color_hover":          "#0F6E56",
        "button_primary_text_color":                  "#FFFFFF",
        "button_primary_text_color_dark":             "#FFFFFF",
        "button_primary_text_color_hover":            "#FFFFFF",

        # Buttons — secondary
        "button_secondary_background_fill":            "#F5F5F3",
        "button_secondary_background_fill_dark":       "#2C2C2A",
        "button_secondary_background_fill_hover":      "#E1F5EE",
        "button_secondary_background_fill_hover_dark": "#1D3B2F",
        "button_secondary_border_color":               "#CFCFC8",
        "button_secondary_border_color_dark":          "#4A4A47",
        "button_secondary_border_color_hover":         "#1D9E75",
        "button_secondary_border_color_hover_dark":    "#2DB88A",
        "button_secondary_text_color":                 "#444441",
        "button_secondary_text_color_dark":            "#C2C0B6",
        "button_secondary_text_color_hover":           "#0F6E56",
        "button_secondary_text_color_hover_dark":      "#2DB88A",

        # Buttons — cancel
        "button_cancel_background_fill":      "#FCEBEB",
        "button_cancel_background_fill_dark": "#3D1F1F",
        "button_cancel_border_color":         "#F09595",
        "button_cancel_text_color":           "#A32D2D",

        # Buttons — shared geometry
        "button_border_width":       "0.5px",
        "button_large_padding":      "10px 20px",
        "button_large_radius":       "10px",
        "button_large_text_size":    "14px",
        "button_large_text_weight":  "500",
        "button_small_padding":      "5px 12px",
        "button_small_radius":       "8px",
        "button_small_text_size":    "12px",
        "button_small_text_weight":  "500",

        # Accent
        "color_accent":           "#1D9E75",
        "color_accent_soft":      "#E1F5EE",
        "color_accent_soft_dark": "#1D3B2F",

        # Links
        "link_text_color":         "#0F6E56",
        "link_text_color_dark":    "#2DB88A",
        "link_text_color_hover":   "#085041",
        "link_text_color_active":  "#0F6E56",
        "link_text_color_visited": "#0F6E56",

        # Layout
        "layout_gap":      "16px",
        "form_gap_width":  "8px",

        # Slider
        "slider_color":      "#1D9E75",
        "slider_color_dark": "#2DB88A",

        # Table
        "table_border_color": "#E0E0DA",
        "table_row_focus":    "#E1F5EE",
    }

    # Filter to only kwargs the installed Gradio version actually accepts
    import inspect
    valid = set(inspect.signature(theme.set).parameters.keys())
    filtered = {k: v for k, v in safe_kwargs.items() if k in valid}
    theme.set(**filtered)

    return theme