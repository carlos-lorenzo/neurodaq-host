"""Professional dark theme (slate/graphite + teal accent). Replaces Dracula."""
TOKENS = {
    "bg": "#161a20", "surface": "#1e242e", "raised": "#252c38",
    "border": "#2f3744", "text": "#e6e9ef", "muted": "#9aa3b2",
    "accent": "#2dd4bf", "accent_dim": "#14b8a6",
    "ok": "#34d399", "warn": "#fbbf24", "err": "#f87171",
    "info": "#60a5fa",
}
CH_COLORS = ["#2dd4bf", "#60a5fa", "#a78bfa", "#34d399",
             "#fbbf24", "#fb923c", "#f87171", "#e6e9ef"]

FONT = "'Inter', 'Segoe UI', system-ui, sans-serif"


def stylesheet() -> str:
    t = TOKENS
    return f"""
    QMainWindow, QWidget {{ background-color: {t['bg']}; color: {t['text']}; font-family: {FONT}; font-size: 13px; }}
    QGroupBox {{ font-weight: 600; border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px; padding: 12px; background: {t['surface']}; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {t['accent']}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
    QPushButton {{ background-color: {t['raised']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 7px 14px; font-weight: 600; }}
    QPushButton:hover {{ border-color: {t['accent']}; }}
    QPushButton:disabled {{ color: {t['muted']}; }}
    QPushButton#accent {{ background-color: {t['accent']}; color: #06281f; border: none; }}
    QPushButton#danger {{ background-color: {t['err']}; color: #2b0a0a; border: none; }}
    QPushButton#ok {{ background-color: {t['ok']}; color: #05281b; border: none; }}
    QLabel#h1 {{ font-size: 15px; font-weight: 700; }}
    QLabel#muted {{ color: {t['muted']}; }}
    QLabel#pill {{ border-radius: 10px; padding: 3px 10px; font-weight: 700; font-size: 12px; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background-color: {t['raised']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 5px 8px; }}
    QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: 0 8px 8px 8px; background: {t['bg']}; }}
    QTabBar::tab {{ background: transparent; color: {t['muted']}; padding: 9px 18px; margin-right: 2px; font-weight: 600; }}
    QTabBar::tab:selected {{ color: {t['accent']}; border-bottom: 2px solid {t['accent']}; }}
    QTableWidget, QTableView {{ background-color: {t['surface']}; gridline-color: {t['border']}; border: 1px solid {t['border']}; border-radius: 6px; }}
    QHeaderView::section {{ background-color: {t['raised']}; color: {t['text']}; font-weight: 600; border: none; border-bottom: 1px solid {t['border']}; padding: 6px; }}
    QCheckBox {{ spacing: 6px; }}
    QTextEdit {{ background-color: {t['surface']}; border: 1px solid {t['border']}; border-radius: 6px; color: {t['muted']}; }}
    """
