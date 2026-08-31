import pathlib

import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from src.ui_helpers import CSS_BLOCK, LOGIN_CSS, BRANDING_HTML

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Resume Sifter",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS_BLOCK, unsafe_allow_html=True)

# ── Authentication ─────────────────────────────────────────────────────────────
_CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"

if not _CONFIG_PATH.exists():
    st.error(
        "**config.yaml not found.** Run the following command to create the first user:\n\n"
        "```\npy -3.11 scripts/create_user.py --add hr_admin --password yourpassword "
        '--name "HR Admin" --email hr@company.com\n```'
    )
    st.stop()

with open(_CONFIG_PATH) as _f:
    _auth_config = yaml.load(_f, Loader=SafeLoader)

_authenticator = stauth.Authenticate(
    _auth_config["credentials"],
    _auth_config["cookie"]["name"],
    _auth_config["cookie"]["key"],
    _auth_config["cookie"]["expiry_days"],
)

# Slot appears ABOVE the login form; filled with branding panel when unauthenticated.
_login_brand_slot = st.empty()

_auth_result = _authenticator.login(location="main", fields={
    "Form name": "Sign in",
    "Username": "Username",
    "Password": "Password",
    "Login": "Sign in",
})
_auth_status = _auth_result[1] if _auth_result else st.session_state.get("authentication_status")

if not _auth_status:
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    _login_brand_slot.markdown(BRANDING_HTML, unsafe_allow_html=True)
    if _auth_status is False:
        st.error("Incorrect username or password. Please try again.")
    st.stop()

# ── Authenticated flow ─────────────────────────────────────────────────────────
_display_name = (
    st.session_state.get("name") or st.session_state.get("username") or "HR User"
)
_ready = bool(st.session_state.get("ready", False))

pg = st.navigation(
    [
        st.Page("pages/upload.py", title="Upload & Analyse", url_path="upload", default=True),
        st.Page("pages/results.py", title="Results", url_path="results"),
        st.Page("pages/bias.py", title="Bias Audit", url_path="bias"),
    ],
    position="hidden",
)

# ── Top navigation bar ─────────────────────────────────────────────────────────
_nb0, _nb1, _nb2, _nb3, _nbr = st.columns([2.8, 2.4, 1.2, 1.6, 2])
with _nb0:
    st.markdown('<div class="ars-brand-text">Auto Resume <em>Sifter</em></div>', unsafe_allow_html=True)
with _nb1:
    st.page_link("pages/upload.py", label="Upload & Analyse", use_container_width=True)
with _nb2:
    st.page_link("pages/results.py", label="Results", use_container_width=True, disabled=not _ready)
with _nb3:
    st.page_link("pages/bias.py", label="Bias Audit", use_container_width=True, disabled=not _ready)
with _nbr:
    _nri, _nro = st.columns([3, 2])
    with _nri:
        st.markdown(
            f'<div class="ars-user-info">Signed in as <strong>{_display_name}</strong></div>',
            unsafe_allow_html=True,
        )
    with _nro:
        _authenticator.logout("Sign out", location="main", key="nav_signout")
st.markdown('<hr class="ars-nav-divider">', unsafe_allow_html=True)

pg.run()
