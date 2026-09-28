import os
import joblib
import pandas as pd
import reflex as rx
from openai import AsyncOpenAI

# 1. Load trained inference pipeline
MODEL_PATH = "diabetes_pipeline.joblib"
pipeline = joblib.load(MODEL_PATH)

class State(rx.State):
    # Active View Tab: "predictor" or "analytics"
    active_tab: str = "predictor"

    # Patient Biomarkers
    pregnancies: float = 2.0
    glucose: float = 120.0
    blood_pressure: float = 72.0
    skin_thickness: float = 23.0
    insulin: float = 85.0
    bmi: float = 28.5
    dpf: float = 0.47
    age: float = 33.0

    # Risk Diagnostics
    risk_score: float = 28.4
    risk_tier: str = "Optimal Baseline"
    risk_color: str = "#10B981"
    risk_glow: str = "rgba(16, 185, 129, 0.45)"

    # AI Chatbox State
    chat_open: bool = False
    chat_input: str = ""
    messages: list[dict[str, str]] = [
        {
            "role": "assistant",
            "content": "Hello! I am your Endocrine AI clinical copilot. Ask me any question about your biomarkers, risk stratification, or diabetes prevention.",
        }
    ]

    # Navigation Handlers
    def set_predictor_tab(self):
        self.active_tab = "predictor"

    def set_analytics_tab(self):
        self.active_tab = "analytics"

    # Simulation Profile Handlers
    def load_profile_healthy(self):
        self.pregnancies = 1.0
        self.glucose = 105.0
        self.blood_pressure = 70.0
        self.skin_thickness = 20.0
        self.insulin = 75.0
        self.bmi = 23.5
        self.dpf = 0.32
        self.age = 29.0

    def load_profile_elevated(self):
        self.pregnancies = 3.0
        self.glucose = 175.0
        self.blood_pressure = 90.0
        self.skin_thickness = 38.0
        self.insulin = 220.0
        self.bmi = 38.5
        self.dpf = 1.15
        self.age = 58.0

    # Slider Value Handlers
    def set_val_pregnancies(self, val):
        self.pregnancies = float(val[0] if isinstance(val, list) else val)

    def set_val_glucose(self, val):
        self.glucose = float(val[0] if isinstance(val, list) else val)

    def set_val_blood_pressure(self, val):
        self.blood_pressure = float(val[0] if isinstance(val, list) else val)

    def set_val_skin_thickness(self, val):
        self.skin_thickness = float(val[0] if isinstance(val, list) else val)

    def set_val_insulin(self, val):
        self.insulin = float(val[0] if isinstance(val, list) else val)

    def set_val_bmi(self, val):
        num = float(val[0] if isinstance(val, list) else val)
        self.bmi = round(num, 1)

    def set_val_dpf(self, val):
        num = float(val[0] if isinstance(val, list) else val)
        self.dpf = round(num, 2)

    def set_val_age(self, val):
        self.age = float(val[0] if isinstance(val, list) else val)

    # ML Inference Handler
    def run_inference(self):
        cols = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
        sample = pd.DataFrame(
            [
                [
                    self.pregnancies,
                    self.glucose,
                    self.blood_pressure,
                    self.skin_thickness,
                    self.insulin,
                    self.bmi,
                    self.dpf,
                    self.age,
                ]
            ],
            columns=cols,
        )

        proba = pipeline.predict_proba(sample)[0][1]
        self.risk_score = round(float(proba) * 100, 1)

        if self.risk_score < 33.0:
            self.risk_tier = "Optimal Baseline"
            self.risk_color = "#10B981"
            self.risk_glow = "rgba(16, 185, 129, 0.45)"
        elif self.risk_score < 62.0:
            self.risk_tier = "Moderate Vigilance"
            self.risk_color = "#F59E0B"
            self.risk_glow = "rgba(245, 158, 11, 0.45)"
        else:
            self.risk_tier = "Elevated Diagnostic Risk"
            self.risk_color = "#EF4444"
            self.risk_glow = "rgba(239, 68, 68, 0.55)"

    # AI Chat Handlers
    def toggle_chat(self):
        self.chat_open = not self.chat_open

    def set_chat_input(self, val: str):
        self.chat_input = val

    async def send_chat_message(self):
        if not self.chat_input.strip():
            return

        user_query = self.chat_input.strip()
        self.chat_input = ""

        self.messages = [*self.messages, {"role": "user", "content": user_query}]

        system_context = (
            f"You are Endocrine AI, an empathetic, highly intelligent clinical assistant. "
            f"Current active patient readings: Glucose={self.glucose} mg/dL, BMI={self.bmi} kg/m², "
            f"Blood Pressure={self.blood_pressure} mm Hg, Insulin={self.insulin} μU/mL, Age={self.age}, "
            f"Diabetes Pedigree Function={self.dpf}, Model Risk Score={self.risk_score}% ({self.risk_tier}). "
            f"Provide direct, concise, and scientifically grounded answers in plain English for laypeople. "
            f"Remind the user this is educational guidance and risk stratification, not a formal prescription."
        )

        history = [{"role": "system", "content": system_context}] + [
            {"role": m["role"], "content": m["content"]} for m in self.messages
        ]

        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                self.messages = [*self.messages, {
                    "role": "assistant",
                    "content": "The AI chat is not configured yet. Add GROQ_API_KEY to the deployment environment.",
                }]
                return

            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=45.0,
            )
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=history,
                stream=False,
            )
            answer = response.choices[0].message.content or "I could not generate a response."
            self.messages = [*self.messages, {"role": "assistant", "content": answer}]
        except Exception as e:
            self.messages = [*self.messages, {
                "role": "assistant",
                "content": f"Unable to reach the AI service. Check GROQ_API_KEY and deployment logs. ({str(e)})",
            }]


# --- UI Helpers & Explanations ---

HELP_STRINGS = {
    "Glucose": "Blood sugar (2hr after a clinical glucose drink). Normal is < 140 mg/dL.",
    "BMI": "Body Mass Index (weight in kg / height in m²). Healthy range is 18.5 - 24.9.",
    "Age": "Patient age in completed years.",
    "Insulin": "Serum insulin hormone level in the bloodstream (μU/mL).",
    "Pressure": "Diastolic blood pressure: arterial pressure between heartbeats (mm Hg).",
    "DPF": "Diabetes Pedigree Function: clinical index scoring hereditary/genetic risk.",
    "Pregnancies": "Total count of pregnancies for the patient.",
}

# Base glass card style dictionary
APPLE_CARD_STYLE = {
    "background": "linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%)",
    "backdrop_filter": "blur(30px)",
    "-webkit-backdrop-filter": "blur(30px)",
    "border": "1px solid rgba(255, 255, 255, 0.08)",
    "border_radius": "24px",
    "box_shadow": "0 30px 60px -12px rgba(0, 0, 0, 0.65), inset 0 1px 0 0 rgba(255, 255, 255, 0.1)",
    "transition": "all 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
}


def glass_card(*children, **props):
    # Safely merge default styles with any overrides in props to prevent duplicate argument errors
    style = {**APPLE_CARD_STYLE, "padding": "32px"}
    style.update(props)
    return rx.box(*children, **style)


def info_tooltip(param_key: str) -> rx.Component:
    help_text = HELP_STRINGS.get(param_key, "Clinical measurement context.")
    return rx.tooltip(
        rx.button(
            rx.icon("help-circle", size=14, color="#64748B"),
            variant="ghost",
            cursor="help",
            padding="2px",
            height="auto",
            margin_left="6px",
        ),
        content=help_text,
        delay_duration=200,
    )


def control_slider(label: str, unit: str, val_var: rx.Var, handler, min_v, max_v, help_key: str, step_v=1.0):
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.text(label, font_size="13px", font_weight="500", color="#94A3B8", letter_spacing="0.3px"),
                info_tooltip(help_key),
                spacing="0",
                align="center",
            ),
            rx.spacer(),
            rx.box(
                rx.text(f"{val_var} {unit}", font_size="12px", font_weight="600", color="#E2E8F0"),
                padding="3px 10px",
                border_radius="100px",
                background="rgba(255, 255, 255, 0.06)",
                border="1px solid rgba(255, 255, 255, 0.08)",
            ),
            width="100%",
            align="center",
        ),
        rx.slider(
            value=[val_var],
            min=min_v,
            max=max_v,
            step=step_v,
            on_change=handler,
            width="100%",
            color_scheme="teal",
        ),
        spacing="2",
        width="100%",
    )


# --- Chat Component Helpers ---


def chat_message_item(msg: dict) -> rx.Component:
    is_user = msg["role"] == "user"
    return rx.hstack(
        rx.box(
            rx.text(msg["content"], font_size="13px", line_height="1.5"),
            padding="10px 14px",
            border_radius="14px",
            background=rx.cond(is_user, "rgba(14, 165, 233, 0.3)", "rgba(255, 255, 255, 0.06)"),
            border=rx.cond(is_user, "1px solid rgba(14, 165, 233, 0.5)", "1px solid rgba(255, 255, 255, 0.1)"),
            color="#F1F5F9",
            max_width="85%",
        ),
        justify=rx.cond(is_user, "end", "start"),
        width="100%",
        margin_y="4px",
    )


def ai_chatbox_widget() -> rx.Component:
    return rx.box(
        # 1. Floating Launcher Button
        rx.button(
            rx.hstack(
                rx.icon("bot", size=20, color="#04121E"),
                rx.text("Clinical AI", font_size="13px", font_weight="700", color="#04121E"),
                spacing="2",
                align="center",
            ),
            on_click=State.toggle_chat,
            border_radius="100px",
            padding="0 20px",
            height="48px",
            background="linear-gradient(135deg, #0EA5E9 0%, #2DD4BF 100%)",
            box_shadow="0 10px 25px -5px rgba(14, 165, 233, 0.5)",
            cursor="pointer",
            position="fixed",
            bottom="24px",
            right="28px",
            z_index="50",
            _hover={"transform": "scale(1.04)"},
        ),
        # 2. Expanded Glass Chat Window
        rx.box(
            glass_card(
                # Header
                rx.hstack(
                    rx.hstack(
                        rx.box(
                            rx.icon("sparkles", size=16, color="#2DD4BF"),
                            padding="6px",
                            border_radius="8px",
                            background="rgba(45, 212, 191, 0.15)",
                        ),
                        rx.vstack(
                            rx.text("Endocrine Copilot", font_weight="700", font_size="13px", color="#FFFFFF"),
                            rx.text("Llama 3.3 70B • Context Aware", font_size="10px", color="#64748B"),
                            spacing="0",
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.spacer(),
                    rx.button(
                        rx.icon("x", size=16),
                        on_click=State.toggle_chat,
                        variant="ghost",
                        size="1",
                        cursor="pointer",
                    ),
                    width="100%",
                    align="center",
                    margin_bottom="12px",
                ),
                # Messages Area
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(State.messages, chat_message_item),
                        width="100%",
                        spacing="1",
                    ),
                    height="320px",
                    type="auto",
                    scrollbars="vertical",
                    padding_right="8px",
                ),
                # Input Field + Send
                rx.hstack(
                    rx.input(
                        placeholder="Ask about your risk...",
                        value=State.chat_input,
                        on_change=State.set_chat_input,
                        border_radius="10px",
                        background="rgba(255, 255, 255, 0.05)",
                        border="1px solid rgba(255, 255, 255, 0.1)",
                        size="2",
                        flex_grow="1",
                    ),
                    rx.button(
                        rx.icon("send", size=15),
                        on_click=State.send_chat_message,
                        color_scheme="teal",
                        size="2",
                        cursor="pointer",
                    ),
                    spacing="2",
                    width="100%",
                    margin_top="12px",
                ),
                padding="20px",
                width="380px",
            ),
            position="fixed",
            bottom="84px",
            right="28px",
            z_index="50",
            display=rx.cond(State.chat_open, "block", "none"),
        ),
    )


# --- Section Views ---


def diagnostic_view() -> rx.Component:
    return rx.box(
        rx.grid(
            glass_card(
                rx.hstack(
                    rx.box(
                        rx.icon("sliders-horizontal", size=18, color="#2DD4BF"),
                        padding="8px",
                        border_radius="10px",
                        background="rgba(45, 212, 191, 0.1)",
                    ),
                    rx.vstack(
                        rx.text("Clinical Parameters", font_size="16px", font_weight="600", color="#F8FAFC"),
                        rx.text("Adjust laboratory biomarkers for real-time risk simulation", font_size="12px", color="#64748B"),
                        spacing="0",
                    ),
                    spacing="3",
                    align="center",
                    margin_bottom="18px",
                ),
                rx.hstack(
                    rx.text("PROFILES:", font_size="10px", font_weight="700", letter_spacing="1px", color="#64748B"),
                    rx.spacer(),
                    rx.button(
                        rx.hstack(rx.icon("user", size=14), rx.text("Healthy / Baseline")),
                        on_click=State.load_profile_healthy,
                        variant="soft",
                        color_scheme="teal",
                        size="2",
                        cursor="pointer",
                    ),
                    rx.button(
                        rx.hstack(rx.icon("alert-triangle", size=14), rx.text("Elevated Risk")),
                        on_click=State.load_profile_elevated,
                        variant="soft",
                        color_scheme="amber",
                        size="2",
                        cursor="pointer",
                    ),
                    spacing="3",
                    width="100%",
                    align="center",
                    padding="12px",
                    border_radius="14px",
                    background="rgba(255, 255, 255, 0.02)",
                    border="1px solid rgba(255, 255, 255, 0.05)",
                    margin_bottom="24px",
                ),
                rx.vstack(
                    control_slider("Plasma Glucose (2-hr)", "mg/dL", State.glucose, State.set_val_glucose, 50, 250, "Glucose"),
                    control_slider("Body Mass Index", "kg/m²", State.bmi, State.set_val_bmi, 15.0, 60.0, "BMI", step_v=0.1),
                    control_slider("Patient Age", "yrs", State.age, State.set_val_age, 21, 90, "Age"),
                    control_slider("Serum Insulin", "μU/mL", State.insulin, State.set_val_insulin, 10, 500, "Insulin"),
                    control_slider("Diastolic Blood Pressure", "mm Hg", State.blood_pressure, State.set_val_blood_pressure, 40, 140, "Pressure"),
                    control_slider("Diabetes Pedigree Function", "score", State.dpf, State.set_val_dpf, 0.05, 2.5, "DPF", step_v=0.01),
                    control_slider("Total Pregnancies", "count", State.pregnancies, State.set_val_pregnancies, 0, 17, "Pregnancies"),
                    rx.button(
                        rx.hstack(
                            rx.icon("sparkles", size=18),
                            rx.text("Assess Diabetic Risk", font_weight="600", letter_spacing="0.4px"),
                            spacing="2",
                            align="center",
                        ),
                        on_click=State.run_inference,
                        width="100%",
                        size="3",
                        margin_top="16px",
                        border_radius="14px",
                        background="linear-gradient(135deg, #0EA5E9 0%, #2DD4BF 100%)",
                        box_shadow="0 15px 30px -5px rgba(14, 165, 233, 0.4)",
                        color="#04121E",
                        cursor="pointer",
                    ),
                    spacing="4",
                ),
            ),
            rx.vstack(
                glass_card(
                    rx.vstack(
                        rx.text("PREDICTIVE STRATIFICATION", font_size="11px", font_weight="700", letter_spacing="2.5px", color="#64748B"),
                        rx.box(
                            rx.vstack(
                                rx.text(f"{State.risk_score}%", font_size="68px", font_weight="900", color=State.risk_color, line_height="1", letter_spacing="-1.5px"),
                                rx.text("PROBABILITY INDEX", font_size="10px", font_weight="700", letter_spacing="2px", color="#94A3B8"),
                                align="center",
                                spacing="1",
                            ),
                            padding="52px",
                            border_radius="100%",
                            background="radial-gradient(circle at 35% 30%, rgba(255, 255, 255, 0.08), rgba(0, 0, 0, 0.7))",
                            border=f"2px solid {State.risk_color}",
                            box_shadow=f"0 0 70px -5px {State.risk_glow}, inset 0 0 30px {State.risk_glow}",
                            margin_y="24px",
                            transition="all 0.5s cubic-bezier(0.16, 1, 0.3, 1)",
                        ),
                        rx.box(
                            rx.text(State.risk_tier, font_size="13px", font_weight="700", color=State.risk_color, letter_spacing="0.5px"),
                            padding="6px 18px",
                            border_radius="100px",
                            background="rgba(255, 255, 255, 0.03)",
                            border=f"1px solid {State.risk_color}",
                        ),
                        align="center",
                        width="100%",
                    ),
                    width="100%",
                ),
                glass_card(
                    rx.vstack(
                        rx.text("Model Traceability & Specifications", font_size="13px", font_weight="600", color="#E2E8F0"),
                        rx.hstack(
                            rx.box(rx.text("CLASSIFIER", font_size="9px", font_weight="700", color="#38BDF8"), padding="2px 6px", border_radius="4px", background="rgba(56, 189, 248, 0.1)"),
                            rx.text("Balanced Random Forest (250 Estimators, Depth 6)", font_size="12px", color="#94A3B8"),
                            align="center",
                        ),
                        rx.hstack(
                            rx.box(rx.text("IMPUTATION", font_size="9px", font_weight="700", color="#34D399"), padding="2px 6px", border_radius="4px", background="rgba(52, 211, 153, 0.1)"),
                            rx.text("Train-Set Median Imputer (Zero Target Leakage)", font_size="12px", color="#94A3B8"),
                            align="center",
                        ),
                        rx.hstack(
                            rx.box(rx.text("CALIBRATION", font_size="9px", font_weight="700", color="#F472B6"), padding="2px 6px", border_radius="4px", background="rgba(244, 114, 182, 0.1)"),
                            rx.text("Clinical Sensitivity & False-Negative Minimization", font_size="12px", color="#94A3B8"),
                            align="center",
                        ),
                        spacing="3",
                        width="100%",
                    ),
                    width="100%",
                ),
                spacing="5",
                width="100%",
            ),
            columns="2",
            spacing="6",
            width="100%",
        ),
        display=rx.cond(State.active_tab == "predictor", "block", "none"),
        width="100%",
    )


def analytics_view() -> rx.Component:
    return rx.box(
        glass_card(
            rx.hstack(
                rx.vstack(
                    rx.heading("Population Health Surveillance Matrix", size="5", weight="bold", color="#F8FAFC"),
                    rx.text("Aggregated cohort exploratory metrics, age-bracket prevalence curves, and biomarker risk distributions.", font_size="13px", color="#64748B"),
                    spacing="1",
                ),
                rx.spacer(),
                rx.badge("Power BI Analytical View", variant="surface", color_scheme="teal", size="3"),
                width="100%",
                align="center",
                margin_bottom="20px",
            ),
            rx.box(
                rx.image(
                    src="/powerbi_preview.png",
                    width="100%",
                    border_radius="16px",
                    border="1px solid rgba(255, 255, 255, 0.12)",
                    box_shadow="0 30px 60px -15px rgba(0, 0, 0, 0.9)",
                ),
                padding="6px",
                border_radius="20px",
                background="linear-gradient(180deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.01) 100%)",
                width="100%",
            ),
            width="100%",
        ),
        display=rx.cond(State.active_tab == "analytics", "block", "none"),
        width="100%",
    )


# --- Master Page ---


def index() -> rx.Component:
    return rx.box(
        # Navigation Bar
        rx.hstack(
            rx.hstack(
                rx.box(
                    rx.icon("activity", size=24, color="#38BDF8"),
                    padding="10px",
                    border_radius="12px",
                    background="rgba(56, 189, 248, 0.1)",
                    border="1px solid rgba(56, 189, 248, 0.25)",
                ),
                rx.vstack(
                    rx.text("ENDOCRINE AI", font_size="18px", font_weight="800", letter_spacing="2px", color="#FFFFFF"),
                    rx.text("CLINICAL INTELLIGENCE PLATFORM", font_size="10px", font_weight="600", letter_spacing="1.5px", color="#64748B"),
                    spacing="0",
                ),
                spacing="3",
                align="center",
            ),
            rx.spacer(),
            rx.box(
                rx.hstack(
                    rx.button(
                        rx.hstack(rx.icon("cpu", size=16), rx.text("Diagnostic Engine", font_size="13px")),
                        on_click=State.set_predictor_tab,
                        background=rx.cond(State.active_tab == "predictor", "rgba(255, 255, 255, 0.12)", "transparent"),
                        color=rx.cond(State.active_tab == "predictor", "#FFFFFF", "#94A3B8"),
                        border_radius="100px",
                        padding="8px 18px",
                        variant="ghost",
                        cursor="pointer",
                    ),
                    rx.button(
                        rx.hstack(rx.icon("bar-chart-2", size=16), rx.text("Surveillance Analytics", font_size="13px")),
                        on_click=State.set_analytics_tab,
                        background=rx.cond(State.active_tab == "analytics", "rgba(255, 255, 255, 0.12)", "transparent"),
                        color=rx.cond(State.active_tab == "analytics", "#FFFFFF", "#94A3B8"),
                        border_radius="100px",
                        padding="8px 18px",
                        variant="ghost",
                        cursor="pointer",
                    ),
                    spacing="1",
                ),
                padding="4px",
                border_radius="100px",
                background="rgba(15, 23, 42, 0.6)",
                backdrop_filter="blur(20px)",
                border="1px solid rgba(255, 255, 255, 0.08)",
            ),
            width="100%",
            align="center",
            margin_bottom="36px",
        ),
        # Tab Views
        diagnostic_view(),
        analytics_view(),
        # Embedded Floating AI Chatbot
        ai_chatbox_widget(),
        # Master Background Styling
        background="radial-gradient(circle at 50% 0%, #0F172A 0%, #020617 75%, #000000 100%)",
        min_height="100vh",
        color="#F8FAFC",
        padding="40px 64px",
    )


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        accent_color="teal",
    )
)
app.add_page(index, title="Endocrine AI // Diagnostic Intelligence")