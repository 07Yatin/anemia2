import gradio as gr
import requests as r
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
from datetime import datetime
import pandas as pd
import os
import re
import tempfile

# --------------------
# Styling / Theming
# --------------------
CUSTOM_CSS = """
.gradio-container, body { color: #111111 !important; }
html, body { height: 100%; }
/* Force gradient background on container so Gradio white doesn't override */
.gradio-container {
    background-image: linear-gradient(135deg, #f8fafc 0%, #fdf2e9 40%, #fff8ed 100%) !important;
    background-attachment: fixed !important;
    background-size: cover !important;
    backdrop-filter: saturate(110%) contrast(102%);
}
/* Make inner wrappers transparent so gradient is visible */
.gradio-container .app, 
.gradio-container .wrap, 
.gradio-container .block, 
.gradio-container .tabs, 
.gradio-container .tabitem, 
.gradio-container .main {
    background: transparent !important;
}
.hero {
    background: linear-gradient(135deg, #ff7a18, #ffb347 60%, #ffd18f);
    border-radius: 16px;
    padding: 28px 28px 22px 28px;
    color: #1f2937;
    box-shadow: 0 12px 24px rgba(0,0,0,0.08);
}
.hero h1 {
    font-size: 28px;
    margin: 0 0 4px 0;
}
.hero p {
    margin: 0;
    opacity: 0.9;
}
.card {
    background: linear-gradient(180deg, #ffffff 0%, #fff8f0 100%);
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 10px 24px rgba(17, 24, 39, 0.08);
}
.cta-btn button {
    height: 48px;
    font-weight: 600;
    border-radius: 12px !important;
    color: #111111 !important;
    background: linear-gradient(135deg, #ffd18f, #ffb347 60%, #ff7a18);
    box-shadow: 0 8px 18px rgba(255, 122, 24, 0.25);
}
.cta-btn button span { 
    color: #111111 !important; 
    opacity: 1 !important;
}
.cta-btn button svg { filter: none !important; }
.muted-card {
    background: linear-gradient(180deg, #f7f7fb 0%, #f2f4f8 100%);
    border: 1px dashed #e5e7eb;
}
.muted-card button, .card button {
    color: #111111 !important;
}
.gr-button, .gr-button * { color: #111111 !important; }
.gr-button-primary { color: #111111 !important; }
.gr-button-primary span { color: #111111 !important; }
.gr-button { 
    font-size: 16px !important; 
    line-height: 1.2 !important; 
    background: linear-gradient(135deg, #ffe8c6, #ffb347 70%, #ff7a18);
    border: 0 !important;
    box-shadow: 0 6px 14px rgba(255, 122, 24, 0.18);
}
.gr-button > span { 
    color: #111111 !important; 
    font-size: 16px !important; 
    visibility: visible !important; 
    opacity: 1 !important; 
    text-indent: 0 !important; 
    display: inline-flex !important; 
    align-items: center; 
    gap: 8px; 
}
.gr-button svg { filter: none !important; }
.gr-button:hover { filter: brightness(1.02); transform: translateY(-1px); transition: all 160ms ease; }
.gr-button:active { transform: translateY(0); filter: brightness(0.98); }
.gr-markdown, .gr-markdown * { color: #111111 !important; }
.gradio-container .prose p, .gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 {
    color: #111111 !important;
}
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 12px;
    margin-top: 10px;
}
.center-row {
    justify-content: center !important;
}
.narrow-col {
    max-width: 720px;
}
"""

# Local history directory (replaces DB persistence)
LOCAL_HISTORY_DIR = 'user_history'
os.makedirs(LOCAL_HISTORY_DIR, exist_ok=True)

def numpy_array_to_bytes(np_array, format='png'):
    img = Image.fromarray(np_array)
    with BytesIO() as output:
        img.save(output, format=format)
        img_bytes = output.getvalue()
    return img_bytes

def call_api(img):
    img_data = {"file": ("image.jpg", img, "image/jpeg")}
    response = r.post("http://127.0.0.1:8081/predict", files=img_data)

    if response.status_code == 200:
        hgl = response.json().get("hgl")
        status = response.json().get("status")
        result = f"{hgl}"
        return result, status
    else:
        return "Error: Unable to process the image", "Error"

def save_result(hgl, status, image=None, username=None, flagged=False):
    # No-op for image saving
    return None

def parse_hgl_value(hgl_str):
    try:
        return float(str(hgl_str).replace('g/dl', '').strip())
    except Exception:
        return None

def generate_diet_plan(hgl_value):
    if hgl_value is None:
        return "<div class='card'><h4>Diet Plan</h4><p>Unable to determine hemoglobin value.</p></div>"
    if hgl_value < 9:
        severity = "Severe Anemia"
        advice = [
            "Consult a clinician urgently; supplements/transfusion may be required.",
            "Daily iron supplement as prescribed.",
            "Iron-rich foods: liver, red meat, fish, chicken, beans, lentils, tofu, spinach, fortified cereals.",
            "Add vitamin C sources with meals: citrus, berries, tomatoes, bell peppers.",
            "Avoid tea/coffee and calcium within 1–2 hours of iron intake."
        ]
    elif hgl_value < 11:
        severity = "Moderate Anemia"
        advice = [
            "Begin oral iron per clinician advice.",
            "Two iron-rich meals per day (see list above).",
            "Pair iron with vitamin C; cook in cast-iron cookware if available.",
            "Add B12 (eggs, dairy, fish) and folate (leafy greens, legumes)."
        ]
    elif hgl_value < 12:
        severity = "Mild / Borderline"
        advice = [
            "Focus on dietary iron daily (meat/legumes/greens).",
            "Include vitamin C; limit tea/coffee with meals.",
            "Consider multivitamin with iron if advised."
        ]
    else:
        severity = "Within Normal Range"
        advice = [
            "Maintain balanced diet with regular iron sources 3–4x/week.",
            "Include B12 and folate sources.",
            "Stay hydrated; continue periodic checks if recommended."
        ]
    items = ''.join([f"<li>{a}</li>" for a in advice])
    return f"""
    <div class='card'>
        <h4>Diet Plan • {severity}</h4>
        <ul>{items}</ul>
        <p style='font-size:12px;color:#6b7280'>General guidance only; not medical advice.</p>
    </div>
    """

## Downloadable PDF report removed

def process_image_from_upload(image, sex):
    if image is not None:
        img_bytes = numpy_array_to_bytes(image)
        result, status = call_api(img_bytes)
        hgl_val = parse_hgl_value(result)
        diet_html = generate_diet_plan(hgl_val)
        global LAST_RESULT
        LAST_RESULT = {
            'hemoglobin_level': hgl_val if hgl_val is not None else 0.0,
            'status': status,
        }
        return result, status, diet_html
    return "No image uploaded", "Error"

def process_image_from_camera(image, sex):
    if image is not None:
        img_bytes = numpy_array_to_bytes(image)
        result, status = call_api(img_bytes)
        hgl_val = parse_hgl_value(result)
        diet_html = generate_diet_plan(hgl_val)
        global LAST_RESULT
        LAST_RESULT = {
            'hemoglobin_level': hgl_val if hgl_val is not None else 0.0,
            'status': status,
        }
        return result, status, diet_html
    return "No image captured", "Error"

def append_local_history(username: str, record: dict) -> None:
    os.makedirs(LOCAL_HISTORY_DIR, exist_ok=True)
    path = os.path.join(LOCAL_HISTORY_DIR, f"{username}.jsonl")
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_local_history(username: str) -> pd.DataFrame:
    path = os.path.join(LOCAL_HISTORY_DIR, f"{username}.jsonl")
    rows = []
    try:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["timestamp","hemoglobin_level","status","username","flagged"])
        
        with open(path, 'r', encoding='utf-8') as f:
            rows = [json.loads(line.strip()) for line in f if line.strip()]
        
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp', ascending=False)
    except Exception as e:
        print(f"Error loading history for {username}: {e}")
        return pd.DataFrame(columns=["timestamp","hemoglobin_level","status","username","flagged"])

def get_filtered_history(start_date=None, end_date=None, min_hgl=None, max_hgl=None, status_filter=None, username=None, flagged_only=False):
    df = load_local_history(username or '')
    
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
    if min_hgl is not None:
        df = df[df['hemoglobin_level'] >= min_hgl]
    if max_hgl is not None:
        df = df[df['hemoglobin_level'] <= max_hgl]
    if status_filter:
        df = df[df['status'] == status_filter]
    if username:
        if 'username' in df.columns:
            df = df[df['username'].fillna('') == username]
    if flagged_only and 'flagged' in df.columns:
        df = df[df['flagged'] == 1]
    
    return df

def generate_statistics(df):
    if len(df) == 0:
        return "No data available for analysis"
    
    stats = {
        "Total Tests": len(df),
        "Average Hemoglobin": f"{df['hemoglobin_level'].mean():.2f}",
        "Highest Level": f"{df['hemoglobin_level'].max():.2f}",
        "Lowest Level": f"{df['hemoglobin_level'].min():.2f}",
        "Standard Deviation": f"{df['hemoglobin_level'].std():.2f}",
        "Last Test Date": df['timestamp'].max().strftime("%Y-%m-%d %H:%M"),
        "Status Distribution": df['status'].value_counts().to_dict()
    }
    
    return stats

def generate_enhanced_plot(df, plot_type="trend"):
    if len(df) == 0:
        return None
    
    plt.figure(figsize=(12, 6))
    
    if plot_type == "trend":
        plt.plot(pd.to_datetime(df['timestamp']), df['hemoglobin_level'], marker='o', linestyle='-')
        plt.title('Hemoglobin Level Trends')
        plt.xlabel('Date')
        plt.ylabel('Hemoglobin Level')
    elif plot_type == "distribution":
        sns.histplot(data=df, x='hemoglobin_level', bins=10)
        plt.title('Hemoglobin Level Distribution')
        plt.xlabel('Hemoglobin Level')
        plt.ylabel('Frequency')
    elif plot_type == "box":
        sns.boxplot(data=df, y='hemoglobin_level')
        plt.title('Hemoglobin Level Distribution')
        plt.ylabel('Hemoglobin Level')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    try:
        from PIL import Image as _PILImage
        import numpy as _np
        img = _PILImage.open(buf).convert('RGB')
        return _np.array(img)
    except Exception:
        return None

LAST_RESULT = None

def flag_latest_with_username(username: str):
    username = (username or '').strip()
    if not username:
        raise gr.Error("Please enter a username before saving.")
    global LAST_RESULT
    if LAST_RESULT is None:
        return gr.update(value="❌ No recent test found to save. Run a test first."), gr.update(visible=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "hemoglobin_level": float(LAST_RESULT.get('hemoglobin_level', 0.0)),
        "status": LAST_RESULT.get('status', ''),
        "username": username,
        "flagged": 1
    }
    append_local_history(username, record)
    msg = f"✅ Saved to history for {username} • Hb {record['hemoglobin_level']:.2f} g/dL • {record['status']}"
    return msg, gr.update(visible=True)

def view_user_history(username: str, plot_type: str):
    try:
        username = (username or '').strip()
        if not username:
            return None, "<div class='card'>Enter a username to view history.</div>"
        df = load_local_history(username)
        if len(df) == 0:
            return None, "<div class='card'>No results found for this username. Use 'Flag / Save' after a test.</div>"
        plot = generate_enhanced_plot(df, plot_type)
        # Customize display for better readability
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['hemoglobin_level'] = df['hemoglobin_level'].round(2)
        display_cols = ['timestamp', 'hemoglobin_level', 'status', 'username']
        html_table = df[display_cols].to_html(
            index=False, 
            classes='table table-striped table-hover', 
            table_id='history-table'
        )
        styled_html = f"""
        <style>
            #history-table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin-bottom: 1rem; 
            }}
            #history-table th, #history-table td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }}
            #history-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            #history-table th {{ 
                background-color: #ff7a18; 
                color: white; 
            }}
        </style>
        {html_table}
        """
        return plot, styled_html
    except Exception as e:
        return None, f"<div class='card'>Error loading history: {str(e)}</div>"

def view_history(start_date=None, end_date=None, min_hgl=None, max_hgl=None, 
                status_filter=None, plot_type="trend"):
    df = get_filtered_history(start_date, end_date, min_hgl, max_hgl, status_filter)
    stats = generate_statistics(df)
    plot = generate_enhanced_plot(df, plot_type)
    
    # Format statistics for display
    stats_html = "<h3>Statistical Analysis</h3><ul>"
    for key, value in stats.items():
        if isinstance(value, dict):
            stats_html += f"<li><strong>{key}:</strong><ul>"
            for k, v in value.items():
                stats_html += f"<li>{k}: {v}</li>"
            stats_html += "</ul></li>"
        else:
            stats_html += f"<li><strong>{key}:</strong> {value}</li>"
    stats_html += "</ul>"
    
    return plot, df.to_html(index=False), stats_html

def download_history(format, start_date=None, end_date=None):
    try:
        # Convert string dates to datetime objects
        start = None
        end = None
        if start_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise gr.Error("Invalid start date format. Please use YYYY-MM-DD")
        if end_date:
            try:
                end = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise gr.Error("Invalid end date format. Please use YYYY-MM-DD")
        
        df = get_filtered_history(start, end, None, None, None)
        if len(df) == 0:
            return None
        
        if format == "CSV":
            return df.to_csv(index=False)
        else:  # PDF
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
            from io import BytesIO
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []
            
            # Convert DataFrame to list of lists
            data = [df.columns.tolist()] + df.values.tolist()
            
            # Create table
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            doc.build(elements)
            buffer.seek(0)
            return buffer
    except Exception as e:
        raise gr.Error(f"Error generating download: {str(e)}")

def chat_with_bot(message, history):
    """
    Simple rule-based chatbot for anemia-related queries
    """
    message = message.lower()
    history = history or []
    
    # Base Q&A pairs (concise answers). We'll expand with synonyms to exceed 100 queries.
    base_pairs = {
        "what is anemia": "Anemia is when blood has too few healthy red cells or too little hemoglobin to carry oxygen effectively.",
        "types of anemia": "Common types: iron deficiency, B12 deficiency, folate deficiency, anemia of chronic disease, hemolytic anemia, aplastic anemia, thalassemia, sickle cell.",
        "causes of anemia": "Causes include poor iron/B12/folate intake, blood loss, chronic disease, kidney disease, pregnancy, inherited disorders, and bone marrow issues.",
        "symptoms of anemia": "Fatigue, weakness, pale skin, shortness of breath, dizziness, headaches, cold hands/feet, rapid heartbeat.",
        "diagnosis of anemia": "CBC to check hemoglobin/hematocrit, RBC indices, iron studies, B12/folate; clinician evaluation for cause.",
        "normal hemoglobin": "Typical Hb: men 13.5–17.5 g/dL, women 12.0–15.5 g/dL; children vary by age. Interpret with a clinician.",
        "what is hemoglobin": "Hemoglobin is the oxygen‑carrying protein in red blood cells.",
        "iron rich foods": "Red meat, liver, poultry, fish, beans, lentils, tofu, spinach, fortified cereals, pumpkin seeds.",
        "vitamin c role": "Vitamin C improves non‑heme iron absorption; pair citrus/berries/peppers with iron foods.",
        "foods to avoid with iron": "Tea/coffee, high‑calcium foods, and some antacids around iron intake; they reduce absorption.",
        "b12 sources": "Fish, meat, eggs, dairy, fortified plant milks/cereals; vegans usually need a B12 supplement.",
        "folate sources": "Leafy greens, legumes, asparagus, avocado, citrus fruits, fortified grains.",
        "iron supplements": "Take as prescribed; best on empty stomach with vitamin C; expect dark stools, possible constipation.",
        "risk factors": "Menstruation, pregnancy, poor diet, chronic disease, GI disorders, kidney disease, older age, inherited conditions.",
        "anemia in pregnancy": "Iron needs rise; prenatal vitamins with iron/folate are recommended; routine screening is standard.",
        "children anemia": "Often dietary iron deficiency; pediatric evaluation is important.",
        "when to see doctor": "If symptoms persist or Hb is low; urgent care for chest pain, fainting, severe breathlessness, or bleeding.",
        "prevent anemia": "Balanced diet with iron/B12/folate, vitamin C with meals, manage chronic diseases, regular checkups.",
        "exercise and anemia": "Light/moderate activity may help energy once treated; severe anemia warrants rest and medical guidance.",
        "difference iron vs b12 anemia": "Iron deficiency usually microcytic; B12 deficiency macrocytic with neurologic signs; different treatments.",
        "thalassemia": "Inherited disorder causing reduced globin chains; may cause microcytosis; managed by specialists.",
        "sickle cell": "Inherited Hb variant causing sickling, pain crises, anemia; requires specialized care.",
        "chronic disease anemia": "Inflammation limits iron availability and RBC production; treat underlying condition.",
        "blood loss anemia": "From heavy periods, GI bleeding, surgery, trauma; identify and treat the source.",
        "can anemia cause dizziness": "Yes—reduced oxygen delivery can cause dizziness and headaches.",
        "hair loss anemia": "Iron deficiency may contribute to hair loss; correct deficiency and review other causes.",
        "diet for anemia": "Include iron‑rich foods plus vitamin C; limit tea/coffee near meals; follow clinician advice.",
        "how accurate is this app": "It estimates Hb from conjunctiva images; lighting/angle/image quality affect accuracy—use as a guide only.",
        "is this medical advice": "No—educational screening only. Always consult a licensed clinician for diagnosis/treatment.",
        "how to use this app": "Choose Upload or Take Photo, capture the conjunctiva clearly, select sex, submit, then review Hb, status, and diet tips.",
        "improving photo quality": "Use bright natural light, avoid shadows, keep camera steady, ensure lower eyelid is pulled down to show conjunctiva.",
        "low hemoglobin symptoms": "Fatigue, pallor, exertional shortness of breath, palpitations, cold extremities.",
        "side effects iron": "Constipation, dark stools, nausea; take with vitamin C, adjust timing, or consult clinician.",
        "transfusion": "Reserved for severe/symptomatic anemia per clinical judgment.",
        "ckd anemia": "Kidney disease lowers EPO; managed with iron optimization and possibly ESAs under specialist care.",
        "ppis and iron": "Long‑term acid suppression may reduce iron absorption; discuss with clinician.",
        "vegetarian anemia": "Plan iron/B12 carefully; use fortified foods/supplements as advised.",
        "calcium and iron": "Separate high‑calcium foods/supplements from iron by a few hours.",
        "menstruation heavy": "Heavy periods are a common cause of iron deficiency—seek evaluation.",
        "gastrectomy anemia": "After bariatric/ GI surgery, malabsorption can cause deficiencies; monitor and supplement.",
        "celiac and anemia": "Celiac disease can reduce iron/folate absorption—testing may be indicated.",
        "lead and anemia": "Lead interferes with heme synthesis—environmental exposure assessment may be needed.",
        "pica": "Craving non‑food items (e.g., ice, clay) can be linked to iron deficiency—seek evaluation.",
        "athlete anemia": "Foot‑strike hemolysis, dilutional effects, and low iron intake may contribute—sports medicine input helps.",
        "altitude hemoglobin": "High altitude raises Hb over time; interpret results with context.",
        "lab monitoring": "Follow clinician guidance on repeat CBC/iron studies to track recovery.",
        "time to recover": "With treatment, Hb may rise ~1 g/dL every 2–3 weeks; full stores take months.",
        "folate vs b12": "Both cause macrocytosis; B12 deficiency can cause neurologic symptoms—don't treat folate alone if B12 low.",
        "tea coffee effect": "Polyphenols reduce iron absorption—avoid for 1–2 hours around iron‑rich meals.",
        "vitamin d anemia": "Links are mixed; address vitamin D separately if deficient.",
        "app privacy": "Processing is local; images are not saved unless you choose to save history.",
        "why conjunctiva": "Conjunctival pallor correlates with anemia; color cues inform the estimate.",
        "thresholds": "Severe <9, Moderate 9–11, Mild 11–12, Normal ≥12 g/dL (example bands).",
        "diet plan rationale": "Provides simple iron/B12/folate guidance; not a prescription.",
        "retake advice": "If result seems off, retake in better light with clearer conjunctiva framing.",
        "hydration effect": "Mild changes; true anemia requires lab correlation.",
        "infection and anemia": "Inflammation can suppress RBC production temporarily.",
        "covid and anemia": "Some patients experience anemia; management is individualized.",
        "gi symptoms": "Black stools, abdominal pain, or weight loss with anemia warrant urgent evaluation.",
        "hereditary anemia": "Family history of thalassemia/sickle cell warrants clinician testing and counseling.",
        "iron infusion": "IV iron is used when oral iron fails or is intolerable; administered in clinics.",
        "constipation tips": "Hydration, fiber, stool softener, or switching iron formulation can help—ask your clinician.",
        "pregnancy diet": "Iron‑rich foods, prenatal with iron/folate, vitamin C with meals; avoid excess tea/coffee.",
        "folate in pregnancy": "Folate prevents neural tube defects; prenatal vitamins supply recommended amounts.",
        "b12 deficiency signs": "Numbness, tingling, balance issues, glossitis—seek evaluation.",
        "app limitations": "Lighting, camera, and positioning affect accuracy; confirm with lab tests.",
        "export results": "Use View History to review and copy your past flagged results.",
        "contact clinician": "If concerned, contact your healthcare provider for proper testing and treatment."
    }

    # Synonym expansions to exceed 100 recognized queries
    synonyms = {
        "what is anemia": ["define anemia","anemia meaning","explain anemia","about anemia"],
        "symptoms of anemia": ["anemia symptoms","signs of anemia","how do I know if I am anemic"],
        "prevent anemia": ["how to prevent","prevention of anemia","avoid anemia"],
        "what is hemoglobin": ["define hemoglobin","hgb meaning","what does hemoglobin do"],
        "how to use this app": ["how to use app","how to use","guide","instructions"],
        "diet for anemia": ["diet recommendations","what to eat for anemia","foods for anemia"],
        "iron rich foods": ["best iron foods","iron sources","increase iron diet"],
        "b12 sources": ["vitamin b12 foods","b12 rich foods","sources of b12"],
        "folate sources": ["folic acid foods","folate rich foods","vitamin b9 foods"],
        "causes of anemia": ["why anemia happens","reason for anemia","anemia reasons"],
        "when to see doctor": ["when to see a doctor","should I see doctor","doctor for anemia"],
        "how accurate is this app": ["app accuracy","model accuracy","is result reliable"],
        "is this medical advice": ["is this diagnosis","is this clinical advice","medical disclaimer"],
        "improving photo quality": ["photo tips","image tips","how to take good photo"],
        "normal hemoglobin": ["normal hgb levels","normal hb","reference range hemoglobin"],
        "low hemoglobin symptoms": ["low hb symptoms","symptoms low hemoglobin"],
        "iron supplements": ["how to take iron","iron tablets tips","iron side effects"],
        "difference iron vs b12 anemia": ["iron vs b12","b12 vs iron deficiency","compare iron and b12 anemia"],
        "thresholds": ["cutoffs","bands","severity levels"],
        "app privacy": ["privacy","is data saved","do you save images"],
        "retake advice": ["retake photo","result seems wrong","try again"],
        "time to recover": ["how long to recover","recovery time anemia","how fast hemoglobin rises"],
        "hair loss anemia": ["anemia hair loss","does anemia cause hair fall"],
        "tea coffee effect": ["tea coffee iron","does tea block iron","coffee and iron"],
    }

    responses = dict(base_pairs)
    for key, alts in synonyms.items():
        for alt in alts:
            responses[alt] = base_pairs.get(key, base_pairs.get("what is anemia"))
    
    # Default response for unknown queries
    default_response = "I'm here to help with anemia-related questions. You can ask about: symptoms, prevention, treatment, diet, risk factors, or how to use this app. What would you like to know?"
    
    # Check for matching keywords in the message
    response = default_response
    for key in responses:
        if key in message:
            response = responses[key]
            break
    
    # Add some personality and follow-up suggestions
    if response == default_response:
        response += "\n\nYou can ask me about:\n• What is anemia?\n• What are the symptoms?\n• How to prevent anemia?\n• What is hemoglobin?\n• How to use this app?"
    
    return response

# Create the upload interface
upload_interface = gr.Interface(
    fn=process_image_from_upload,
    inputs=[
        gr.Image(label="Upload conjunctiva image", type="numpy"),
        gr.Radio(["Male", "Female"], label="Sex")
    ],
    outputs=[gr.Label(label="Hemoglobin Levels"), gr.Label(label="Status"), gr.HTML(label="Diet Plan")],
    title="Anemia Detector - Upload Image",
    description="Upload an image of the conjunctiva (the red part under the lower eyelid) to detect anemia and estimate hemoglobin levels.",
    allow_flagging="never"
)

# Create the camera interface
camera_interface = gr.Interface(
    fn=process_image_from_camera,
    inputs=[
        gr.Image(
            label="Capture when conjunctiva is visible",
            type="numpy",
            streaming=False,
        ),
        gr.Radio(["Male", "Female"], label="Sex")
    ],
    outputs=[gr.Label(label="Hemoglobin Levels"), gr.Label(label="Status"), gr.HTML(label="Diet Plan")],
    title="Anemia Detector - Camera Mode",
    description="Position your eye so the conjunctiva (red part under lower eyelid) is visible and take a photo.",
    allow_flagging="never"
)

# Create the chatbot interface using ChatInterface instead of Interface
chatbot_interface = gr.ChatInterface(
    fn=chat_with_bot,
    title="Anemia Information Assistant",
    description="Ask questions about anemia, symptoms, prevention, treatment, or how to use this app.",
        examples=[
            "What is anemia?",
            "Symptoms of anemia",
            "How to prevent anemia",
            "What is hemoglobin?",
            "Normal hemoglobin levels",
            "Iron rich foods",
            "Vitamin C role",
            "Foods to avoid with iron",
            "Vitamin B12 foods",
            "Folate foods",
            "Iron supplements tips",
            "Risk factors",
            "Anemia in pregnancy",
            "When to see doctor",
            "Difference iron vs B12 anemia",
            "Chronic disease anemia",
            "Blood loss anemia",
            "Can anemia cause dizziness?",
            "Hair loss and anemia",
            "Improve photo quality"
        ],
    theme="soft",
    type="messages"
)

## (history interface removed)

## (download interface removed)

def validate_date(date_str):
    try:
        if date_str:
            datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def create_main_interface():
    with gr.Blocks(title="Anemia Detection System", css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="orange", secondary_hue="zinc")) as main_interface:
        with gr.Row(elem_classes=["hero"]):
            with gr.Column(scale=8):
                gr.Markdown("# Anemia Detection System")
                gr.Markdown("Detect anemia and estimate hemoglobin quickly with a modern, friendly interface.")
        
        with gr.Row(elem_classes=["center-row"]):
            with gr.Column(elem_classes=["narrow-col"], scale=1):
                gr.Markdown("### Test Options")
                upload_btn = gr.Button("Upload Image", variant="primary") 
                camera_btn = gr.Button("Take Photo", variant="primary") 
            with gr.Column(elem_classes=["narrow-col"], scale=1):
                gr.Markdown("### More Options")
                history_btn = gr.Button("View History", variant="primary") 
                chatbot_btn = gr.Button("Ask Questions", variant="primary") 
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Quick Actions")
                help_btn = gr.Button("❓ Help & Guide", variant="secondary", elem_classes=["muted-card"]) 
        
        # Hidden interfaces that will be shown when buttons are clicked
        with gr.Row(visible=False, elem_classes=["card","center-row"]) as upload_section:
            with gr.Column(elem_classes=["narrow-col"]):
                upload_interface.render()
                gr.Markdown("### Save to History")
                with gr.Row():
                    username_input_u = gr.Textbox(label="Username", placeholder="enter username to save")
                    save_btn_u = gr.Button("Flag / Save", variant="secondary")
                save_msg_u = gr.Markdown(visible=False)
        
        with gr.Row(visible=False, elem_classes=["card","center-row"]) as camera_section:
            with gr.Column(elem_classes=["narrow-col"]):
                camera_interface.render()
                gr.Markdown("### Save to History")
                with gr.Row():
                    username_input_c = gr.Textbox(label="Username", placeholder="enter username to save")
                    save_btn_c = gr.Button("Flag / Save", variant="secondary")
                save_msg_c = gr.Markdown(visible=False)
        
        with gr.Row(visible=False, elem_classes=["card"]) as chatbot_section:
            chatbot_interface.render()
        
        with gr.Row(visible=False, elem_classes=["card","center-row"]) as history_section:
            with gr.Column(elem_classes=["narrow-col"]):
                gr.Markdown("### Your History")
                with gr.Row():
                    username_hist = gr.Textbox(label="Username", placeholder="enter username to filter")
                    plot_type = gr.Dropdown(["trend","distribution","box"], value="trend", label="Plot Type")
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                hist_plot = gr.Image(label="Trend", interactive=False)
                hist_table = gr.HTML(label="History Table")

        # History/Download sections removed per requirements
        
        # Filter section removed
        
        with gr.Row(visible=False, elem_classes=["card"]) as help_section:
            with gr.Column():
                gr.Markdown("### Help & Guide")
                gr.Markdown("""
                ## How to Use This App
                
                ### Taking a Test
                1. Click 'Take Photo' or 'Upload Image'
                2. Position your eye so the conjunctiva is visible
                3. Take or upload the photo
                4. View your results
                
                ### Viewing History
                1. Click 'View History' to see all your tests
                2. Use filters to find specific results
                3. Download your history in CSV or PDF format
                
                ### Getting Help
                1. Click 'Ask Questions' to chat with our AI assistant
                2. Use the help guide for step-by-step instructions
                3. Contact support if you need additional help
                """)
        
        # Button click handlers
        def show_section(section):
            return {section: gr.update(visible=True)}
        
        def hide_all_sections():
            return {
                upload_section: gr.update(visible=False),
                camera_section: gr.update(visible=False),
                chatbot_section: gr.update(visible=False),
                history_section: gr.update(visible=False),
                # filter_section removed
                help_section: gr.update(visible=False)
            }
        
        # Connect buttons to their respective sections
        upload_btn.click(
            fn=lambda: hide_all_sections() | show_section(upload_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section, history_section]
        )
        
        camera_btn.click(
            fn=lambda: hide_all_sections() | show_section(camera_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section, history_section]
        )
        
        chatbot_btn.click(
            fn=lambda: hide_all_sections() | show_section(chatbot_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section, history_section]
        )
        
        help_btn.click(
            fn=lambda: hide_all_sections() | show_section(help_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section, history_section]
        )
        
        history_btn.click(
            fn=lambda: hide_all_sections() | show_section(history_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section, history_section]
        )

        # Flag / Save actions
        save_btn_u.click(fn=flag_latest_with_username, inputs=[username_input_u], outputs=[save_msg_u, save_msg_u])
        save_btn_c.click(fn=flag_latest_with_username, inputs=[username_input_c], outputs=[save_msg_c, save_msg_c])

        # History refresh
        refresh_btn.click(fn=view_user_history, inputs=[username_hist, plot_type], outputs=[hist_plot, hist_table])
        # Filter logic removed
    
        gr.Markdown("Made with ❤️ for accessible screening.", elem_classes=["footer"]) 
    return main_interface

# Launch the main interface instead of the tabbed interface
create_main_interface().launch(share=True)