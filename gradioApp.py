import gradio as gr
import requests as r
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
import pandas as pd
import os
import re
import tempfile

# --------------------
# Styling / Theming
# --------------------
CUSTOM_CSS = """
.gradio-container, body {
    color: #111111 !important;
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
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 6px 14px rgba(0,0,0,0.06);
}
.cta-btn button {
    height: 48px;
    font-weight: 600;
    border-radius: 12px !important;
    color: #111111 !important;
    background: linear-gradient(135deg, #ffd18f, #ffb347 60%, #ff7a18);
}
.cta-btn button span { 
    color: #111111 !important; 
    opacity: 1 !important;
}
.cta-btn button svg { filter: none !important; }
.muted-card {
    background: #f3f4f6;
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

# Database setup
def setup_database():
    conn = sqlite3.connect('anemia_results.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS test_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  hemoglobin_level FLOAT,
                  status TEXT,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

# Create database and tables
setup_database()

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

def save_result(hgl, status, image=None):
    conn = sqlite3.connect('anemia_results.db')
    c = conn.cursor()
    
    # Save image if provided
    image_path = None
    if image is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"test_images/{timestamp}.jpg"
        os.makedirs("test_images", exist_ok=True)
        Image.fromarray(image).save(image_path)
    
    # Convert hemoglobin level string to float, removing the unit
    try:
        hgl_value = float(hgl.replace('g/dl', '').strip())
    except (ValueError, AttributeError):
        hgl_value = 0.0  # Default value if conversion fails
    
    # Save to database
    c.execute('''INSERT INTO test_results (timestamp, hemoglobin_level, status, image_path)
                 VALUES (?, ?, ?, ?)''',
              (datetime.now(), hgl_value, status, image_path))
    conn.commit()
    conn.close()

def parse_hgl_value(hgl_str):
    try:
        return float(str(hgl_str).replace('g/dl', '').strip())
    except Exception:
        return None

def generate_diet_plan(hgl_value):
    if hgl_value is None:
        return "<div class='card'><h4>Diet Plan</h4><p>Unable to determine hemoglobin value.</p></div>"
    if hgl_value < 7:
        severity = "Severe Anemia"
        advice = [
            "Consult a clinician urgently; supplements/transfusion may be required.",
            "Daily iron supplement as prescribed.",
            "Iron-rich foods: liver, red meat, fish, chicken, beans, lentils, tofu, spinach, fortified cereals.",
            "Add vitamin C sources with meals: citrus, berries, tomatoes, bell peppers.",
            "Avoid tea/coffee and calcium within 1–2 hours of iron intake."
        ]
    elif hgl_value < 10:
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

def generate_pdf_report(hgl_value, status, diet_html):
    try:
        # Lazy import to avoid hard dependency at module import time
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        # Build text versions
        diet_text = re.sub('<[^<]+?>', '', diet_html or '')
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        # Create a persistent file path; Gradio File downloads best from a real path
        os.makedirs("reports", exist_ok=True)
        filename = f"reports/anemia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        abs_path = os.path.abspath(filename)
        doc = SimpleDocTemplate(abs_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Anemia Detection Report", styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Date: {ts}", styles['Normal']))
        story.append(Paragraph(f"Hemoglobin: {hgl_value if hgl_value is not None else 'N/A'} g/dL", styles['Normal']))
        story.append(Paragraph(f"Status: {status}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Diet Plan", styles['Heading2']))
        story.append(Paragraph(diet_text.replace('\n', '<br/>'), styles['Normal']))
        doc.build(story)
        return abs_path
    except Exception:
        return None

def process_image_from_upload(image):
    if image is not None:
        img_bytes = numpy_array_to_bytes(image)
        result, status = call_api(img_bytes)
        hgl_val = parse_hgl_value(result)
        diet_html = generate_diet_plan(hgl_val)
        save_result(result, status, image)
        pdf_path = generate_pdf_report(hgl_val, status, diet_html)
        return result, status, diet_html, pdf_path
    return "No image uploaded", "Error"

def process_image_from_camera(image):
    if image is not None:
        img_bytes = numpy_array_to_bytes(image)
        result, status = call_api(img_bytes)
        hgl_val = parse_hgl_value(result)
        diet_html = generate_diet_plan(hgl_val)
        save_result(result, status, image)
        pdf_path = generate_pdf_report(hgl_val, status, diet_html)
        return result, status, diet_html, pdf_path
    return "No image captured", "Error"

def get_test_history():
    conn = sqlite3.connect('anemia_results.db')
    df = pd.read_sql_query("SELECT * FROM test_results ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_filtered_history(start_date=None, end_date=None, min_hgl=None, max_hgl=None, status_filter=None):
    df = get_test_history()
    
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
        plt.plot(df['timestamp'], df['hemoglobin_level'], marker='o', linestyle='-')
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
    return buf

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
    
    # Common questions and their answers
    responses = {
        "what is anemia": "Anemia is a condition where your blood lacks enough healthy red blood cells or hemoglobin. Hemoglobin is the main part of red blood cells and binds oxygen.",
        
        "what are the symptoms": "Common symptoms of anemia include fatigue, weakness, pale skin, shortness of breath, dizziness, and cold hands and feet.",
        
        "how to prevent": "To prevent anemia: 1) Eat iron-rich foods, 2) Include vitamin B12 and folate in your diet, 3) Get regular check-ups, 4) Maintain a balanced diet.",
        
        "what is hemoglobin": "Hemoglobin is a protein in red blood cells that carries oxygen from your lungs to the rest of your body. Normal levels are typically 13.5-17.5 g/dL for men and 12.0-15.5 g/dL for women.",
        
        "how to use this app": "1) Click on 'Take Photo' or 'Upload Image', 2) Position your eye so the conjunctiva (red part under lower eyelid) is visible, 3) Take or upload the photo, 4) Get your results and recommendations.",
        
        "what is conjunctiva": "The conjunctiva is the clear tissue that covers the white part of your eye and the inside of your eyelids. It's the red part under your lower eyelid that we analyze for anemia detection.",
        
        "treatment": "Treatment depends on the cause and severity. Common treatments include: 1) Iron supplements, 2) Vitamin B12 supplements, 3) Dietary changes, 4) Blood transfusions in severe cases. Always consult a healthcare provider.",
        
        "risk factors": "Risk factors include: 1) Iron deficiency, 2) Vitamin B12 deficiency, 3) Chronic diseases, 4) Family history, 5) Age (elderly), 6) Pregnancy, 7) Menstruation.",
        
        "when to see doctor": "See a doctor if you experience: 1) Persistent fatigue, 2) Shortness of breath, 3) Dizziness, 4) Pale skin, 5) Irregular heartbeat, 6) Cold hands and feet.",
        
        "diet recommendations": "Eat foods rich in: 1) Iron (red meat, beans, spinach), 2) Vitamin B12 (fish, meat, dairy), 3) Folate (leafy greens, citrus fruits), 4) Vitamin C (helps iron absorption).",

        "normal hgb levels": "Typical hemoglobin ranges: men 13.5–17.5 g/dL, women 12.0–15.5 g/dL, children vary by age. Always interpret with a clinician.",
        
        "iron rich foods": "Good sources: red meat, liver, chicken, fish, beans, lentils, tofu, spinach, fortified cereals, pumpkin seeds.",

        "b12 sources": "Vitamin B12 foods: fish, meat, eggs, dairy, fortified cereals. Vegans usually need a B12 supplement.",

        "folate sources": "Folate foods: leafy greens, legumes, asparagus, avocado, citrus fruits, fortified grains.",

        "causes of anemia": "Common causes: iron deficiency, chronic disease, blood loss, B12/folate deficiency, hemolysis, bone marrow problems, inherited conditions (e.g., thalassemia).",

        "anemia in pregnancy": "Pregnancy increases iron needs. Prenatal vitamins with iron/folate are recommended; screening is routine. Follow your obstetrician's guidance.",

        "can anemia cause dizziness": "Yes. Low hemoglobin reduces oxygen delivery, which can cause dizziness, fatigue, headaches, and shortness of breath.",

        "how to increase iron absorption": "Pair iron with vitamin C (e.g., citrus). Avoid tea/coffee and calcium supplements around iron-rich meals.",

        "is my result medical advice": "This app is for screening and education only. It does not replace professional diagnosis or treatment. Consult a licensed clinician.",

        "how accurate is this app": "The model estimates hemoglobin from eye images and has limitations. Lighting, camera quality, and positioning affect results. Use it as a guide, not a diagnosis.",

        "low hemoglobin symptoms": "Fatigue, pallor, shortness of breath, chest discomfort on exertion, cold extremities, brittle nails, hair loss in severe cases.",

        "when to go to er": "Seek urgent care if you have severe shortness of breath, chest pain, fainting, very fast heartbeat, or signs of major bleeding."
    }
    
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
    inputs=gr.Image(label="Upload conjunctiva image"),
    outputs=[gr.Label(label="Hemoglobin Levels"), gr.Label(label="Status"), gr.HTML(label="Diet Plan"), gr.File(label="Download Report (PDF)")],
    title="Anemia Detector - Upload Image",
    description="Upload an image of the conjunctiva (the red part under the lower eyelid) to detect anemia and estimate hemoglobin levels."
)

# Create the camera interface
camera_interface = gr.Interface(
    fn=process_image_from_camera,
    inputs=gr.Image(
        label="Capture when conjunctiva is visible",
        type="numpy",
        streaming=False,
    ),
    outputs=[gr.Label(label="Hemoglobin Levels"), gr.Label(label="Status"), gr.HTML(label="Diet Plan"), gr.File(label="Download Report (PDF)")],
    title="Anemia Detector - Camera Mode",
    description="Position your eye so the conjunctiva (red part under lower eyelid) is visible and take a photo."
)

# Create the chatbot interface using ChatInterface instead of Interface
chatbot_interface = gr.ChatInterface(
    fn=chat_with_bot,
    title="Anemia Information Assistant",
    description="Ask questions about anemia, symptoms, prevention, treatment, or how to use this app.",
        examples=[
            "What is anemia?",
            "What are the symptoms of anemia?",
            "How can I prevent anemia?",
            "What is hemoglobin?",
            "What are the risk factors for anemia?",
            "What should I eat if I have anemia?",
            "What are normal Hgb levels?",
            "Best sources of iron?",
            "Sources of vitamin B12?",
            "How to increase iron absorption?",
            "Is this medical advice?",
            "How accurate is this app?",
            "When should I see a doctor?",
            "Anemia in pregnancy",
            "Can anemia cause dizziness?"
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
            with gr.Column(elem_classes=["narrow-col"]):
                gr.Markdown("### Test Options")
                upload_btn = gr.Button("Upload Image", variant="primary") 
                camera_btn = gr.Button("Take Photo", variant="primary") 
                chatbot_btn = gr.Button("Ask Questions", variant="primary") 
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Quick Actions")
                help_btn = gr.Button("❓ Help & Guide", variant="secondary", elem_classes=["muted-card"]) 
        
        # Hidden interfaces that will be shown when buttons are clicked
        with gr.Row(visible=False, elem_classes=["card","center-row"]) as upload_section:
            with gr.Column(elem_classes=["narrow-col"]):
                upload_interface.render()
        
        with gr.Row(visible=False, elem_classes=["card","center-row"]) as camera_section:
            with gr.Column(elem_classes=["narrow-col"]):
                camera_interface.render()
        
        with gr.Row(visible=False, elem_classes=["card"]) as chatbot_section:
            chatbot_interface.render()
        
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
                # filter_section removed
                help_section: gr.update(visible=False)
            }
        
        # Connect buttons to their respective sections
        upload_btn.click(
            fn=lambda: hide_all_sections() | show_section(upload_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section]
        )
        
        camera_btn.click(
            fn=lambda: hide_all_sections() | show_section(camera_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section]
        )
        
        chatbot_btn.click(
            fn=lambda: hide_all_sections() | show_section(chatbot_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section]
        )
        
        help_btn.click(
            fn=lambda: hide_all_sections() | show_section(help_section),
            inputs=None,
            outputs=[upload_section, camera_section, chatbot_section, help_section]
        )
        # Filter logic removed
    
        gr.Markdown("Made with ❤️ for accessible screening.", elem_classes=["footer"]) 
    return main_interface

# Launch the main interface instead of the tabbed interface
create_main_interface().launch(share=True)