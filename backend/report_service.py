"""Generate a downloadable PDF from a saved interview record."""
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def build_interview_pdf(record: dict) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=(595.28, 841.89),
                                 rightMargin=48, leftMargin=48,
                                 topMargin=54, bottomMargin=48)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"],
                              fontSize=19, leading=25, alignment=TA_CENTER,
                              textColor=colors.HexColor("#17365D"), spaceAfter=12))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"],
                              fontSize=10, leading=15))
    is_test = record.get("record_type") == "VIDEO_TEST"
    report_type = "Video Test" if is_test else "Live Interview"
    subtitle = ("Known Test Video - AI Screening Summary" if is_test
                else "Corporate HR Video Interview - AI Screening Summary")
    story = [Paragraph("DEEPFAKE DETECTION REPORT", styles["ReportTitle"]),
             Paragraph(subtitle, styles["Normal"]),
             Spacer(1, 22)]

    def safe(value):
        return escape(str(value if value is not None and value != "" else "Not provided"))

    rows = [
        ("Database Record ID", record.get("id")),
        ("Detection Type", report_type),
        ("Test ID" if is_test else "Interview ID", record.get("interview_id")),
        ("Test Video Label" if is_test else "Candidate Name", record.get("candidate_name")),
        *(([]) if is_test else [("Candidate USN", record.get("candidate_usn"))]),
        ("Recorded At (UTC)", record.get("created_at")),
        ("Final Prediction", record.get("prediction")),
        ("Deepfake Risk Score", f'{float(record.get("risk_score", 0)):.2f}%'),
        ("Detection Threshold", "45.00%"),
        ("Frames Analyzed", record.get("frames_analyzed")),
        ("Classification Model", "EfficientNet-B0"),
        ("Face Processing", "MediaPipe"),
    ]
    data = [[Paragraph("FIELD", styles["ReportBody"]), Paragraph("VALUE", styles["ReportBody"])]]
    data += [[Paragraph(safe(label), styles["ReportBody"]),
              Paragraph(safe(value), styles["ReportBody"])] for label, value in rows]
    table = Table(data, colWidths=[185, 310], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0F9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DEE8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story += [table, Spacer(1, 22),
              Paragraph("Important Notice", styles["Heading3"]),
              Paragraph("This report is an automated AI screening result, not definitive proof "
                        "that a video is authentic or manipulated. Review flagged results "
                        "manually before making any employment decision.", styles["ReportBody"])]
    document.build(story)
    return buffer.getvalue()
