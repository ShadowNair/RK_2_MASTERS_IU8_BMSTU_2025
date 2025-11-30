"""
Technical Support Analytics Dashboard
Course Project for "Systems Programming"  
Group: IU8-13M
Author: Gleb Borodin

Objective:  
Implement an API client to fetch technical support ticket data,  
generate a comprehensive analytical dashboard, and produce visual  
insights to support managerial decision-making.

API Endpoints Used:
- /api/v1/tickets — list of all tickets assigned to the user
- /api/v1/timeline?days=30 — daily metrics over the last 30 days

Visualizations Generated:
1. Ticket creation trend (line plot)
2. Hourly ticket distribution (bar chart)
3. Activity heatmap (day of week × hour)
4. Ticket category distribution (pie chart)
5. Average resolution time by category (horizontal bar chart)
6. Top-5 most problematic categories (logged + CSV export)
7. Timeline of opened vs. resolved tickets (line plot)

Implementation Date: November 30, 2025
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# === Import and initialize custom logger ===
from logger import analysis_logger  # Assumes logger class is defined in logger.py

# Configure consistent plot styling
plt.rcParams.update({'font.size': 10})

# === API Configuration ===
BASE_URL = "http://193.233.171.205:5000"
LOGIN = "analyst_ts"
CODE = "XyZ67iOp89Ij"
OUTPUT_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize dedicated logger for analytics
logger = analysis_logger.get_analysis_logger("SupportAnalytics")


def fetch_data(endpoint, params=None):
    """
    Performs an authenticated GET request to the specified API endpoint.

    Args:
        endpoint (str): API path (e.g., '/api/v1/tickets')
        params (dict, optional): Additional query parameters

    Returns:
        dict or list: JSON response from the server, or None on failure
    """
    url = f"{BASE_URL}{endpoint}"
    request_params = params.copy() if params else {}
    request_params.update({"login": LOGIN, "code": CODE})
    
    try:
        response = requests.get(url, params=request_params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while accessing {endpoint}: {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON received from {endpoint}")
        return None


def main():
    """Main entry point: fetches data, processes it, and generates visualizations."""
    
    logger.info("Starting technical support analytics dashboard")
    
    # Step 1: Check server health
    logger.info("Checking server health...")
    health = fetch_data("/api/v1/health")
    if not health:
        logger.error("Server is unavailable. Exiting.")
        return

    # Step 2: Fetch ticket data
    logger.info("Fetching ticket data...")
    tickets = fetch_data("/api/v1/tickets")
    if not tickets:
        logger.error("Failed to retrieve ticket list.")
        return

    # Step 3: Fetch timeline metrics
    logger.info("Fetching timeline data for the last 30 days...")
    timeline = fetch_data("/api/v1/timeline", {"days": 30})

    # Step 4: Data preprocessing
    df = pd.DataFrame(tickets)
    
    # Parse datetime fields from RFC 2822 format
    df["created_at"] = pd.to_datetime(df["created_at"], format="%a, %d %b %Y %H:%M:%S GMT", errors='coerce')
    df["closed_at"] = pd.to_datetime(df["closed_at"], format="%a, %d %b %Y %H:%M:%S GMT", errors='coerce')
    df.dropna(subset=["created_at"], inplace=True)

    # Derive additional time-based features
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["resolve_time_hours"] = (df["closed_at"] - df["created_at"]).dt.total_seconds() / 3600.0

    # === Visualization 1: Ticket creation trend (30 days) ===
    logger.info("Generating plot: ticket creation trend...")
    thirty_days_ago = datetime.now() - pd.Timedelta(days=30)
    last_30_days = df[df["created_at"] >= thirty_days_ago]
    daily_counts = last_30_days.groupby(last_30_days["created_at"].dt.date).size()
    
    plt.figure(figsize=(12, 5))
    plt.plot(daily_counts.index, daily_counts.values, marker='o', color='steelblue')
    plt.title("Ticket Creation Trend (Last 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Number of Tickets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ticket_trend.png"))
    plt.close()
    logger.info("Saved 'ticket_trend.png'")

    # === Visualization 2: Hourly ticket distribution ===
    logger.info("Generating plot: hourly ticket distribution...")
    hourly_counts = df["hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
    
    plt.figure(figsize=(12, 5))
    plt.bar(hourly_counts.index, hourly_counts.values, color='orange')
    plt.title("Ticket Distribution by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Number of Tickets")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ticket_by_hour.png"))
    plt.close()
    logger.info("Saved 'ticket_by_hour.png'")

    # === Visualization 3: Activity heatmap (weekday × hour) ===
    logger.info("Generating plot: activity heatmap...")
    heatmap_data = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
    heatmap_matrix = np.zeros((7, 24))
    for day in range(7):
        for hour in range(24):
            heatmap_matrix[day, hour] = heatmap_data.get(hour, pd.Series()).get(day, 0)
    
    plt.figure(figsize=(14, 6))
    im = plt.imshow(heatmap_matrix, cmap="YlGnBu", aspect='auto', origin='lower')
    plt.colorbar(im, label='Number of Tickets')
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")
    plt.title("Activity Heatmap: Day of Week vs. Hour")
    plt.yticks(ticks=range(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xticks(ticks=range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_weekday_hour.png"))
    plt.close()
    logger.info("Saved 'heatmap_weekday_hour.png'")

    # === Visualization 4: Category distribution (pie chart) ===
    logger.info("Generating plot: ticket category distribution...")
    category_counts = df["category_name"].value_counts()
    
    plt.figure(figsize=(9, 9))
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Ticket Distribution by Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "categories_pie.png"))
    plt.close()
    logger.info("Saved 'categories_pie.png'")

    # === Visualization 5: Average resolution time by category ===
    logger.info("Generating plot: average resolution time by category...")
    resolved_tickets = df.dropna(subset=["resolve_time_hours"])
    if not resolved_tickets.empty:
        avg_resolve_time = resolved_tickets.groupby("category_name")["resolve_time_hours"].mean().sort_values()
        plt.figure(figsize=(10, 7))
        plt.barh(avg_resolve_time.index, avg_resolve_time.values, color='green')
        plt.title("Average Ticket Resolution Time by Category (Hours)")
        plt.xlabel("Average Time (Hours)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "avg_resolve_time.png"))
        plt.close()
        logger.info("Saved 'avg_resolve_time.png'")
    else:
        logger.warning("No resolved tickets found for resolution time calculation.")

    # === Output 6: Top-5 most problematic categories ===
    logger.info("Generating Top-5 most problematic categories report...")
    top5 = category_counts.head(5)
    top5_str = top5.to_string()
    logger.info(f"Top-5 most problematic categories:\n{top5_str}")
    top5.to_csv(os.path.join(OUTPUT_DIR, "top5_categories.csv"), header=["Count"], encoding="utf-8")
    logger.info("Saved 'top5_categories.csv'")

    # === Visualization 7: Timeline of opened vs. resolved tickets ===
    if timeline:
        logger.info("Generating plot: opened vs. resolved tickets timeline...")
        tl_data = timeline.get("data", [])
        if tl_data:
            tl_df = pd.DataFrame(tl_data)
            tl_df["date"] = pd.to_datetime(tl_df["date"])
            tl_df.sort_values("date", inplace=True)
            
            plt.figure(figsize=(12, 6))
            plt.plot(tl_df["date"], tl_df["tickets_created"], label="Opened", marker='o')
            plt.plot(tl_df["date"], tl_df["tickets_resolved"], label="Resolved", marker='x')
            plt.title("Timeline of Opened vs. Resolved Tickets (30 Days)")
            plt.xlabel("Date")
            plt.ylabel("Number of Tickets")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "timeline_opened_closed.png"))
            plt.close()
            logger.info("Saved 'timeline_opened_closed.png'")
        else:
            logger.warning("Timeline data is empty.")
    else:
        logger.warning("Failed to fetch timeline data.")
    

    generate_pdf_report()

    logger.info(f"Analytics dashboard completed. All outputs saved to: {OUTPUT_DIR}")

def generate_pdf_report():
    """
    Generates a comprehensive PDF report with Cyrillic support.
    Uses DejaVuSans font for Russian text.
    """
    logger.info("Generating PDF analytical report with Cyrillic support...")

    pdf_path = os.path.join(OUTPUT_DIR, "support_analytics_report.pdf")

    # === Регистрация шрифта с кириллицей ===
    try:
        # Путь к DejaVuSans (стандартный для Linux/WSL)
        font_path = "DejaVuSans.ttf"
        if not os.path.exists(font_path):
            # Альтернатива: попробуем найти через matplotlib (часто есть)
            import matplotlib.font_manager as fm
            font_paths = fm.findfont("DejaVu Sans", fallback_to_default=False)
            if font_paths and os.path.exists(font_paths):
                font_path = font_paths
            else:
                # Если ничего не найдено — ошибка
                raise FileNotFoundError("DejaVuSans.ttf not found")
        
        pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
        font_name = 'DejaVuSans'
        logger.info(f"Using font: {font_path}")
    except Exception as e:
        logger.error(f"Failed to load Cyrillic font: {e}. Falling back to default (may break Cyrillic).")
        font_name = 'Helvetica'

    # === Настройка стилей с кириллическим шрифтом ===
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        spaceAfter=14,
        alignment=1  # center
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=11,
        spaceAfter=8
    )
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        spaceAfter=10
    )

    # === Сборка документа ===
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    # Заголовок (на русском или английском — ваш выбор)
    elements.append(Paragraph("Аналитическая панель технической поддержки", title_style))
    elements.append(Paragraph("Курсовая работа по системному программированию", normal_style))
    elements.append(Paragraph("Группа: RK_2_MASTERS_IU8_BMSTU_2025", normal_style))
    elements.append(Paragraph("Автор: Бородин Глеб, ИУ8", normal_style))
    elements.append(Paragraph(f"Дата формирования отчёта: {datetime.now().strftime('%d.%m.%Y %H:%M')}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Введение
    intro = (
        "Данный отчёт представляет комплексный анализ данных тикетов технической поддержки "
        "за последние 30 дней. Визуализации включают динамику обращений, распределение по категориям, "
        "время решения и активность пользователей для поддержки управленческих решений."
    )
    elements.append(Paragraph(intro, normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Графики
    figure_files = [
        ("Динамика создания тикетов", "ticket_trend.png"),
        ("Распределение тикетов по часам", "ticket_by_hour.png"),
        ("Тепловая карта активности (день × час)", "heatmap_weekday_hour.png"),
        ("Распределение по категориям", "categories_pie.png"),
        ("Среднее время решения по категориям", "avg_resolve_time.png"),
        ("Динамика открытых и закрытых тикетов", "timeline_opened_closed.png")
    ]

    for title, filename in figure_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            elements.append(PageBreak())
            elements.append(Paragraph(title, heading2_style))
            img = Image(filepath, width=6.5 * inch, height=4 * inch)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"PDF: пропущен отсутствующий файл: {filename}")

    # Таблица ТОП-5
    elements.append(Paragraph("ТОП-5 самых проблемных категорий", heading2_style))
    csv_path = os.path.join(OUTPUT_DIR, "top5_categories.csv")
    if os.path.exists(csv_path):
        top5_df = pd.read_csv(csv_path)
        table_data = [["Категория", "Количество"]] + top5_df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("Данные о категориях недоступны.", normal_style))

    # Формирование PDF
    try:
        doc.build(elements)
        logger.info(f"PDF-отчёт успешно сохранён: {pdf_path}")
    except Exception as e:
        logger.error(f"Ошибка при генерации PDF: {e}")

if __name__ == "__main__":
    main()