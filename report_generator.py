"""
PDF Report Generator for Resume Analysis
Creates professional PDF reports with charts and visualizations
"""
import os
from datetime import datetime
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
import logging

logger = logging.getLogger(__name__)


class ResumeReportGenerator:
    """Generate professional PDF reports for resume analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Score style
        self.styles.add(ParagraphStyle(
            name='ScoreText',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=10,
        ))
    
    def generate_report(self, data: dict, output_path: str = None) -> str:
        """
        Generate PDF report from analysis data
        
        Args:
            data: Analysis results dictionary
            output_path: Optional custom output path
            
        Returns:
            Path to generated PDF file
        """
        if not output_path:
            output_dir = Path('reports')
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f'ResumeIQ_Report_{timestamp}.pdf'
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Build content
        story = []
        
        # Header
        story.extend(self._build_header(data))
        
        # Executive Summary
        story.extend(self._build_summary_section(data))
        
        # Score Breakdown
        story.extend(self._build_scores_section(data))
        
        # Skills Analysis
        story.extend(self._build_skills_section(data))
        
        # Keywords Analysis
        story.extend(self._build_keywords_section(data))
        
        # Recommendations
        story.extend(self._build_recommendations_section(data))
        
        # AI Insights (if available)
        if data.get('ai_summary'):
            story.extend(self._build_ai_insights_section(data))
        
        # Footer
        story.extend(self._build_footer())
        
        # Build PDF
        try:
            doc.build(story)
            logger.info(f"Report generated: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise
    
    def _build_header(self, data: dict) -> list:
        """Build report header"""
        elements = []
        
        # Title
        title = Paragraph("ResumeIQ Pro - Analysis Report", self.styles['CustomTitle'])
        elements.append(title)
        
        # Date
        date_text = f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        date_para = Paragraph(date_text, self.styles['Normal'])
        elements.append(date_para)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_summary_section(self, data: dict) -> list:
        """Build executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Overall scores table
        match_score = data.get('match_score', 0)
        ats_score = data.get('ats_scores', {}).get('overall', 0)
        level = data.get('level', 'Unknown')
        
        summary_data = [
            ['Match Score', f"{match_score}%", self._get_score_label(match_score)],
            ['ATS Score', f"{ats_score}%", self._get_score_label(ats_score)],
            ['Overall Level', level, ''],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_scores_section(self, data: dict) -> list:
        """Build detailed scores breakdown"""
        elements = []
        
        elements.append(Paragraph("ATS Score Breakdown", self.styles['SectionHeader']))
        
        ats_scores = data.get('ats_scores', {})
        
        score_data = [
            ['Metric', 'Score', 'Weight'],
            ['Keyword Density', f"{ats_scores.get('keyword_density', 0)}%", '30%'],
            ['Skill Match', f"{ats_scores.get('skill_match', 0)}%", '30%'],
            ['Resume Length', f"{ats_scores.get('length', 0)}%", '15%'],
            ['Section Structure', f"{ats_scores.get('sections', 0)}%", '15%'],
            ['Action Verbs', f"{ats_scores.get('action_verbs', 0)}%", '10%'],
        ]
        
        score_table = Table(score_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ]))
        
        elements.append(score_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_skills_section(self, data: dict) -> list:
        """Build skills analysis section"""
        elements = []
        
        elements.append(Paragraph("Skills Analysis", self.styles['SectionHeader']))
        
        resume_skills = data.get('resume_skills', [])[:15]
        missing_skills = data.get('missing_skills', [])[:15]
        
        # Resume skills
        if resume_skills:
            elements.append(Paragraph("<b>Skills Found in Resume:</b>", self.styles['Normal']))
            skills_text = ', '.join(resume_skills)
            elements.append(Paragraph(skills_text, self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
        
        # Missing skills
        if missing_skills:
            elements.append(Paragraph("<b>Skills to Add:</b>", self.styles['Normal']))
            missing_text = ', '.join(missing_skills)
            elements.append(Paragraph(
                f'<font color="red">{missing_text}</font>',
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        return elements
    
    def _build_keywords_section(self, data: dict) -> list:
        """Build keywords analysis section"""
        elements = []
        
        elements.append(Paragraph("Keyword Analysis", self.styles['SectionHeader']))
        
        matching_kw = data.get('matching_keywords', [])[:20]
        missing_kw = data.get('missing_keywords', [])[:20]
        
        keyword_data = [
            ['Status', 'Keywords'],
            ['✓ Matched', ', '.join(matching_kw) if matching_kw else 'None'],
            ['✗ Missing', ', '.join(missing_kw) if missing_kw else 'None'],
        ]
        
        keyword_table = Table(keyword_data, colWidths=[1.5*inch, 5*inch])
        keyword_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(keyword_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_recommendations_section(self, data: dict) -> list:
        """Build recommendations section"""
        elements = []
        
        elements.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        
        recommendations = data.get('recommendations', [])
        
        for i, rec in enumerate(recommendations, 1):
            rec_text = f"{i}. {rec}"
            elements.append(Paragraph(rec_text, self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        return elements
    
    def _build_ai_insights_section(self, data: dict) -> list:
        """Build AI insights section"""
        elements = []
        
        elements.append(Paragraph("AI-Powered Insights", self.styles['SectionHeader']))
        
        # AI Summary
        ai_summary = data.get('ai_summary', '')
        if ai_summary:
            elements.append(Paragraph("<b>Summary:</b>", self.styles['Normal']))
            elements.append(Paragraph(ai_summary, self.styles['Normal']))
            elements.append(Spacer(1, 0.15*inch))
        
        # Strengths
        strengths = data.get('strengths', [])
        if strengths:
            elements.append(Paragraph("<b>Strengths:</b>", self.styles['Normal']))
            for strength in strengths:
                elements.append(Paragraph(f"• {strength}", self.styles['Normal']))
            elements.append(Spacer(1, 0.15*inch))
        
        # Weaknesses
        weaknesses = data.get('weaknesses', [])
        if weaknesses:
            elements.append(Paragraph("<b>Areas for Improvement:</b>", self.styles['Normal']))
            for weakness in weaknesses:
                elements.append(Paragraph(f"• {weakness}", self.styles['Normal']))
            elements.append(Spacer(1, 0.15*inch))
        
        # Interview Tips
        interview_tips = data.get('interview_tips', [])
        if interview_tips:
            elements.append(Paragraph("<b>Interview Preparation Tips:</b>", self.styles['Normal']))
            for tip in interview_tips:
                elements.append(Paragraph(f"• {tip}", self.styles['Normal']))
            elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _build_footer(self) -> list:
        """Build report footer"""
        elements = []
        
        elements.append(Spacer(1, 0.5*inch))
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        
        footer_text = "Generated by ResumeIQ Pro | AI-Powered Resume Analysis"
        elements.append(Paragraph(footer_text, footer_style))
        
        return elements
    
    def _get_score_label(self, score: float) -> str:
        """Get label for score"""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Needs Improvement"


# Convenience function
def generate_pdf_report(data: dict, output_path: str = None) -> str:
    """Generate PDF report"""
    generator = ResumeReportGenerator()
    return generator.generate_report(data, output_path)
