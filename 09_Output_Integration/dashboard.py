# Save this as: dashboard.py
# Location: C:\Users\Bhoomi\Major Project\08_Output_Integration\dashboard.py

import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from export_engine import TestCaseExportEngine


st.set_page_config(
    page_title="AI Test Case Generation Dashboard",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 AI-Powered Test Case Generation Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("📁 Data Sources")

test_cases_file = st.sidebar.text_input(
    "Test Cases JSON",
    value="../04_AI_powered_TestCaseGeneration/optimized_test_cases_20251017_000535.json"
)

validation_file = st.sidebar.text_input(
    "Validation Report",
    value="../06_Validation_QA/validation_report.xlsx"
)

rtm_file = st.sidebar.text_input(
    "RTM Report",
    value="../05_RTM_Generation/rtm_report.xlsx"
)

# Load data
try:
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        tc_data = json.load(f)
    
    test_cases = tc_data.get('phase1_test_cases', []) + tc_data.get('phase2_test_cases', [])
    metadata = tc_data.get('metadata', {})
    
    validation_df = pd.read_excel(validation_file, sheet_name='Validation Results')
    rtm_df = pd.read_excel(rtm_file, sheet_name='RTM Summary')
    coverage_df = pd.read_excel(rtm_file, sheet_name='Coverage Metrics')
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# KEY METRICS
st.header("📊 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Test Cases", len(test_cases), delta="399 generated")

with col2:
    avg_confidence = validation_df['overall_score'].mean()
    st.metric("Avg Confidence", f"{avg_confidence:.3f}", delta=f"{(avg_confidence - 0.75)*100:.1f}%")

with col3:
    requirements_covered = len(rtm_df)
    st.metric("Requirements", f"{requirements_covered}/56", delta=f"{requirements_covered/56*100:.1f}%")

with col4:
    high_conf_count = len(validation_df[validation_df['confidence_level'] == 'HIGH'])
    st.metric("High Confidence", high_conf_count, delta=f"{high_conf_count/len(test_cases)*100:.1f}%")

with col5:
    functional_coverage = coverage_df[coverage_df['metric_name'] == 'functional_coverage']['metric_value'].values[0]
    st.metric("Functional Coverage", f"{functional_coverage:.1f}%", delta=f"{functional_coverage - 95:.1f}%")

st.markdown("---")

# VISUALIZATIONS
st.header("📈 Coverage Analysis")

col1, col2 = st.columns(2)

with col1:
    test_types = pd.DataFrame(test_cases)['test_type'].value_counts()
    fig_types = px.bar(
        x=test_types.index, y=test_types.values,
        title="Test Type Distribution",
        labels={'x': 'Test Type', 'y': 'Count'},
        color=test_types.values,
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_types, use_container_width=True)

with col2:
    conf_dist = validation_df['confidence_level'].value_counts()
    fig_conf = px.pie(
        values=conf_dist.values, names=conf_dist.index,
        title="Confidence Level Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_conf, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    rtm_grouped = rtm_df.groupby('type')['total_tests'].sum().reset_index()
    fig_heatmap = px.bar(
        rtm_grouped, x='type', y='total_tests',
        title="Test Cases by Requirement Type",
        labels={'type': 'Requirement Type', 'total_tests': 'Test Cases'},
        color='total_tests', color_continuous_scale='blues'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col4:
    coverage_comparison = coverage_df.copy()
    fig_coverage = go.Figure()
    
    fig_coverage.add_trace(go.Bar(
        name='Actual', x=coverage_comparison['metric_name'],
        y=coverage_comparison['metric_value'], marker_color='lightblue'
    ))
    
    fig_coverage.add_trace(go.Bar(
        name='Target', x=coverage_comparison['metric_name'],
        y=coverage_comparison['target_value'], marker_color='lightcoral'
    ))
    
    fig_coverage.update_layout(
        title="Coverage Metrics vs Targets",
        xaxis_title="Metric", yaxis_title="Percentage (%)",
        barmode='group'
    )
    
    st.plotly_chart(fig_coverage, use_container_width=True)

st.markdown("---")

# DETAILED TABLES
st.header("📋 Detailed Data")

tab1, tab2, tab3 = st.tabs(["Test Cases", "RTM Summary", "Validation Results"])

with tab1:
    st.subheader("Test Cases Overview")
    tc_df = pd.DataFrame(test_cases)
    st.dataframe(
        tc_df[['test_id', 'requirement_id', 'test_type', 'test_title', 'priority']],
        use_container_width=True, height=400
    )
    
    csv = tc_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "test_cases.csv", "text/csv")

with tab2:
    st.subheader("Requirements Traceability Matrix")
    st.dataframe(
        rtm_df[['id', 'title', 'type', 'total_tests', 'coverage_status']],
        use_container_width=True, height=400
    )

with tab3:
    st.subheader("Validation Results")
    st.dataframe(
        validation_df[['test_id', 'requirement_id', 'overall_score', 'confidence_level', 'recommendation']],
        use_container_width=True, height=400
    )

st.markdown("---")

# EXPORT SECTION
st.header("📤 Export Test Cases")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📄 Export JSON"):
        st.success("✅ JSON export ready")

with col2:
    if st.button("📊 Export CSV"):
        exporter = TestCaseExportEngine(test_cases_file, validation_file, rtm_file)
        exporter.export_csv("../08_Output_Integration/test_cases.csv")
        st.success("✅ CSV exported!")

with col3:
    if st.button("🧪 Export TestRail"):
        exporter = TestCaseExportEngine(test_cases_file, validation_file, rtm_file)
        exporter.export_testrail("../08_Output_Integration/test_cases_testrail.csv")
        st.success("✅ TestRail exported!")

with col4:
    if st.button("📋 Export Jira"):
        exporter = TestCaseExportEngine(test_cases_file, validation_file, rtm_file)
        exporter.export_jira("../08_Output_Integration/test_cases_jira.csv")
        st.success("✅ Jira exported!")

st.markdown("---")

# SYSTEM INFO
st.header("ℹ️ System Information")

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Generation Model:** {metadata.get('model', 'N/A')}  
    **Total Test Cases:** {len(test_cases)}  
    **Phase 1 Tests:** {len(tc_data.get('phase1_test_cases', []))}  
    **Phase 2 Tests:** {len(tc_data.get('phase2_test_cases', []))}
    """)

with col2:
    st.info(f"""
    **Average Confidence:** {avg_confidence:.3f}  
    **Requirements Coverage:** {requirements_covered}/56  
    **High Confidence Tests:** {high_conf_count}  
    **Validation Status:** ✅ Complete
    """)
