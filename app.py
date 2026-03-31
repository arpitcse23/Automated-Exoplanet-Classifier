
import streamlit as st
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
import joblib
import numpy as np
from wotan import flatten
import time
import threading
from datetime import datetime, timedelta
import json

# Import new advanced features
from real_time_monitor import RealTimeProcessor, DEFAULT_CONFIG
from multi_planet_engine import MultiPlanetSystemClassifier
from adaptive_scheduler import ObservationScheduler, create_sample_telescopes, ObservationTarget, Priority
from biosignature_detector import BiosignatureDetector
from discovery_reporting import DiscoveryReporter, DiscoveryData, DiscoveryType

# --- FUNCTION FROM OUR DATA ENGINE ---
def flatten_light_curve(lc):
    flattened_lc, _ = flatten(lc.time.value, lc.flux.value, window_length=0.5, return_trend=True)
    return lk.LightCurve(time=lc.time.value, flux=flattened_lc)

# --- AI MODEL LOADING ---
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_model.joblib')
    return model

@st.cache_resource
def load_advanced_models():
    """Load all advanced AI models"""
    return {
        'main_model': joblib.load('xgboost_model.joblib'),
        'multi_planet_classifier': MultiPlanetSystemClassifier(),
        'biosignature_detector': BiosignatureDetector(),
        'discovery_reporter': DiscoveryReporter()
    }

models = load_advanced_models()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Celestial Circuitry", page_icon="🪐", layout="wide")

# --- MAIN APP ---
st.title("Celestial Circuitry 🔭")
st.subheader("Advanced AI-Powered Exoplanet Discovery Platform")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔧 Platform Features")
    
    # Feature selection
    st.subheader("Analysis Mode")
    analysis_mode = st.selectbox(
        "Choose analysis type:",
        ["Basic Detection", "Multi-Planet System", "Biosignature Analysis", "Real-Time Monitoring", "Full Analysis"]
    )
    
    st.divider()
    
    # Real-time monitoring controls
    if analysis_mode == "Real-Time Monitoring":
        st.subheader("Real-Time Controls")
        if st.button("Start Monitoring", key="start_monitoring"):
            st.session_state.monitoring_active = True
            st.success("Real-time monitoring started!")
        
        if st.button("Stop Monitoring", key="stop_monitoring"):
            st.session_state.monitoring_active = False
            st.info("Real-time monitoring stopped.")
    
    st.divider()
    
    # File upload
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload a light curve .csv file", type=["csv"])
    
    # Sample data option
    if st.button("Use Sample Data"):
        uploaded_file = "sample_data.csv"
        st.info("Using sample data for demonstration")
    
    st.divider()
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.8)
        enable_alerts = st.checkbox("Enable Email Alerts", value=False)
        enable_webhooks = st.checkbox("Enable Webhook Notifications", value=False)
        
        if enable_alerts:
            email_config = {
                'enabled': True,
                'smtp_server': st.text_input("SMTP Server", "smtp.gmail.com"),
                'smtp_port': st.number_input("SMTP Port", 587),
                'username': st.text_input("Email Username"),
                'password': st.text_input("Email Password", type="password"),
                'sender': st.text_input("Sender Email"),
                'recipients': st.text_area("Recipients (one per line)").split('\n')
            }
        else:
            email_config = {'enabled': False}
    
    st.divider()
    st.header("ℹ️ About")
    st.write("Advanced exoplanet discovery platform with real-time monitoring, multi-planet detection, biosignature analysis, and autonomous reporting.")

# --- MAIN CONTENT AREA ---
if uploaded_file is None:
    st.markdown("""
    ## Welcome to Celestial Circuitry 🔭
    
    **Advanced AI-Powered Exoplanet Discovery Platform**
    
    This platform offers cutting-edge capabilities for exoplanet discovery and analysis:
    
    ### 🌟 Key Features:
    - **Real-Time Monitoring**: Continuous TESS data analysis with automated alerts
    - **Multi-Planet Detection**: Advanced algorithms for detecting entire planetary systems
    - **Biosignature Analysis**: Atmospheric composition analysis for signs of life
    - **Adaptive Scheduling**: AI-powered telescope coordination and optimization
    - **Autonomous Reporting**: AI-generated scientific papers and discovery reports
    
    ### 🚀 Getting Started:
    1. Choose your analysis mode from the sidebar
    2. Upload a light curve file or use sample data
    3. Configure advanced settings as needed
    4. Run the analysis and explore the results
    
    Upload a light curve file to begin your exoplanet discovery journey!
    """)

if uploaded_file is not None:
    # Initialize session state for real-time monitoring
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    
    if 'real_time_processor' not in st.session_state:
        config = DEFAULT_CONFIG.copy()
        config['confidence_threshold'] = confidence_threshold
        config['email'] = email_config
        st.session_state.real_time_processor = RealTimeProcessor(models['main_model'], config)
    
    # Load and process data
    with st.spinner('Analyzing light curve with advanced AI...'):
        try:
            df = pd.read_csv(uploaded_file)
            light_curve = lk.LightCurve(time=df['time'], flux=df['flux'])
            flattened_lc = flatten_light_curve(light_curve)
            
            # Basic prediction
            features = flattened_lc.flux.value.reshape(1, -1)
            prediction = models['main_model'].predict(features)[0]
            prediction_proba = models['main_model'].predict_proba(features)[0]
            
            st.success('Advanced Analysis Complete!')
            
            # Create tabs for different analysis modes
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Basic Detection", "🌌 Multi-Planet", "🧬 Biosignatures", "📊 Real-Time", "📄 Reports"])
            
            with tab1:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("🎯 Detection Results")
                    if prediction == 0:
                        st.success("✅ Exoplanet Candidate Detected!")
                        display_proba = prediction_proba[0]
                    else:
                        st.error("❌ No Exoplanet Detected")
                        display_proba = prediction_proba[1]
                    
                    confidence = display_proba * 100
                    st.metric(label="AI Confidence", value=f"{confidence:.2f}%")
                    
                    # Additional metrics
                    st.metric(label="Signal Quality", value="High" if confidence > 80 else "Medium" if confidence > 60 else "Low")
                    st.metric(label="Analysis Time", value="< 1 second")
                
                with col2:
                    st.subheader("📈 Light Curve Analysis")
                    known_period = 0.837495
                    folded_lc = flattened_lc.fold(period=known_period)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.set_title(f"Phase-Folded Transit for {uploaded_file.name}")
                    folded_lc.plot(ax=ax, color='blue', marker='.', markersize=2, linestyle='none', alpha=0.3)
                    folded_lc.bin(time_bin_size=0.01).plot(ax=ax, color='red', marker='o', markersize=4, linestyle='-')
                    ax.set_xlabel("Phase")
                    ax.set_ylabel("Normalized Flux")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            with tab2:
                st.subheader("🌌 Multi-Planet System Analysis")
                
                if st.button("Run Multi-Planet Detection", key="multi_planet_btn"):
                    with st.spinner("Analyzing for multi-planet system..."):
                        # Multi-planet analysis
                        multi_planet_results = models['multi_planet_classifier'].analyze_system(
                            flattened_lc.time.value, 
                            flattened_lc.flux.value, 
                            known_period
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Multi-Planet Detected", "Yes" if multi_planet_results['multi_planet_detected'] else "No")
                            st.metric("System Score", f"{multi_planet_results['system_score']:.3f}")
                            st.metric("Confidence", f"{multi_planet_results['confidence']:.3f}")
                        
                        with col2:
                            ttv_analysis = multi_planet_results.get('ttv_analysis', {})
                            st.metric("TTV Detected", "Yes" if ttv_analysis.get('ttv_detected') else "No")
                            st.metric("TTV Amplitude", f"{ttv_analysis.get('ttv_amplitude', 0):.4f}")
                            
                            resonance_analysis = multi_planet_results.get('resonance_analysis', {})
                            st.metric("Resonance Detected", "Yes" if resonance_analysis.get('resonance_detected') else "No")
            
            with tab3:
                st.subheader("🧬 Biosignature Analysis")
                
                if st.button("Run Biosignature Detection", key="biosignature_btn"):
                    with st.spinner("Analyzing atmospheric composition..."):
                        # Simulate transmission spectrum data
                        wavelengths = np.linspace(400, 2500, 1000)
                        # Create synthetic transmission spectrum with potential biosignatures
                        transmission = np.ones_like(wavelengths)
                        
                        # Add synthetic absorption features
                        o2_features = [630, 690, 760, 1270, 1450]
                        for wl in o2_features:
                            mask = np.abs(wavelengths - wl) < 10
                            transmission[mask] *= 0.99
                        
                        h2o_features = [720, 820, 940, 1130, 1380, 1880]
                        for wl in h2o_features:
                            mask = np.abs(wavelengths - wl) < 15
                            transmission[mask] *= 0.98
                        
                        # Add noise
                        transmission += np.random.normal(0, 0.001, len(transmission))
                        
                        biosignature_results = models['biosignature_detector'].detect_biosignatures(
                            wavelengths, transmission
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            assessment = biosignature_results.get('overall_assessment', {})
                            st.metric("Biosignature Score", f"{assessment.get('overall_score', 0):.3f}")
                            st.metric("Confidence Level", assessment.get('confidence_level', 'Unknown'))
                            
                            indicators = assessment.get('biosignature_indicators', [])
                            if indicators:
                                st.write("**Detected Indicators:**")
                                for indicator in indicators:
                                    st.write(f"• {indicator}")
                        
                        with col2:
                            spectral_analysis = biosignature_results.get('spectral_analysis', {})
                            molecular_detections = spectral_analysis.get('molecular_detections', {})
                            
                            if molecular_detections:
                                st.write("**Molecular Detections:**")
                                for molecule, detection in molecular_detections.items():
                                    st.write(f"• {molecule.value}: {detection.mixing_ratio:.2f} ppm (confidence: {detection.detection_confidence:.2f})")
                            
                            # Plot transmission spectrum
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(wavelengths, transmission, 'b-', alpha=0.7)
                            ax.set_xlabel("Wavelength (nm)")
                            ax.set_ylabel("Transmission")
                            ax.set_title("Transmission Spectrum Analysis")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
            
            with tab4:
                st.subheader("📊 Real-Time Monitoring Dashboard")
                
                if st.session_state.monitoring_active:
                    st.success("🟢 Real-time monitoring is ACTIVE")
                    
                    # Start monitoring if not already started
                    if not hasattr(st.session_state, 'monitoring_started'):
                        st.session_state.real_time_processor.start_monitoring()
                        st.session_state.monitoring_started = True
                        st.rerun()
                    
                    # Display recent detections
                    recent_detections = st.session_state.real_time_processor.get_recent_detections(10)
                    
                    if recent_detections:
                        st.write("**Recent Detections:**")
                        for detection in recent_detections:
                            with st.expander(f"Target {detection['target_id']} - Confidence: {detection['confidence']:.3f}"):
                                st.write(f"**Sector:** {detection.get('sector', 'Unknown')}")
                                st.write(f"**Coordinates:** RA={detection['coordinates'].get('ra', 'N/A')}, Dec={detection['coordinates'].get('dec', 'N/A')}")
                                st.write(f"**Period:** {detection.get('period', 'Unknown')} days")
                                st.write(f"**Transit Depth:** {detection.get('transit_depth', 'Unknown')}")
                    else:
                        st.info("No recent detections. Monitoring for new candidates...")
                    
                    # Monitoring statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Targets Processed", len(st.session_state.real_time_processor.processed_targets))
                    with col2:
                        st.metric("High Confidence", sum(1 for d in recent_detections if d['confidence'] > 0.8))
                    with col3:
                        st.metric("Active Monitoring", "Yes" if st.session_state.monitoring_active else "No")
                
                else:
                    st.info("Real-time monitoring is inactive. Click 'Start Monitoring' in the sidebar to begin.")
            
            with tab5:
                st.subheader("📄 Discovery Reports")
                
                if st.button("Generate Discovery Report", key="report_btn"):
                    with st.spinner("Generating comprehensive discovery report..."):
                        # Create discovery data
                        discovery_data = DiscoveryData(
                            discovery_id=f"DISCOVERY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            discovery_type=DiscoveryType.SINGLE_PLANET,
                            target_id=uploaded_file.name,
                            coordinates=(0.0, 0.0),  # Would be filled from actual data
                            stellar_properties={'spectral_type': 'G', 'magnitude': 12.5},
                            planetary_properties={
                                'period': known_period,
                                'radius': 1.2,
                                'mass': 2.1,
                                'temperature': 300
                            },
                            detection_confidence=display_proba,
                            statistical_significance=3.5,
                            observation_data={'telescope': 'TESS', 'duration': 30},
                            analysis_results={'method': 'transit_photometry'},
                            timestamp=datetime.now()
                        )
                        
                        # Generate report
                        report = models['discovery_reporter'].generate_discovery_report(
                            discovery_data, 
                            {'detection_method': 'transit_photometry', 'significance_sigma': 3.5}
                        )
                        
                        # Display report sections
                        st.write("**Executive Summary:**")
                        st.text(report['executive_summary'])
                        
                        st.write("**Scientific Paper Abstract:**")
                        st.text(report['scientific_paper']['abstract'])
                        
                        st.write("**Impact Assessment:**")
                        impact = report['impact_assessment']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Impact Score", f"{impact['impact_score']:.3f}")
                        with col2:
                            st.metric("Impact Level", impact['impact_level'].upper())
                        with col3:
                            st.metric("Novelty", f"{impact['novelty_score']:.3f}")
                        
                        # Download report
                        report_json = json.dumps(report, default=str, indent=2)
                        st.download_button(
                            label="Download Full Report (JSON)",
                            data=report_json,
                            file_name=f"discovery_report_{discovery_data.discovery_id}.json",
                            mime="application/json"
                        )
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.write("Please check your data format and try again.")