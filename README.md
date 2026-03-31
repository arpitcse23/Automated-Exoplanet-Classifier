# Celestial Circuitry 🔭
## Advanced AI-Powered Exoplanet Discovery Platform

A sophisticated, AI-powered web application designed to automate and accelerate the search for exoplanets. This platform transforms the discovery process from months of manual work into seconds of automated analysis, democratizing the search for researchers, citizen scientists, and students alike.

## 🌟 Key Features

### 1. Real-Time Monitoring & Alert System
- **Continuous TESS Data Analysis**: Automated ingestion and processing of new TESS observations
- **Intelligent Alert System**: Email and webhook notifications for high-confidence detections
- **Live Dashboard**: Real-time monitoring of detection statistics and recent discoveries
- **Confidence Thresholds**: Configurable alert levels based on detection confidence

### 2. Multi-Planet System Discovery Engine
- **Transit Timing Variations (TTV) Analysis**: Detects gravitational interactions between planets
- **Resonant Chain Detection**: Identifies planets in orbital resonances using machine learning
- **Stability Prediction**: AI-powered assessment of long-term orbital stability
- **Signal Decomposition**: Separates overlapping transit signals from multiple planets

### 3. Real-Time Adaptive Observation Scheduler
- **Dynamic Target Prioritization**: AI-driven ranking based on confidence scores and scientific value
- **Multi-Telescope Coordination**: Optimizes observations across global telescope networks
- **Weather Integration**: Incorporates real-time weather predictions into scheduling
- **Resource Optimization**: Maximizes scientific return per observation hour

### 4. Atmospheric Biosignature Detector
- **Transmission Spectroscopy Analysis**: Analyzes atmospheric composition from light curves
- **Biosignature Molecule Detection**: Identifies O2, H2O, CH4, CO2, and other life indicators
- **Atmospheric Disequilibrium Analysis**: Detects chemical imbalances suggesting biological processes
- **Climate Modeling**: Predicts surface conditions and habitability potential

### 5. Autonomous Discovery Reporting System
- **AI Scientific Writing**: Generates publication-ready abstracts and paper sections
- **Statistical Analysis**: Calculates significance levels and uncertainty propagation
- **Impact Assessment**: Evaluates scientific importance and novelty of discoveries
- **Multi-Language Support**: Generates reports in multiple languages for global accessibility

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Automated-Exoplanet-Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Quick Start Guide

1. **Choose Analysis Mode**: Select from Basic Detection, Multi-Planet System, Biosignature Analysis, Real-Time Monitoring, or Full Analysis
2. **Upload Data**: Upload a light curve CSV file or use the provided sample data
3. **Configure Settings**: Adjust confidence thresholds and notification preferences
4. **Run Analysis**: Click the appropriate analysis buttons to explore different features
5. **View Results**: Explore the results through the interactive tabs and visualizations

## 📊 Data Format

The application expects CSV files with the following columns:
- `time`: Time values (any units, will be normalized)
- `flux`: Flux values (will be normalized to 1.0)

Example:
```csv
time,flux
0.0,1.000
0.1,0.999
0.2,1.001
...
```

## 🔧 Advanced Configuration

### Real-Time Monitoring Setup

1. **Email Alerts**: Configure SMTP settings in the Advanced Settings panel
2. **Webhook Notifications**: Set up webhook URLs for real-time notifications
3. **Confidence Thresholds**: Adjust detection sensitivity based on your needs

### Telescope Network Configuration

The adaptive scheduler supports multiple telescope configurations:
- **TESS**: Space-based photometry
- **Kepler**: High-precision photometry
- **JWST**: Infrared spectroscopy
- **Ground-based**: Custom telescope configurations

## 🧬 Biosignature Analysis

The platform analyzes transmission spectra for:
- **Oxygen (O2)**: Primary biosignature gas
- **Water (H2O)**: Essential for life as we know it
- **Methane (CH4)**: Potential biological marker
- **Carbon Dioxide (CO2)**: Atmospheric composition indicator
- **Ozone (O3)**: Secondary biosignature

## 📈 Multi-Planet Detection

Advanced algorithms detect:
- **Transit Timing Variations**: Gravitational interactions between planets
- **Orbital Resonances**: Planets in integer period ratios (2:1, 3:2, etc.)
- **System Stability**: Long-term orbital evolution predictions
- **Signal Separation**: Individual planet characteristics from overlapping transits

## 🔬 Scientific Output

The platform generates:
- **Publication-Ready Abstracts**: Following astronomical journal standards
- **Statistical Significance Tests**: Proper uncertainty propagation
- **Impact Assessments**: Scientific importance evaluation
- **Discovery Reports**: Comprehensive analysis summaries

## 🌐 API Integration

### TESS Data Access
- Real-time data ingestion from NASA's TESS archive
- Automated light curve processing
- Quality assessment and filtering

### Weather Services
- OpenWeatherMap integration for telescope scheduling
- Real-time seeing quality predictions
- Cloud cover and visibility monitoring

## 📚 Scientific Background

This platform implements cutting-edge techniques from:
- **Exoplanet Detection**: Transit photometry and timing analysis
- **Atmospheric Science**: Transmission spectroscopy and climate modeling
- **Machine Learning**: Deep learning for pattern recognition
- **Statistical Analysis**: Bayesian inference and significance testing

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature proposals

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA TESS mission for providing high-quality light curve data
- The exoplanet research community for advancing detection techniques
- Open source contributors who made this project possible

## 📞 Support

For technical support or scientific questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

## 🔮 Future Developments

Planned features include:
- **Machine Learning Improvements**: Enhanced detection algorithms
- **Additional Telescopes**: Support for more observatory networks
- **Advanced Spectroscopy**: Higher resolution atmospheric analysis
- **Citizen Science Integration**: Crowdsourced analysis capabilities

---

**Celestial Circuitry** - Transforming exoplanet discovery through artificial intelligence.

*Built for the NASA Space Apps Challenge by Team Celestial Circuitry*


