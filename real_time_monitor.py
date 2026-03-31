"""
Real-Time Monitoring & Alert System for Exoplanet Discovery
Integrates with NASA TESS API for continuous monitoring and automated alerts
"""

import requests
import pandas as pd
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import streamlit as st
from typing import Dict, List, Tuple, Optional
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TESSDataIngestion:
    """Handles real-time data ingestion from NASA TESS archive"""
    
    def __init__(self):
        self.base_url = "https://exo.mast.stsci.edu/api/v0.1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Celestial-Circuitry-AEC/1.0'
        })
    
    def get_latest_observations(self, sector: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Fetch latest TESS observations"""
        try:
            url = f"{self.base_url}/tess/observations"
            params = {
                'limit': limit,
                'format': 'json'
            }
            if sector:
                params['sector'] = sector
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching TESS data: {e}")
            return []
    
    def get_lightcurve_data(self, target_id: str) -> Optional[pd.DataFrame]:
        """Download light curve data for a specific target"""
        try:
            url = f"{self.base_url}/tess/lightcurve/{target_id}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data:
                return pd.DataFrame(data['data'])
            return None
        except Exception as e:
            logger.error(f"Error fetching lightcurve for {target_id}: {e}")
            return None

class AlertSystem:
    """Handles automated alert generation and notifications"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.email_config = config.get('email', {})
        self.webhook_config = config.get('webhook', {})
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
    def send_email_alert(self, subject: str, body: str, recipients: List[str]):
        """Send email alert to specified recipients"""
        if not self.email_config.get('enabled', False):
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], 
                                self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], 
                        self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['sender'], recipients, text)
            server.quit()
            
            logger.info(f"Email alert sent to {recipients}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_webhook_alert(self, data: Dict):
        """Send webhook notification"""
        if not self.webhook_config.get('enabled', False):
            return
        
        try:
            response = requests.post(
                self.webhook_config['url'],
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            logger.info("Webhook alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def generate_alert(self, detection_data: Dict, model_confidence: float) -> Optional[Dict]:
        """Generate alert if confidence exceeds threshold"""
        if model_confidence < self.confidence_threshold:
            return None
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'target_id': detection_data.get('target_id', 'Unknown'),
            'confidence': model_confidence,
            'sector': detection_data.get('sector', 'Unknown'),
            'coordinates': detection_data.get('coordinates', {}),
            'predicted_period': detection_data.get('period', 'Unknown'),
            'predicted_depth': detection_data.get('transit_depth', 'Unknown'),
            'alert_level': self._get_alert_level(model_confidence)
        }
        
        return alert
    
    def _get_alert_level(self, confidence: float) -> str:
        """Determine alert level based on confidence"""
        if confidence >= 0.95:
            return "CRITICAL"
        elif confidence >= 0.85:
            return "HIGH"
        elif confidence >= 0.75:
            return "MEDIUM"
        else:
            return "LOW"

class RealTimeProcessor:
    """Main processor for real-time exoplanet detection"""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.tess_ingestion = TESSDataIngestion()
        self.alert_system = AlertSystem(config)
        self.detection_queue = queue.Queue()
        self.running = False
        self.processed_targets = set()
        
    def start_monitoring(self, check_interval: int = 300):  # 5 minutes default
        """Start real-time monitoring process"""
        self.running = True
        monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        monitoring_thread.start()
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        while self.running:
            try:
                # Get latest observations
                observations = self.tess_ingestion.get_latest_observations()
                
                for obs in observations:
                    target_id = obs.get('target_id')
                    if target_id and target_id not in self.processed_targets:
                        self._process_target(target_id, obs)
                        self.processed_targets.add(target_id)
                
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _process_target(self, target_id: str, observation_data: Dict):
        """Process a single target for exoplanet detection"""
        try:
            # Download light curve data
            lc_data = self.tess_ingestion.get_lightcurve_data(target_id)
            if lc_data is None or lc_data.empty:
                return
            
            # Prepare features for model
            features = self._prepare_features(lc_data)
            if features is None:
                return
            
            # Make prediction
            confidence = self.model.predict_proba(features)[0]
            prediction = self.model.predict(features)[0]
            
            # Store detection data
            detection_data = {
                'target_id': target_id,
                'sector': observation_data.get('sector'),
                'coordinates': {
                    'ra': observation_data.get('ra'),
                    'dec': observation_data.get('dec')
                },
                'period': self._estimate_period(lc_data),
                'transit_depth': self._estimate_transit_depth(lc_data),
                'prediction': prediction,
                'confidence': max(confidence)
            }
            
            # Generate alert if high confidence
            alert = self.alert_system.generate_alert(detection_data, max(confidence))
            if alert:
                self._handle_alert(alert, detection_data)
            
            # Add to detection queue for dashboard
            self.detection_queue.put(detection_data)
            
        except Exception as e:
            logger.error(f"Error processing target {target_id}: {e}")
    
    def _prepare_features(self, lc_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for model prediction"""
        try:
            # Ensure we have the required columns
            if 'flux' not in lc_data.columns or 'time' not in lc_data.columns:
                return None
            
            # Resample to match training data size (3197 points)
            target_size = 3197
            if len(lc_data) > target_size:
                lc_data = lc_data.sample(n=target_size).sort_values('time')
            elif len(lc_data) < target_size:
                # Pad with interpolation
                lc_data = lc_data.reindex(range(target_size))
                lc_data['flux'] = lc_data['flux'].interpolate()
                lc_data['time'] = lc_data['time'].interpolate()
            
            # Flatten the light curve
            from wotan import flatten
            flattened_flux, _ = flatten(lc_data['time'].values, lc_data['flux'].values, 
                                      window_length=0.5, return_trend=True)
            
            return flattened_flux.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _estimate_period(self, lc_data: pd.DataFrame) -> float:
        """Estimate orbital period using Lomb-Scargle periodogram"""
        try:
            from scipy.signal import find_peaks
            from astropy.timeseries import LombScargle
            
            time = lc_data['time'].values
            flux = lc_data['flux'].values
            
            # Remove NaN values
            mask = ~np.isnan(flux)
            time_clean = time[mask]
            flux_clean = flux[mask]
            
            if len(time_clean) < 10:
                return 0.0
            
            # Lomb-Scargle periodogram
            frequency = np.linspace(0.01, 10, 1000)  # 0.1 to 100 days
            power = LombScargle(time_clean, flux_clean).power(frequency)
            periods = 1 / frequency
            
            # Find peaks
            peaks, _ = find_peaks(power, height=np.max(power) * 0.1)
            
            if len(peaks) > 0:
                # Return the period with highest power
                best_peak = peaks[np.argmax(power[peaks])]
                return periods[best_peak]
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error estimating period: {e}")
            return 0.0
    
    def _estimate_transit_depth(self, lc_data: pd.DataFrame) -> float:
        """Estimate transit depth from light curve"""
        try:
            flux = lc_data['flux'].values
            flux_clean = flux[~np.isnan(flux)]
            
            if len(flux_clean) < 10:
                return 0.0
            
            # Calculate depth as difference between median and minimum
            median_flux = np.median(flux_clean)
            min_flux = np.min(flux_clean)
            
            depth = (median_flux - min_flux) / median_flux
            return max(0, depth)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error estimating transit depth: {e}")
            return 0.0
    
    def _handle_alert(self, alert: Dict, detection_data: Dict):
        """Handle generated alert"""
        # Send email alert
        subject = f"🚨 Exoplanet Alert: {alert['alert_level']} Confidence Detection"
        body = self._format_alert_email(alert, detection_data)
        
        recipients = self.alert_system.config.get('email', {}).get('recipients', [])
        if recipients:
            self.alert_system.send_email_alert(subject, body, recipients)
        
        # Send webhook alert
        self.alert_system.send_webhook_alert(alert)
        
        logger.info(f"Alert generated for target {alert['target_id']} with {alert['confidence']:.3f} confidence")
    
    def _format_alert_email(self, alert: Dict, detection_data: Dict) -> str:
        """Format alert email content"""
        return f"""
        <html>
        <body>
        <h2>🔭 Exoplanet Detection Alert</h2>
        <p><strong>Alert Level:</strong> {alert['alert_level']}</p>
        <p><strong>Target ID:</strong> {alert['target_id']}</p>
        <p><strong>Confidence:</strong> {alert['confidence']:.3f}</p>
        <p><strong>Timestamp:</strong> {alert['timestamp']}</p>
        <p><strong>Sector:</strong> {alert['sector']}</p>
        <p><strong>Coordinates:</strong> RA={alert['coordinates'].get('ra', 'N/A')}, Dec={alert['coordinates'].get('dec', 'N/A')}</p>
        <p><strong>Predicted Period:</strong> {alert['predicted_period']:.3f} days</p>
        <p><strong>Transit Depth:</strong> {alert['predicted_depth']:.4f}</p>
        
        <h3>Next Steps:</h3>
        <ul>
        <li>Verify detection with follow-up observations</li>
        <li>Check for false positives in known variable stars</li>
        <li>Consider spectroscopic follow-up for confirmation</li>
        </ul>
        
        <p>This alert was generated by the Celestial Circuitry Automated Exoplanet Classifier.</p>
        </body>
        </html>
        """
    
    def get_recent_detections(self, limit: int = 50) -> List[Dict]:
        """Get recent detections for dashboard display"""
        detections = []
        while not self.detection_queue.empty() and len(detections) < limit:
            try:
                detection = self.detection_queue.get_nowait()
                detections.append(detection)
            except queue.Empty:
                break
        
        return detections

# Configuration template
DEFAULT_CONFIG = {
    'confidence_threshold': 0.8,
    'email': {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': '',
        'password': '',
        'sender': '',
        'recipients': []
    },
    'webhook': {
        'enabled': False,
        'url': ''
    }
}
