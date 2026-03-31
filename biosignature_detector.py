"""
Atmospheric Biosignature Detector
Analyzes spectroscopic data to search for signs of life in exoplanet atmospheres
"""

import numpy as np
import pandas as pd
from scipy import signal, optimize
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes for when TensorFlow is not available
    class tf:
        @staticmethod
        def keras():
            return None
    class keras:
        class Sequential:
            def __init__(self, *args, **kwargs):
                pass
        class layers:
            @staticmethod
            def Conv1D(*args, **kwargs):
                return None
            @staticmethod
            def MaxPooling1D(*args, **kwargs):
                return None
            @staticmethod
            def GlobalMaxPooling1D(*args, **kwargs):
                return None
            @staticmethod
            def Dense(*args, **kwargs):
                return None
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BiosignatureType(Enum):
    OXYGEN = "O2"
    WATER = "H2O"
    METHANE = "CH4"
    CARBON_DIOXIDE = "CO2"
    OZONE = "O3"
    NITROUS_OXIDE = "N2O"
    CARBON_MONOXIDE = "CO"
    AMMONIA = "NH3"
    SULFUR_DIOXIDE = "SO2"

@dataclass
class AtmosphericModel:
    """Represents atmospheric composition and properties"""
    temperature: float  # K
    pressure: float  # Pa
    composition: Dict[str, float]  # molecule -> mixing ratio
    scale_height: float  # km
    mean_molecular_weight: float  # g/mol

@dataclass
class BiosignatureDetection:
    """Results of biosignature detection analysis"""
    molecule: BiosignatureType
    detection_confidence: float
    mixing_ratio: float
    significance: float
    wavelength_range: Tuple[float, float]
    spectral_features: List[Dict]

class SpectralAnalyzer:
    """Analyzes transmission spectroscopy data for atmospheric composition"""
    
    def __init__(self):
        self.molecular_database = self._load_molecular_database()
        self.spectral_resolution = 0.01  # nm
        
    def _load_molecular_database(self) -> Dict:
        """Load molecular absorption line database"""
        # Simplified molecular line database
        # In practice, this would load from HITRAN or similar database
        return {
            BiosignatureType.OXYGEN: {
                'wavelengths': np.array([630.0, 690.0, 760.0, 1270.0, 1450.0]),
                'strengths': np.array([0.1, 0.05, 0.2, 0.15, 0.08]),
                'widths': np.array([0.5, 0.3, 0.8, 0.4, 0.2])
            },
            BiosignatureType.WATER: {
                'wavelengths': np.array([720.0, 820.0, 940.0, 1130.0, 1380.0, 1880.0]),
                'strengths': np.array([0.3, 0.4, 0.6, 0.8, 0.9, 0.7]),
                'widths': np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.4])
            },
            BiosignatureType.METHANE: {
                'wavelengths': np.array([890.0, 1000.0, 1300.0, 1600.0, 2200.0, 3300.0]),
                'strengths': np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.4]),
                'widths': np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.2])
            },
            BiosignatureType.CARBON_DIOXIDE: {
                'wavelengths': np.array([1500.0, 2000.0, 2300.0, 2800.0, 4300.0]),
                'strengths': np.array([0.4, 0.5, 0.6, 0.7, 0.3]),
                'widths': np.array([0.3, 0.4, 0.5, 0.6, 0.4])
            },
            BiosignatureType.OZONE: {
                'wavelengths': np.array([255.0, 310.0, 600.0, 1000.0]),
                'strengths': np.array([0.8, 0.6, 0.3, 0.1]),
                'widths': np.array([0.1, 0.2, 0.3, 0.4])
            }
        }
    
    def analyze_transmission_spectrum(self, wavelengths: np.ndarray, 
                                    transmission: np.ndarray,
                                    error_bars: np.ndarray = None) -> Dict:
        """Analyze transmission spectrum for atmospheric composition"""
        try:
            if error_bars is None:
                error_bars = np.ones_like(transmission) * 0.01  # 1% error
            
            # Detect molecular features
            molecular_detections = {}
            
            for molecule, line_data in self.molecular_database.items():
                detection = self._detect_molecular_features(
                    wavelengths, transmission, error_bars, molecule, line_data
                )
                if detection:
                    molecular_detections[molecule] = detection
            
            # Calculate atmospheric properties
            atmospheric_properties = self._calculate_atmospheric_properties(
                wavelengths, transmission, molecular_detections
            )
            
            # Assess biosignature potential
            biosignature_assessment = self._assess_biosignature_potential(
                molecular_detections, atmospheric_properties
            )
            
            return {
                'molecular_detections': molecular_detections,
                'atmospheric_properties': atmospheric_properties,
                'biosignature_assessment': biosignature_assessment,
                'spectral_quality': self._assess_spectral_quality(transmission, error_bars)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transmission spectrum: {e}")
            return {}
    
    def _detect_molecular_features(self, wavelengths: np.ndarray, 
                                 transmission: np.ndarray, 
                                 error_bars: np.ndarray,
                                 molecule: BiosignatureType, 
                                 line_data: Dict) -> Optional[BiosignatureDetection]:
        """Detect specific molecular features in spectrum"""
        try:
            line_wavelengths = line_data['wavelengths']
            line_strengths = line_data['strengths']
            line_widths = line_data['widths']
            
            detections = []
            total_confidence = 0.0
            total_significance = 0.0
            
            for i, (line_wl, strength, width) in enumerate(zip(line_wavelengths, line_strengths, line_widths)):
                # Find wavelength range for this line
                wl_range = (line_wl - 3*width, line_wl + 3*width)
                mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
                
                if not np.any(mask):
                    continue
                
                # Extract spectrum in this range
                wl_subset = wavelengths[mask]
                trans_subset = transmission[mask]
                error_subset = error_bars[mask]
                
                # Fit absorption line
                line_fit = self._fit_absorption_line(wl_subset, trans_subset, error_subset, 
                                                   line_wl, strength, width)
                
                if line_fit:
                    detections.append(line_fit)
                    total_confidence += line_fit['confidence']
                    total_significance += line_fit['significance']
            
            if not detections:
                return None
            
            # Calculate average mixing ratio
            mixing_ratios = [d['mixing_ratio'] for d in detections]
            avg_mixing_ratio = np.mean(mixing_ratios)
            
            # Calculate overall confidence and significance
            avg_confidence = total_confidence / len(detections)
            avg_significance = total_significance / len(detections)
            
            return BiosignatureDetection(
                molecule=molecule,
                detection_confidence=avg_confidence,
                mixing_ratio=avg_mixing_ratio,
                significance=avg_significance,
                wavelength_range=(min(line_wavelengths), max(line_wavelengths)),
                spectral_features=detections
            )
            
        except Exception as e:
            logger.error(f"Error detecting {molecule.value} features: {e}")
            return None
    
    def _fit_absorption_line(self, wavelengths: np.ndarray, 
                           transmission: np.ndarray, 
                           error_bars: np.ndarray,
                           center_wl: float, 
                           expected_strength: float, 
                           expected_width: float) -> Optional[Dict]:
        """Fit absorption line to spectrum data"""
        try:
            # Define line profile function (Gaussian)
            def line_profile(wl, center, depth, width):
                return 1.0 - depth * np.exp(-0.5 * ((wl - center) / width)**2)
            
            # Initial parameter guess
            p0 = [center_wl, 0.01, expected_width]  # [center, depth, width]
            
            # Fit the line
            try:
                popt, pcov = optimize.curve_fit(
                    line_profile, wavelengths, transmission, 
                    p0=p0, sigma=error_bars, maxfev=1000
                )
                
                center, depth, width = popt
                
                # Calculate confidence and significance
                confidence = self._calculate_line_confidence(depth, error_bars)
                significance = self._calculate_line_significance(
                    wavelengths, transmission, error_bars, line_profile, popt
                )
                
                # Calculate mixing ratio (simplified)
                mixing_ratio = depth * 1000  # Convert to ppm
                
                return {
                    'center': center,
                    'depth': depth,
                    'width': width,
                    'confidence': confidence,
                    'significance': significance,
                    'mixing_ratio': mixing_ratio
                }
                
            except (RuntimeError, ValueError):
                return None
                
        except Exception as e:
            logger.error(f"Error fitting absorption line: {e}")
            return None
    
    def _calculate_line_confidence(self, depth: float, error_bars: np.ndarray) -> float:
        """Calculate confidence in line detection"""
        if depth <= 0:
            return 0.0
        
        # Signal-to-noise ratio
        snr = depth / np.mean(error_bars)
        
        # Convert to confidence (0-1)
        confidence = min(1.0, max(0.0, snr / 5.0))  # 5σ = 100% confidence
        
        return confidence
    
    def _calculate_line_significance(self, wavelengths: np.ndarray, 
                                   transmission: np.ndarray, 
                                   error_bars: np.ndarray,
                                   model_func, params) -> float:
        """Calculate statistical significance of line detection"""
        try:
            # Calculate chi-squared
            model_transmission = model_func(wavelengths, *params)
            chi2_stat = np.sum(((transmission - model_transmission) / error_bars)**2)
            
            # Degrees of freedom
            dof = len(wavelengths) - len(params)
            
            # Calculate p-value
            p_value = 1.0 - chi2.cdf(chi2_stat, dof)
            
            # Convert to significance (sigma)
            significance = np.sqrt(2) * np.sqrt(-np.log(p_value)) if p_value > 0 else 10.0
            
            return min(10.0, max(0.0, significance))
            
        except Exception as e:
            logger.error(f"Error calculating line significance: {e}")
            return 0.0
    
    def _calculate_atmospheric_properties(self, wavelengths: np.ndarray, 
                                        transmission: np.ndarray,
                                        molecular_detections: Dict) -> Dict:
        """Calculate atmospheric properties from spectrum"""
        try:
            # Estimate atmospheric scale height
            scale_height = self._estimate_scale_height(wavelengths, transmission)
            
            # Calculate mean molecular weight
            mean_molecular_weight = self._calculate_mean_molecular_weight(molecular_detections)
            
            # Estimate temperature (simplified)
            temperature = self._estimate_temperature(molecular_detections)
            
            # Estimate pressure
            pressure = self._estimate_pressure(scale_height, mean_molecular_weight, temperature)
            
            return {
                'scale_height': scale_height,
                'mean_molecular_weight': mean_molecular_weight,
                'temperature': temperature,
                'pressure': pressure,
                'atmospheric_thickness': scale_height * 8  # 8 scale heights
            }
            
        except Exception as e:
            logger.error(f"Error calculating atmospheric properties: {e}")
            return {}
    
    def _estimate_scale_height(self, wavelengths: np.ndarray, 
                              transmission: np.ndarray) -> float:
        """Estimate atmospheric scale height from spectrum slope"""
        try:
            # Calculate slope of transmission vs wavelength
            # Steeper slope indicates thicker atmosphere
            slope = np.polyfit(wavelengths, transmission, 1)[0]
            
            # Convert slope to scale height (simplified relationship)
            scale_height = abs(slope) * 1000  # km
            
            return max(5.0, min(50.0, scale_height))  # Reasonable range
            
        except Exception as e:
            logger.error(f"Error estimating scale height: {e}")
            return 10.0  # Default value
    
    def _calculate_mean_molecular_weight(self, molecular_detections: Dict) -> float:
        """Calculate mean molecular weight from detected molecules"""
        # Molecular weights (g/mol)
        molecular_weights = {
            BiosignatureType.OXYGEN: 32.0,
            BiosignatureType.WATER: 18.0,
            BiosignatureType.METHANE: 16.0,
            BiosignatureType.CARBON_DIOXIDE: 44.0,
            BiosignatureType.OZONE: 48.0
        }
        
        if not molecular_detections:
            return 28.0  # Default (N2-dominated)
        
        total_weight = 0.0
        total_abundance = 0.0
        
        for molecule, detection in molecular_detections.items():
            if molecule in molecular_weights:
                weight = molecular_weights[molecule]
                abundance = detection.mixing_ratio
                
                total_weight += weight * abundance
                total_abundance += abundance
        
        if total_abundance > 0:
            return total_weight / total_abundance
        else:
            return 28.0
    
    def _estimate_temperature(self, molecular_detections: Dict) -> float:
        """Estimate atmospheric temperature from molecular detections"""
        # Simplified temperature estimation based on molecular abundances
        # In practice, this would use more sophisticated atmospheric modeling
        
        base_temp = 300.0  # K
        
        # Adjust based on detected molecules
        if BiosignatureType.WATER in molecular_detections:
            water_abundance = molecular_detections[BiosignatureType.WATER].mixing_ratio
            if water_abundance > 1000:  # ppm
                base_temp += 50  # Warmer with more water
        
        if BiosignatureType.CARBON_DIOXIDE in molecular_detections:
            co2_abundance = molecular_detections[BiosignatureType.CARBON_DIOXIDE].mixing_ratio
            if co2_abundance > 10000:  # ppm
                base_temp += 30  # Greenhouse effect
        
        return base_temp
    
    def _estimate_pressure(self, scale_height: float, 
                          mean_molecular_weight: float, 
                          temperature: float) -> float:
        """Estimate atmospheric pressure from scale height"""
        # Using scale height formula: H = kT/(mg)
        # where k = Boltzmann constant, T = temperature, m = molecular weight, g = gravity
        
        k = 1.38e-23  # J/K
        g = 9.8  # m/s² (assuming Earth-like gravity)
        m = mean_molecular_weight * 1.66e-27  # kg (convert from g/mol)
        
        # Calculate pressure at surface (simplified)
        pressure = (k * temperature) / (m * g) * 1e-5  # Convert to Pa
        
        return max(1000, min(1000000, pressure))  # Reasonable range
    
    def _assess_biosignature_potential(self, molecular_detections: Dict, 
                                      atmospheric_properties: Dict) -> Dict:
        """Assess potential for biosignatures"""
        try:
            biosignature_score = 0.0
            biosignature_indicators = []
            
            # Check for oxygen
            if BiosignatureType.OXYGEN in molecular_detections:
                o2_detection = molecular_detections[BiosignatureType.OXYGEN]
                if o2_detection.mixing_ratio > 1000:  # > 0.1%
                    biosignature_score += 0.3
                    biosignature_indicators.append("High O2 abundance")
            
            # Check for ozone
            if BiosignatureType.OZONE in molecular_detections:
                o3_detection = molecular_detections[BiosignatureType.OZONE]
                if o3_detection.mixing_ratio > 100:  # > 0.01%
                    biosignature_score += 0.2
                    biosignature_indicators.append("Ozone detected")
            
            # Check for water
            if BiosignatureType.WATER in molecular_detections:
                h2o_detection = molecular_detections[BiosignatureType.WATER]
                if h2o_detection.mixing_ratio > 1000:  # > 0.1%
                    biosignature_score += 0.2
                    biosignature_indicators.append("Water vapor present")
            
            # Check for methane
            if BiosignatureType.METHANE in molecular_detections:
                ch4_detection = molecular_detections[BiosignatureType.METHANE]
                if ch4_detection.mixing_ratio > 10:  # > 1 ppm
                    biosignature_score += 0.1
                    biosignature_indicators.append("Methane detected")
            
            # Check for atmospheric disequilibrium
            disequilibrium_score = self._assess_atmospheric_disequilibrium(molecular_detections)
            biosignature_score += disequilibrium_score * 0.2
            
            # Check for habitable zone compatibility
            habitable_score = self._assess_habitable_zone_compatibility(atmospheric_properties)
            biosignature_score += habitable_score * 0.1
            
            return {
                'biosignature_score': min(1.0, biosignature_score),
                'biosignature_indicators': biosignature_indicators,
                'disequilibrium_score': disequilibrium_score,
                'habitable_zone_score': habitable_score,
                'life_probability': self._calculate_life_probability(biosignature_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing biosignature potential: {e}")
            return {'biosignature_score': 0.0, 'life_probability': 0.0}
    
    def _assess_atmospheric_disequilibrium(self, molecular_detections: Dict) -> float:
        """Assess atmospheric disequilibrium (indicator of life)"""
        try:
            # Check for O2-CH4 disequilibrium
            o2_present = BiosignatureType.OXYGEN in molecular_detections
            ch4_present = BiosignatureType.METHANE in molecular_detections
            
            if o2_present and ch4_present:
                o2_abundance = molecular_detections[BiosignatureType.OXYGEN].mixing_ratio
                ch4_abundance = molecular_detections[BiosignatureType.METHANE].mixing_ratio
                
                # Both O2 and CH4 present indicates disequilibrium
                if o2_abundance > 1000 and ch4_abundance > 10:
                    return 1.0
            
            # Check for N2O presence (strong biosignature)
            if BiosignatureType.NITROUS_OXIDE in molecular_detections:
                n2o_abundance = molecular_detections[BiosignatureType.NITROUS_OXIDE].mixing_ratio
                if n2o_abundance > 1:  # > 0.1 ppm
                    return 0.8
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error assessing atmospheric disequilibrium: {e}")
            return 0.0
    
    def _assess_habitable_zone_compatibility(self, atmospheric_properties: Dict) -> float:
        """Assess compatibility with habitable zone conditions"""
        try:
            temperature = atmospheric_properties.get('temperature', 300)
            pressure = atmospheric_properties.get('pressure', 100000)
            
            # Habitable zone temperature range (simplified)
            if 250 < temperature < 350:  # K
                temp_score = 1.0
            elif 200 < temperature < 400:
                temp_score = 0.5
            else:
                temp_score = 0.0
            
            # Habitable pressure range
            if 10000 < pressure < 1000000:  # Pa
                pressure_score = 1.0
            elif 1000 < pressure < 10000000:
                pressure_score = 0.5
            else:
                pressure_score = 0.0
            
            return (temp_score + pressure_score) / 2.0
            
        except Exception as e:
            logger.error(f"Error assessing habitable zone compatibility: {e}")
            return 0.0
    
    def _calculate_life_probability(self, biosignature_score: float) -> float:
        """Calculate probability of life based on biosignature score"""
        # Simple sigmoid function to convert score to probability
        return 1.0 / (1.0 + np.exp(-10 * (biosignature_score - 0.5)))
    
    def _assess_spectral_quality(self, transmission: np.ndarray, 
                               error_bars: np.ndarray) -> float:
        """Assess quality of spectral data"""
        try:
            # Calculate signal-to-noise ratio
            snr = np.mean(np.abs(transmission - 1.0)) / np.mean(error_bars)
            
            # Calculate spectral resolution
            resolution = np.std(transmission)
            
            # Calculate data completeness
            completeness = np.sum(~np.isnan(transmission)) / len(transmission)
            
            # Combine metrics
            quality_score = (snr / 10.0 + resolution * 10 + completeness) / 3.0
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error assessing spectral quality: {e}")
            return 0.5

class ClimateModeler:
    """Model planetary climate and surface conditions"""
    
    def __init__(self):
        self.climate_model = self._build_climate_model()
    
    def _build_climate_model(self):
        """Build neural network for climate modeling"""
        if TENSORFLOW_AVAILABLE:
            # Use TensorFlow if available
            model = keras.Sequential([
                keras.layers.Dense(100, activation='relu', input_shape=(10,)),
                keras.layers.Dense(50, activation='relu'),
                keras.layers.Dense(25, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='mse')
            return model
        else:
            # Fallback to scikit-learn
            logger.warning("TensorFlow not available, using scikit-learn for climate modeling")
            return MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
    
    def model_planetary_climate(self, atmospheric_properties: Dict, 
                              stellar_properties: Dict) -> Dict:
        """Model planetary climate and surface conditions"""
        try:
            # Extract input parameters
            temperature = atmospheric_properties.get('temperature', 300)
            pressure = atmospheric_properties.get('pressure', 100000)
            composition = atmospheric_properties.get('composition', {})
            
            stellar_temp = stellar_properties.get('temperature', 5800)
            stellar_luminosity = stellar_properties.get('luminosity', 1.0)
            orbital_distance = stellar_properties.get('orbital_distance', 1.0)
            
            # Calculate equilibrium temperature
            equilibrium_temp = self._calculate_equilibrium_temperature(
                stellar_temp, stellar_luminosity, orbital_distance
            )
            
            # Calculate greenhouse effect
            greenhouse_factor = self._calculate_greenhouse_effect(composition, pressure)
            
            # Calculate surface temperature
            surface_temp = equilibrium_temp * greenhouse_factor
            
            # Assess habitability
            habitability = self._assess_habitability(surface_temp, pressure, composition)
            
            # Predict climate zones
            climate_zones = self._predict_climate_zones(surface_temp, pressure)
            
            return {
                'equilibrium_temperature': equilibrium_temp,
                'surface_temperature': surface_temp,
                'greenhouse_factor': greenhouse_factor,
                'habitability_score': habitability,
                'climate_zones': climate_zones,
                'atmospheric_circulation': self._model_atmospheric_circulation(surface_temp, pressure)
            }
            
        except Exception as e:
            logger.error(f"Error modeling planetary climate: {e}")
            return {}
    
    def _calculate_equilibrium_temperature(self, stellar_temp: float, 
                                         stellar_luminosity: float, 
                                         orbital_distance: float) -> float:
        """Calculate planetary equilibrium temperature"""
        # Stefan-Boltzmann law
        sigma = 5.67e-8  # W/m²/K⁴
        solar_constant = 1361  # W/m² (Earth's value)
        
        # Adjust for stellar luminosity and orbital distance
        incident_flux = solar_constant * stellar_luminosity / (orbital_distance ** 2)
        
        # Calculate equilibrium temperature
        equilibrium_temp = (incident_flux / (4 * sigma)) ** 0.25
        
        return equilibrium_temp
    
    def _calculate_greenhouse_effect(self, composition: Dict, pressure: float) -> float:
        """Calculate greenhouse effect factor"""
        greenhouse_factor = 1.0
        
        # CO2 greenhouse effect
        co2_abundance = composition.get('CO2', 0) / 1e6  # Convert ppm to fraction
        greenhouse_factor += co2_abundance * 10  # Simplified relationship
        
        # CH4 greenhouse effect
        ch4_abundance = composition.get('CH4', 0) / 1e6
        greenhouse_factor += ch4_abundance * 20  # CH4 is more potent
        
        # H2O greenhouse effect
        h2o_abundance = composition.get('H2O', 0) / 1e6
        greenhouse_factor += h2o_abundance * 5
        
        return min(3.0, max(1.0, greenhouse_factor))  # Reasonable range
    
    def _assess_habitability(self, temperature: float, pressure: float, 
                            composition: Dict) -> float:
        """Assess planetary habitability"""
        habitability = 0.0
        
        # Temperature habitability
        if 273 < temperature < 373:  # Liquid water range
            temp_score = 1.0
        elif 250 < temperature < 400:
            temp_score = 0.5
        else:
            temp_score = 0.0
        
        # Pressure habitability
        if 10000 < pressure < 1000000:  # Pa
            pressure_score = 1.0
        elif 1000 < pressure < 10000000:
            pressure_score = 0.5
        else:
            pressure_score = 0.0
        
        # Atmospheric composition
        composition_score = 0.0
        if 'H2O' in composition and composition['H2O'] > 1000:  # Water present
            composition_score += 0.3
        if 'O2' in composition and composition['O2'] > 1000:  # Oxygen present
            composition_score += 0.3
        if 'N2' in composition and composition['N2'] > 100000:  # Nitrogen buffer
            composition_score += 0.4
        
        habitability = (temp_score + pressure_score + composition_score) / 3.0
        
        return min(1.0, max(0.0, habitability))
    
    def _predict_climate_zones(self, temperature: float, pressure: float) -> Dict:
        """Predict climate zones on the planet"""
        # Simplified climate zone prediction
        if temperature < 250:
            return {'primary_zone': 'polar', 'secondary_zones': ['ice_cap']}
        elif temperature < 300:
            return {'primary_zone': 'temperate', 'secondary_zones': ['polar', 'tropical']}
        else:
            return {'primary_zone': 'tropical', 'secondary_zones': ['temperate', 'desert']}
    
    def _model_atmospheric_circulation(self, temperature: float, pressure: float) -> Dict:
        """Model atmospheric circulation patterns"""
        # Simplified atmospheric circulation model
        if temperature > 350:
            return {'pattern': 'super_rotating', 'wind_speed': 'high'}
        elif temperature > 300:
            return {'pattern': 'hadley_cells', 'wind_speed': 'medium'}
        else:
            return {'pattern': 'polar_vortex', 'wind_speed': 'low'}

class BiosignatureDetector:
    """Main class for biosignature detection"""
    
    def __init__(self):
        self.spectral_analyzer = SpectralAnalyzer()
        self.climate_modeler = ClimateModeler()
    
    def detect_biosignatures(self, wavelengths: np.ndarray, 
                           transmission: np.ndarray,
                           error_bars: np.ndarray = None,
                           stellar_properties: Dict = None) -> Dict:
        """Comprehensive biosignature detection analysis"""
        try:
            # Analyze transmission spectrum
            spectral_analysis = self.spectral_analyzer.analyze_transmission_spectrum(
                wavelengths, transmission, error_bars
            )
            
            # Model planetary climate
            if stellar_properties:
                climate_model = self.climate_modeler.model_planetary_climate(
                    spectral_analysis.get('atmospheric_properties', {}),
                    stellar_properties
                )
            else:
                climate_model = {}
            
            # Combine results
            biosignature_results = {
                'spectral_analysis': spectral_analysis,
                'climate_model': climate_model,
                'overall_assessment': self._assess_overall_biosignature_potential(
                    spectral_analysis, climate_model
                )
            }
            
            return biosignature_results
            
        except Exception as e:
            logger.error(f"Error in biosignature detection: {e}")
            return {}
    
    def _assess_overall_biosignature_potential(self, spectral_analysis: Dict, 
                                             climate_model: Dict) -> Dict:
        """Assess overall potential for biosignatures"""
        try:
            biosignature_assessment = spectral_analysis.get('biosignature_assessment', {})
            biosignature_score = biosignature_assessment.get('biosignature_score', 0.0)
            life_probability = biosignature_assessment.get('life_probability', 0.0)
            
            habitability_score = climate_model.get('habitability_score', 0.0)
            
            # Combine scores
            overall_score = (biosignature_score + life_probability + habitability_score) / 3.0
            
            # Determine confidence level
            if overall_score > 0.8:
                confidence = "HIGH"
            elif overall_score > 0.6:
                confidence = "MEDIUM"
            elif overall_score > 0.4:
                confidence = "LOW"
            else:
                confidence = "VERY_LOW"
            
            return {
                'overall_score': overall_score,
                'confidence_level': confidence,
                'biosignature_indicators': biosignature_assessment.get('biosignature_indicators', []),
                'habitability_indicators': self._get_habitability_indicators(climate_model),
                'recommendations': self._generate_recommendations(overall_score, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error assessing overall biosignature potential: {e}")
            return {'overall_score': 0.0, 'confidence_level': 'VERY_LOW'}
    
    def _get_habitability_indicators(self, climate_model: Dict) -> List[str]:
        """Get habitability indicators from climate model"""
        indicators = []
        
        habitability_score = climate_model.get('habitability_score', 0.0)
        if habitability_score > 0.7:
            indicators.append("High habitability potential")
        elif habitability_score > 0.4:
            indicators.append("Moderate habitability potential")
        
        surface_temp = climate_model.get('surface_temperature', 0)
        if 273 < surface_temp < 373:
            indicators.append("Temperature suitable for liquid water")
        
        climate_zones = climate_model.get('climate_zones', {})
        if climate_zones.get('primary_zone') == 'temperate':
            indicators.append("Temperate climate zones present")
        
        return indicators
    
    def _generate_recommendations(self, overall_score: float, confidence: str) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if overall_score > 0.8:
            recommendations.extend([
                "High priority target for follow-up observations",
                "Consider spectroscopic confirmation with JWST",
                "Monitor for temporal variations in atmospheric composition",
                "Investigate potential surface biosignatures"
            ])
        elif overall_score > 0.6:
            recommendations.extend([
                "Moderate priority for additional observations",
                "Improve spectral resolution and signal-to-noise",
                "Consider multi-epoch observations",
                "Investigate stellar activity effects"
            ])
        else:
            recommendations.extend([
                "Low priority for biosignature follow-up",
                "Focus on basic atmospheric characterization",
                "Consider as comparison target",
                "Investigate potential false positives"
            ])
        
        return recommendations
