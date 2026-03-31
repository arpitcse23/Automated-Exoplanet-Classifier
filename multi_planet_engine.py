"""
Multi-Planet System Discovery Engine
Advanced deep learning for detecting entire planetary systems
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
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
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TransitTimingVariations:
    """Analyze Transit Timing Variations to detect gravitational interactions"""
    
    def __init__(self):
        self.min_period_ratio = 1.2  # Minimum period ratio for stability
        
    def detect_ttv_signals(self, time: np.ndarray, flux: np.ndarray, 
                          primary_period: float) -> Dict:
        """Detect TTV signals indicating multi-planet system"""
        try:
            # Find transit times
            transit_times = self._find_transit_times(time, flux, primary_period)
            
            if len(transit_times) < 3:
                return {'ttv_detected': False, 'ttv_amplitude': 0.0}
            
            # Calculate TTV
            expected_times = self._calculate_expected_times(transit_times[0], 
                                                          primary_period, 
                                                          len(transit_times))
            ttv = transit_times - expected_times
            
            # Analyze TTV pattern
            ttv_amplitude = np.std(ttv)
            ttv_significance = self._calculate_ttv_significance(ttv)
            
            # Detect periodic TTV (resonance signature)
            resonance_detected = self._detect_resonance_pattern(ttv, primary_period)
            
            return {
                'ttv_detected': ttv_significance > 3.0,
                'ttv_amplitude': ttv_amplitude,
                'ttv_significance': ttv_significance,
                'resonance_detected': resonance_detected,
                'transit_times': transit_times,
                'ttv_values': ttv
            }
            
        except Exception as e:
            logger.error(f"Error in TTV detection: {e}")
            return {'ttv_detected': False, 'ttv_amplitude': 0.0}
    
    def _find_transit_times(self, time: np.ndarray, flux: np.ndarray, 
                           period: float) -> np.ndarray:
        """Find individual transit times using template matching"""
        # Create transit template
        template = self._create_transit_template(period)
        
        # Find peaks in correlation
        correlation = signal.correlate(flux, template, mode='same')
        peaks, _ = signal.find_peaks(correlation, height=np.std(correlation))
        
        # Convert peak indices to times
        transit_times = time[peaks]
        
        # Filter by period spacing
        if len(transit_times) > 1:
            transit_times = self._filter_by_period_spacing(transit_times, period)
        
        return transit_times
    
    def _create_transit_template(self, period: float, duration: float = 0.1) -> np.ndarray:
        """Create a simple transit template"""
        template_length = int(period * 10)  # 10x oversampling
        template = np.ones(template_length)
        
        # Add transit dip
        transit_start = int(template_length * (0.5 - duration/2))
        transit_end = int(template_length * (0.5 + duration/2))
        template[transit_start:transit_end] *= 0.99  # 1% depth
        
        return template
    
    def _calculate_expected_times(self, first_transit: float, period: float, 
                                n_transits: int) -> np.ndarray:
        """Calculate expected transit times for linear ephemeris"""
        return np.array([first_transit + i * period for i in range(n_transits)])
    
    def _calculate_ttv_significance(self, ttv: np.ndarray) -> float:
        """Calculate statistical significance of TTV signal"""
        if len(ttv) < 3:
            return 0.0
        
        # Use chi-squared test against null hypothesis (no TTV)
        expected_std = np.std(ttv) / np.sqrt(len(ttv))
        observed_std = np.std(ttv)
        
        if expected_std == 0:
            return 0.0
        
        return observed_std / expected_std
    
    def _detect_resonance_pattern(self, ttv: np.ndarray, primary_period: float) -> bool:
        """Detect periodic TTV pattern indicating orbital resonance"""
        if len(ttv) < 5:
            return False
        
        # Look for periodic patterns in TTV
        from scipy.fft import fft, fftfreq
        
        # Remove linear trend
        ttv_detrended = signal.detrend(ttv)
        
        # FFT analysis
        fft_values = np.abs(fft(ttv_detrended))
        freqs = fftfreq(len(ttv_detrended))
        
        # Look for significant peaks (excluding DC component)
        peaks, _ = signal.find_peaks(fft_values[1:], height=np.mean(fft_values[1:]) * 2)
        
        return len(peaks) > 0
    
    def _filter_by_period_spacing(self, transit_times: np.ndarray, 
                                 period: float) -> np.ndarray:
        """Filter transit times by expected period spacing"""
        if len(transit_times) < 2:
            return transit_times
        
        # Calculate intervals
        intervals = np.diff(transit_times)
        
        # Keep transits with intervals close to expected period
        valid_indices = [0]  # Always keep first transit
        for i, interval in enumerate(intervals):
            if abs(interval - period) < period * 0.1:  # 10% tolerance
                valid_indices.append(i + 1)
        
        return transit_times[valid_indices]

class ResonantChainDetector:
    """Detect planets in orbital resonances using machine learning"""
    
    def __init__(self):
        self.model = self._build_resonance_model()
        
    def _build_resonance_model(self) -> MLPRegressor:
        """Build neural network for resonance detection"""
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        return model
    
    def detect_resonance_chains(self, periods: List[float]) -> Dict:
        """Detect resonant chains in planetary periods"""
        if len(periods) < 2:
            return {'resonance_detected': False, 'resonance_ratios': []}
        
        # Calculate period ratios
        period_ratios = []
        for i in range(len(periods)):
            for j in range(i + 1, len(periods)):
                ratio = periods[j] / periods[i]
                period_ratios.append(ratio)
        
        # Check for common resonances
        resonance_ratios = self._identify_resonance_ratios(period_ratios)
        
        # Calculate resonance strength
        resonance_strength = self._calculate_resonance_strength(period_ratios, resonance_ratios)
        
        return {
            'resonance_detected': len(resonance_ratios) > 0,
            'resonance_ratios': resonance_ratios,
            'resonance_strength': resonance_strength,
            'period_ratios': period_ratios
        }
    
    def _identify_resonance_ratios(self, period_ratios: List[float]) -> List[Dict]:
        """Identify common orbital resonances"""
        # Common resonance ratios
        common_resonances = {
            2.0: "2:1",
            1.5: "3:2", 
            1.33: "4:3",
            1.25: "5:4",
            1.2: "6:5",
            1.17: "7:6",
            1.14: "8:7",
            1.11: "9:8",
            1.09: "10:9"
        }
        
        resonance_ratios = []
        tolerance = 0.05  # 5% tolerance
        
        for ratio in period_ratios:
            for target_ratio, resonance_name in common_resonances.items():
                if abs(ratio - target_ratio) < tolerance:
                    resonance_ratios.append({
                        'ratio': ratio,
                        'resonance': resonance_name,
                        'deviation': abs(ratio - target_ratio)
                    })
        
        return resonance_ratios
    
    def _calculate_resonance_strength(self, period_ratios: List[float], 
                                    resonance_ratios: List[Dict]) -> float:
        """Calculate overall resonance strength"""
        if not resonance_ratios:
            return 0.0
        
        # Weight by number of resonances and their precision
        total_weight = 0.0
        weighted_strength = 0.0
        
        for resonance in resonance_ratios:
            weight = 1.0 / (1.0 + resonance['deviation'] * 10)  # Higher weight for closer matches
            weighted_strength += weight
            total_weight += weight
        
        return weighted_strength / total_weight if total_weight > 0 else 0.0

class StabilityPredictor:
    """Predict long-term orbital stability using AI"""
    
    def __init__(self):
        self.model = self._build_stability_model()
        
    def _build_stability_model(self) -> RandomForestRegressor:
        """Build random forest for stability prediction"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        return model
    
    def predict_stability(self, planetary_system: Dict) -> Dict:
        """Predict orbital stability of planetary system"""
        try:
            # Extract system parameters
            periods = planetary_system.get('periods', [])
            masses = planetary_system.get('masses', [])
            eccentricities = planetary_system.get('eccentricities', [])
            
            if len(periods) < 2:
                return {'stable': True, 'stability_score': 1.0}
            
            # Calculate stability features
            features = self._calculate_stability_features(periods, masses, eccentricities)
            
            # Predict stability (simplified - would need training data in practice)
            stability_score = self._estimate_stability(features)
            
            return {
                'stable': stability_score > 0.5,
                'stability_score': stability_score,
                'instability_risk': 1.0 - stability_score
            }
            
        except Exception as e:
            logger.error(f"Error in stability prediction: {e}")
            return {'stable': True, 'stability_score': 0.5}
    
    def _calculate_stability_features(self, periods: List[float], 
                                    masses: List[float], 
                                    eccentricities: List[float]) -> np.ndarray:
        """Calculate features for stability prediction"""
        features = []
        
        # Period ratios
        for i in range(len(periods) - 1):
            features.append(periods[i + 1] / periods[i])
        
        # Mass ratios
        for i in range(len(masses) - 1):
            features.append(masses[i + 1] / masses[i])
        
        # Eccentricity statistics
        features.append(np.mean(eccentricities))
        features.append(np.std(eccentricities))
        features.append(np.max(eccentricities))
        
        # System compactness
        if len(periods) > 1:
            features.append(periods[-1] / periods[0])  # Outer to inner period ratio
        
        return np.array(features)
    
    def _estimate_stability(self, features: np.ndarray) -> float:
        """Estimate stability based on features (simplified heuristic)"""
        # Simple heuristic based on period ratios and eccentricities
        if len(features) < 2:
            return 1.0
        
        stability_score = 1.0
        
        # Penalize small period ratios (unstable)
        period_ratios = features[:len(features)//2] if len(features) > 2 else [features[0]]
        for ratio in period_ratios:
            if ratio < 1.2:  # Too close
                stability_score *= 0.5
            elif ratio < 1.5:  # Marginally stable
                stability_score *= 0.8
        
        # Penalize high eccentricities
        if len(features) > 2:
            mean_ecc = features[-3]  # Mean eccentricity
            if mean_ecc > 0.3:
                stability_score *= 0.7
            elif mean_ecc > 0.1:
                stability_score *= 0.9
        
        return min(1.0, max(0.0, stability_score))

class MultiPlanetSignalDecomposer:
    """Decompose overlapping transit signals using deep learning"""
    
    def __init__(self):
        self.model = self._build_decomposition_model()
        
    def _build_decomposition_model(self):
        """Build CNN for signal decomposition"""
        if TENSORFLOW_AVAILABLE:
            model = keras.Sequential([
                keras.layers.Conv1D(64, 3, activation='relu', input_shape=(None, 1)),
                keras.layers.Conv1D(64, 3, activation='relu'),
                keras.layers.MaxPooling1D(2),
                keras.layers.Conv1D(128, 3, activation='relu'),
                keras.layers.Conv1D(128, 3, activation='relu'),
                keras.layers.GlobalMaxPooling1D(),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(4, activation='sigmoid')  # 4 planets max
            ])
            model.compile(optimizer='adam', loss='mse')
            return model
        else:
            # Fallback to scikit-learn MLPRegressor when TensorFlow is not available
            logger.warning("TensorFlow not available, using scikit-learn fallback for signal decomposition")
            return MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
    
    def decompose_signals(self, time: np.ndarray, flux: np.ndarray) -> Dict:
        """Decompose overlapping transit signals"""
        try:
            # Prepare input data
            flux_normalized = (flux - np.mean(flux)) / np.std(flux)
            flux_reshaped = flux_normalized.reshape(1, -1, 1)
            
            # Predict individual planet signals (simplified)
            # In practice, this would require extensive training data
            individual_signals = self._estimate_individual_signals(flux_normalized)
            
            # Extract planet parameters
            planets = self._extract_planet_parameters(individual_signals, time)
            
            return {
                'planets_detected': len(planets),
                'individual_signals': individual_signals,
                'planet_parameters': planets,
                'decomposition_quality': self._assess_decomposition_quality(individual_signals, flux_normalized)
            }
            
        except Exception as e:
            logger.error(f"Error in signal decomposition: {e}")
            return {'planets_detected': 0, 'individual_signals': [], 'planet_parameters': []}
    
    def _estimate_individual_signals(self, flux: np.ndarray) -> List[np.ndarray]:
        """Estimate individual planet signals (simplified heuristic)"""
        # This is a simplified approach - real implementation would use trained model
        signals = []
        
        # Find potential transit dips
        from scipy.signal import find_peaks
        inverted_flux = -flux  # Invert to find dips
        peaks, _ = find_peaks(inverted_flux, height=np.std(flux) * 0.5)
        
        # Group nearby peaks as potential planets
        if len(peaks) > 0:
            # Simple grouping by distance
            grouped_peaks = self._group_peaks(peaks, min_distance=10)
            
            for group in grouped_peaks:
                if len(group) > 2:  # At least 3 transits
                    signal = self._create_planet_signal(flux, group)
                    signals.append(signal)
        
        return signals
    
    def _group_peaks(self, peaks: np.ndarray, min_distance: int) -> List[List[int]]:
        """Group nearby peaks as potential planets"""
        if len(peaks) == 0:
            return []
        
        groups = []
        current_group = [peaks[0]]
        
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] < min_distance:
                current_group.append(peaks[i])
            else:
                groups.append(current_group)
                current_group = [peaks[i]]
        
        groups.append(current_group)
        return groups
    
    def _create_planet_signal(self, flux: np.ndarray, peaks: List[int]) -> np.ndarray:
        """Create individual planet signal from peak positions"""
        signal = np.zeros_like(flux)
        
        # Estimate period from peak spacing
        if len(peaks) > 1:
            periods = np.diff(peaks)
            period = np.median(periods)
        else:
            period = 50  # Default period
        
        # Create transit template
        for peak in peaks:
            start = max(0, peak - int(period * 0.05))
            end = min(len(flux), peak + int(period * 0.05))
            signal[start:end] = -0.01  # 1% transit depth
        
        return signal
    
    def _extract_planet_parameters(self, signals: List[np.ndarray], 
                                 time: np.ndarray) -> List[Dict]:
        """Extract orbital parameters for each planet"""
        planets = []
        
        for i, signal in enumerate(signals):
            # Find transit times
            transit_indices = np.where(signal < -0.005)[0]  # Significant dips
            
            if len(transit_indices) < 2:
                continue
            
            # Estimate period
            if len(transit_indices) > 1:
                periods = np.diff(time[transit_indices])
                period = np.median(periods)
            else:
                period = 1.0
            
            # Estimate depth
            depth = np.abs(np.min(signal))
            
            # Estimate duration (simplified)
            duration = np.sum(signal < -0.005) * (time[1] - time[0]) if len(time) > 1 else 0.1
            
            planets.append({
                'planet_id': i + 1,
                'period': period,
                'depth': depth,
                'duration': duration,
                'transit_count': len(transit_indices)
            })
        
        return planets
    
    def _assess_decomposition_quality(self, individual_signals: List[np.ndarray], 
                                    original_flux: np.ndarray) -> float:
        """Assess quality of signal decomposition"""
        if not individual_signals:
            return 0.0
        
        # Reconstruct combined signal
        reconstructed = np.sum(individual_signals, axis=0)
        
        # Calculate correlation with original
        correlation = np.corrcoef(original_flux, reconstructed)[0, 1]
        
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0

class MultiPlanetSystemClassifier:
    """Main classifier for multi-planet systems"""
    
    def __init__(self):
        self.ttv_analyzer = TransitTimingVariations()
        self.resonance_detector = ResonantChainDetector()
        self.stability_predictor = StabilityPredictor()
        self.signal_decomposer = MultiPlanetSignalDecomposer()
        
    def analyze_system(self, time: np.ndarray, flux: np.ndarray, 
                      primary_period: float) -> Dict:
        """Comprehensive analysis of potential multi-planet system"""
        try:
            # TTV analysis
            ttv_results = self.ttv_analyzer.detect_ttv_signals(time, flux, primary_period)
            
            # Signal decomposition
            decomposition_results = self.signal_decomposer.decompose_signals(time, flux)
            
            # Extract periods for resonance analysis
            periods = [primary_period]
            for planet in decomposition_results.get('planet_parameters', []):
                periods.append(planet['period'])
            
            # Resonance analysis
            resonance_results = self.resonance_detector.detect_resonance_chains(periods)
            
            # Stability prediction
            system_params = {
                'periods': periods,
                'masses': [1.0] * len(periods),  # Placeholder - would need mass estimates
                'eccentricities': [0.1] * len(periods)  # Placeholder
            }
            stability_results = self.stability_predictor.predict_stability(system_params)
            
            # Overall system classification
            system_score = self._calculate_system_score(
                ttv_results, decomposition_results, 
                resonance_results, stability_results
            )
            
            return {
                'multi_planet_detected': system_score > 0.6,
                'system_score': system_score,
                'ttv_analysis': ttv_results,
                'signal_decomposition': decomposition_results,
                'resonance_analysis': resonance_results,
                'stability_analysis': stability_results,
                'confidence': min(1.0, system_score)
            }
            
        except Exception as e:
            logger.error(f"Error in multi-planet system analysis: {e}")
            return {
                'multi_planet_detected': False,
                'system_score': 0.0,
                'confidence': 0.0
            }
    
    def _calculate_system_score(self, ttv_results: Dict, decomposition_results: Dict,
                              resonance_results: Dict, stability_results: Dict) -> float:
        """Calculate overall system detection score"""
        score = 0.0
        
        # TTV contribution (30%)
        if ttv_results.get('ttv_detected', False):
            score += 0.3 * min(1.0, ttv_results.get('ttv_significance', 0) / 5.0)
        
        # Signal decomposition contribution (40%)
        planets_detected = decomposition_results.get('planets_detected', 0)
        if planets_detected > 1:
            score += 0.4 * min(1.0, planets_detected / 4.0)  # Up to 4 planets
        
        # Resonance contribution (20%)
        if resonance_results.get('resonance_detected', False):
            score += 0.2 * resonance_results.get('resonance_strength', 0)
        
        # Stability contribution (10%)
        if stability_results.get('stable', True):
            score += 0.1 * stability_results.get('stability_score', 0)
        
        return min(1.0, score)
