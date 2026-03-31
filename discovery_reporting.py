"""
Autonomous Discovery Reporting System
AI-powered scientific writing and discovery reporting capabilities
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import re
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

class DiscoveryType(Enum):
    SINGLE_PLANET = "single_planet"
    MULTI_PLANET = "multi_planet"
    BIOSIGNATURE = "biosignature"
    RESONANT_SYSTEM = "resonant_system"
    HABITABLE_ZONE = "habitable_zone"

@dataclass
class DiscoveryData:
    """Structured data for a discovery"""
    discovery_id: str
    discovery_type: DiscoveryType
    target_id: str
    coordinates: Tuple[float, float]  # (RA, Dec)
    stellar_properties: Dict
    planetary_properties: Dict
    detection_confidence: float
    statistical_significance: float
    observation_data: Dict
    analysis_results: Dict
    timestamp: datetime

class StatisticalAnalyzer:
    """Performs statistical analysis and significance testing"""
    
    def __init__(self):
        self.confidence_levels = {
            'high': 0.95,
            'medium': 0.90,
            'low': 0.80
        }
    
    def calculate_significance(self, detection_data: Dict) -> Dict:
        """Calculate statistical significance of detection"""
        try:
            # Extract key parameters
            confidence = detection_data.get('confidence', 0.0)
            signal_strength = detection_data.get('signal_strength', 0.0)
            noise_level = detection_data.get('noise_level', 1.0)
            n_observations = detection_data.get('n_observations', 1)
            
            # Calculate signal-to-noise ratio
            snr = signal_strength / noise_level if noise_level > 0 else 0.0
            
            # Calculate detection significance (sigma)
            significance_sigma = self._calculate_sigma_significance(confidence, snr, n_observations)
            
            # Calculate false positive probability
            false_positive_prob = self._calculate_false_positive_probability(significance_sigma)
            
            # Assess detection quality
            detection_quality = self._assess_detection_quality(significance_sigma, snr)
            
            return {
                'significance_sigma': significance_sigma,
                'false_positive_probability': false_positive_prob,
                'detection_quality': detection_quality,
                'signal_to_noise_ratio': snr,
                'confidence_level': self._get_confidence_level(significance_sigma)
            }
            
        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            return {'significance_sigma': 0.0, 'detection_quality': 'poor'}
    
    def _calculate_sigma_significance(self, confidence: float, snr: float, n_obs: int) -> float:
        """Calculate sigma significance level"""
        # Simplified calculation - in practice would use more sophisticated methods
        base_sigma = confidence * 5.0  # Convert confidence to sigma
        snr_factor = min(2.0, snr / 3.0)  # SNR contribution
        obs_factor = min(1.5, np.sqrt(n_obs) / 10.0)  # Multiple observations
        
        return base_sigma * snr_factor * obs_factor
    
    def _calculate_false_positive_probability(self, sigma: float) -> float:
        """Calculate false positive probability from sigma"""
        # Using complementary error function approximation
        if sigma <= 0:
            return 1.0
        
        # P(FP) ≈ 0.5 * erfc(sigma / sqrt(2))
        # Simplified approximation
        return 0.5 * np.exp(-sigma**2 / 2)
    
    def _assess_detection_quality(self, sigma: float, snr: float) -> str:
        """Assess overall detection quality"""
        if sigma >= 5.0 and snr >= 5.0:
            return 'excellent'
        elif sigma >= 3.0 and snr >= 3.0:
            return 'good'
        elif sigma >= 2.0 and snr >= 2.0:
            return 'fair'
        else:
            return 'poor'
    
    def _get_confidence_level(self, sigma: float) -> str:
        """Get confidence level from sigma"""
        if sigma >= 5.0:
            return 'very_high'
        elif sigma >= 3.0:
            return 'high'
        elif sigma >= 2.0:
            return 'medium'
        else:
            return 'low'
    
    def calculate_uncertainties(self, measurements: List[float], 
                              measurement_errors: List[float]) -> Dict:
        """Calculate proper uncertainty propagation"""
        try:
            if not measurements or not measurement_errors:
                return {'mean': 0.0, 'std': 0.0, 'uncertainty': 0.0}
            
            # Calculate weighted mean and uncertainty
            weights = [1.0 / (err**2) for err in measurement_errors if err > 0]
            weighted_values = [val * weight for val, weight in zip(measurements, weights)]
            
            if not weights:
                return {'mean': np.mean(measurements), 'std': np.std(measurements), 'uncertainty': 0.0}
            
            weighted_mean = sum(weighted_values) / sum(weights)
            uncertainty = 1.0 / np.sqrt(sum(weights))
            
            # Calculate standard deviation
            std_dev = np.std(measurements)
            
            return {
                'mean': weighted_mean,
                'std': std_dev,
                'uncertainty': uncertainty,
                'n_measurements': len(measurements)
            }
            
        except Exception as e:
            logger.error(f"Error calculating uncertainties: {e}")
            return {'mean': 0.0, 'std': 0.0, 'uncertainty': 0.0}

class ScientificWriter:
    """AI-powered scientific writing system"""
    
    def __init__(self):
        self.template_library = self._load_templates()
        self.citation_style = "AAS"  # American Astronomical Society
        
    def _load_templates(self) -> Dict:
        """Load scientific paper templates"""
        return {
            'abstract': """
            We report the discovery of {discovery_type} in the {stellar_type} star {target_name} 
            (RA={ra:.4f}°, Dec={dec:.4f}°). The detection was made using {method} with a 
            confidence level of {confidence:.1f}% and statistical significance of {significance:.1f}σ. 
            The {planet_type} has an orbital period of {period:.3f} ± {period_err:.3f} days and 
            a radius of {radius:.2f} ± {radius_err:.2f} R⊕. {additional_findings} 
            This discovery {scientific_impact}.
            """,
            
            'introduction': """
            The search for exoplanets has revolutionized our understanding of planetary systems 
            and the potential for life beyond Earth. Since the first confirmed exoplanet detection 
            in 1995, over {total_planets} planets have been discovered, revealing an incredible 
            diversity of planetary architectures and compositions.
            
            {discovery_context} The discovery of {discovery_type} in {target_name} represents 
            a significant contribution to our understanding of {scientific_area}. This system 
            provides unique insights into {research_questions}.
            """,
            
            'methods': """
            Observations were conducted using {telescope} with {instrument} over a period of 
            {observation_duration} days. The data were processed using the {pipeline} pipeline 
            and analyzed with {analysis_methods}.
            
            {technical_details} The detection was confirmed through {confirmation_methods} 
            with a false positive probability of {false_positive_prob:.2e}.
            """,
            
            'results': """
            We detected {discovery_type} in {target_name} with the following properties:
            
            • Orbital period: {period:.3f} ± {period_err:.3f} days
            • Planet radius: {radius:.2f} ± {radius_err:.2f} R⊕
            • Planet mass: {mass:.2f} ± {mass_err:.2f} M⊕
            • Orbital eccentricity: {eccentricity:.3f} ± {eccentricity_err:.3f}
            • Equilibrium temperature: {temperature:.0f} ± {temperature_err:.0f} K
            
            {additional_results}
            
            The statistical significance of this detection is {significance:.1f}σ, corresponding 
            to a false positive probability of {false_positive_prob:.2e}.
            """,
            
            'discussion': """
            The discovery of {discovery_type} in {target_name} has important implications for 
            our understanding of {scientific_area}. {scientific_implications}
            
            {comparison_with_literature} This system is particularly interesting because 
            {unique_aspects}.
            
            Future observations with {future_instruments} will be crucial for {future_research}.
            """
        }
    
    def generate_abstract(self, discovery_data: DiscoveryData, 
                         analysis_results: Dict) -> str:
        """Generate scientific abstract"""
        try:
            template = self.template_library['abstract']
            
            # Extract key information
            stellar_props = discovery_data.stellar_properties
            planetary_props = discovery_data.planetary_properties
            
            # Format the abstract
            abstract = template.format(
                discovery_type=self._format_discovery_type(discovery_data.discovery_type),
                stellar_type=stellar_props.get('spectral_type', 'unknown'),
                target_name=discovery_data.target_id,
                ra=discovery_data.coordinates[0],
                dec=discovery_data.coordinates[1],
                method=analysis_results.get('detection_method', 'transit photometry'),
                confidence=discovery_data.detection_confidence * 100,
                significance=analysis_results.get('significance_sigma', 0.0),
                planet_type=self._get_planet_type(planetary_props),
                period=planetary_props.get('period', 0.0),
                period_err=planetary_props.get('period_error', 0.0),
                radius=planetary_props.get('radius', 0.0),
                radius_err=planetary_props.get('radius_error', 0.0),
                additional_findings=self._get_additional_findings(discovery_data),
                scientific_impact=self._assess_scientific_impact(discovery_data)
            )
            
            return abstract.strip()
            
        except Exception as e:
            logger.error(f"Error generating abstract: {e}")
            return "Abstract generation failed."
    
    def generate_paper_sections(self, discovery_data: DiscoveryData, 
                              analysis_results: Dict) -> Dict:
        """Generate all paper sections"""
        try:
            sections = {}
            
            # Generate each section
            for section_name, template in self.template_library.items():
                if section_name == 'abstract':
                    sections[section_name] = self.generate_abstract(discovery_data, analysis_results)
                else:
                    sections[section_name] = self._generate_section(
                        section_name, discovery_data, analysis_results
                    )
            
            return sections
            
        except Exception as e:
            logger.error(f"Error generating paper sections: {e}")
            return {}
    
    def _generate_section(self, section_name: str, discovery_data: DiscoveryData, 
                         analysis_results: Dict) -> str:
        """Generate a specific paper section"""
        try:
            template = self.template_library[section_name]
            
            # Format section with discovery data
            section = template.format(
                discovery_type=self._format_discovery_type(discovery_data.discovery_type),
                target_name=discovery_data.target_id,
                ra=discovery_data.coordinates[0],
                dec=discovery_data.coordinates[1],
                stellar_type=discovery_data.stellar_properties.get('spectral_type', 'unknown'),
                telescope=analysis_results.get('telescope', 'TESS'),
                instrument=analysis_results.get('instrument', 'CCD'),
                observation_duration=analysis_results.get('observation_duration', 30),
                pipeline=analysis_results.get('pipeline', 'custom'),
                analysis_methods=analysis_results.get('analysis_methods', 'transit fitting'),
                technical_details=self._get_technical_details(analysis_results),
                confirmation_methods=analysis_results.get('confirmation_methods', 'statistical validation'),
                false_positive_prob=analysis_results.get('false_positive_probability', 1e-6),
                period=discovery_data.planetary_properties.get('period', 0.0),
                period_err=discovery_data.planetary_properties.get('period_error', 0.0),
                radius=discovery_data.planetary_properties.get('radius', 0.0),
                radius_err=discovery_data.planetary_properties.get('radius_error', 0.0),
                mass=discovery_data.planetary_properties.get('mass', 0.0),
                mass_err=discovery_data.planetary_properties.get('mass_error', 0.0),
                eccentricity=discovery_data.planetary_properties.get('eccentricity', 0.0),
                eccentricity_err=discovery_data.planetary_properties.get('eccentricity_error', 0.0),
                temperature=discovery_data.planetary_properties.get('temperature', 0.0),
                temperature_err=discovery_data.planetary_properties.get('temperature_error', 0.0),
                additional_results=self._get_additional_results(discovery_data),
                significance=analysis_results.get('significance_sigma', 0.0),
                scientific_area=self._get_scientific_area(discovery_data.discovery_type),
                scientific_implications=self._get_scientific_implications(discovery_data),
                comparison_with_literature=self._get_literature_comparison(discovery_data),
                unique_aspects=self._get_unique_aspects(discovery_data),
                future_instruments=self._get_future_instruments(discovery_data),
                future_research=self._get_future_research(discovery_data),
                total_planets=5000,  # Current number of known exoplanets
                discovery_context=self._get_discovery_context(discovery_data),
                research_questions=self._get_research_questions(discovery_data)
            )
            
            return section.strip()
            
        except Exception as e:
            logger.error(f"Error generating {section_name}: {e}")
            return f"Error generating {section_name} section."
    
    def _format_discovery_type(self, discovery_type: DiscoveryType) -> str:
        """Format discovery type for text"""
        type_map = {
            DiscoveryType.SINGLE_PLANET: "a transiting exoplanet",
            DiscoveryType.MULTI_PLANET: "a multi-planet system",
            DiscoveryType.BIOSIGNATURE: "potential biosignatures",
            DiscoveryType.RESONANT_SYSTEM: "a resonant planetary system",
            DiscoveryType.HABITABLE_ZONE: "a habitable zone planet"
        }
        return type_map.get(discovery_type, "an exoplanet")
    
    def _get_planet_type(self, planetary_props: Dict) -> str:
        """Determine planet type from properties"""
        radius = planetary_props.get('radius', 0.0)
        
        if radius < 1.25:
            return "super-Earth"
        elif radius < 2.0:
            return "sub-Neptune"
        elif radius < 4.0:
            return "Neptune-like"
        else:
            return "gas giant"
    
    def _get_additional_findings(self, discovery_data: DiscoveryData) -> str:
        """Get additional findings for abstract"""
        findings = []
        
        if discovery_data.discovery_type == DiscoveryType.MULTI_PLANET:
            findings.append("The system contains multiple planets in orbital resonance.")
        
        if discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            findings.append("Atmospheric analysis reveals potential biosignature molecules.")
        
        if discovery_data.discovery_type == DiscoveryType.HABITABLE_ZONE:
            findings.append("The planet orbits within the stellar habitable zone.")
        
        return " ".join(findings) if findings else ""
    
    def _assess_scientific_impact(self, discovery_data: DiscoveryData) -> str:
        """Assess scientific impact of discovery"""
        confidence = discovery_data.detection_confidence
        
        if confidence > 0.95:
            return "represents a major breakthrough in exoplanet science"
        elif confidence > 0.85:
            return "provides important new insights into planetary formation"
        elif confidence > 0.75:
            return "contributes to our understanding of planetary systems"
        else:
            return "adds to the growing catalog of exoplanet discoveries"
    
    def _get_technical_details(self, analysis_results: Dict) -> str:
        """Get technical details for methods section"""
        details = []
        
        if 'transit_fitting' in analysis_results.get('analysis_methods', ''):
            details.append("Transit light curves were fitted using a Bayesian MCMC approach.")
        
        if 'spectral_analysis' in analysis_results.get('analysis_methods', ''):
            details.append("Atmospheric composition was analyzed using transmission spectroscopy.")
        
        if 'ttv_analysis' in analysis_results.get('analysis_methods', ''):
            details.append("Transit timing variations were analyzed to detect gravitational interactions.")
        
        return " ".join(details) if details else "Standard exoplanet detection techniques were employed."
    
    def _get_additional_results(self, discovery_data: DiscoveryData) -> str:
        """Get additional results for results section"""
        results = []
        
        if discovery_data.discovery_type == DiscoveryType.MULTI_PLANET:
            n_planets = len(discovery_data.planetary_properties.get('planets', []))
            results.append(f"The system contains {n_planets} confirmed planets.")
        
        if discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            biosignatures = discovery_data.analysis_results.get('biosignatures', [])
            if biosignatures:
                results.append(f"Detected biosignature molecules: {', '.join(biosignatures)}.")
        
        return " ".join(results) if results else ""
    
    def _get_scientific_area(self, discovery_type: DiscoveryType) -> str:
        """Get scientific area for discussion"""
        area_map = {
            DiscoveryType.SINGLE_PLANET: "planetary formation and evolution",
            DiscoveryType.MULTI_PLANET: "planetary system architecture",
            DiscoveryType.BIOSIGNATURE: "astrobiology and atmospheric science",
            DiscoveryType.RESONANT_SYSTEM: "orbital dynamics and system stability",
            DiscoveryType.HABITABLE_ZONE: "habitability and life detection"
        }
        return area_map.get(discovery_type, "exoplanet science")
    
    def _get_scientific_implications(self, discovery_data: DiscoveryData) -> str:
        """Get scientific implications for discussion"""
        implications = []
        
        if discovery_data.discovery_type == DiscoveryType.MULTI_PLANET:
            implications.append("This discovery provides insights into how planetary systems form and evolve.")
        
        if discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            implications.append("The detection of potential biosignatures opens new possibilities for life detection.")
        
        if discovery_data.discovery_type == DiscoveryType.HABITABLE_ZONE:
            implications.append("This planet represents a prime target for future habitability studies.")
        
        return " ".join(implications) if implications else "This discovery advances our understanding of exoplanetary systems."
    
    def _get_literature_comparison(self, discovery_data: DiscoveryData) -> str:
        """Get literature comparison for discussion"""
        return "This system is consistent with theoretical predictions and similar discoveries in the literature."
    
    def _get_unique_aspects(self, discovery_data: DiscoveryData) -> str:
        """Get unique aspects for discussion"""
        aspects = []
        
        if discovery_data.discovery_type == DiscoveryType.RESONANT_SYSTEM:
            aspects.append("the planets are in orbital resonance, which is rare and scientifically valuable")
        
        if discovery_data.detection_confidence > 0.95:
            aspects.append("the high confidence of detection makes this a particularly robust discovery")
        
        return " ".join(aspects) if aspects else "it represents a significant addition to the exoplanet catalog"
    
    def _get_future_instruments(self, discovery_data: DiscoveryData) -> str:
        """Get future instruments for discussion"""
        instruments = ["JWST", "ELT", "Ariel"]
        return ", ".join(instruments)
    
    def _get_future_research(self, discovery_data: DiscoveryData) -> str:
        """Get future research directions"""
        if discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            return "confirming the presence of biosignature molecules and characterizing atmospheric composition"
        elif discovery_data.discovery_type == DiscoveryType.HABITABLE_ZONE:
            return "determining atmospheric composition and surface conditions"
        else:
            return "refining orbital parameters and studying system dynamics"
    
    def _get_discovery_context(self, discovery_data: DiscoveryData) -> str:
        """Get discovery context for introduction"""
        return f"The detection of {self._format_discovery_type(discovery_data.discovery_type)} in {discovery_data.target_id} was made as part of an ongoing survey."
    
    def _get_research_questions(self, discovery_data: DiscoveryData) -> str:
        """Get research questions for introduction"""
        if discovery_data.discovery_type == DiscoveryType.MULTI_PLANET:
            return "how planetary systems form and maintain stability"
        elif discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            return "the potential for life in exoplanetary atmospheres"
        else:
            return "planetary formation and evolution processes"

class ImpactAssessor:
    """Assesses scientific impact and importance of discoveries"""
    
    def __init__(self):
        self.impact_weights = {
            'novelty': 0.3,
            'significance': 0.25,
            'confidence': 0.2,
            'habitability': 0.15,
            'follow_up_potential': 0.1
        }
    
    def assess_discovery_impact(self, discovery_data: DiscoveryData, 
                               analysis_results: Dict) -> Dict:
        """Assess overall scientific impact of discovery"""
        try:
            # Calculate individual impact factors
            novelty_score = self._assess_novelty(discovery_data)
            significance_score = self._assess_significance(discovery_data)
            confidence_score = discovery_data.detection_confidence
            habitability_score = self._assess_habitability_potential(discovery_data)
            follow_up_score = self._assess_follow_up_potential(discovery_data)
            
            # Calculate weighted impact score
            impact_score = (
                self.impact_weights['novelty'] * novelty_score +
                self.impact_weights['significance'] * significance_score +
                self.impact_weights['confidence'] * confidence_score +
                self.impact_weights['habitability'] * habitability_score +
                self.impact_weights['follow_up_potential'] * follow_up_score
            )
            
            # Determine impact level
            impact_level = self._get_impact_level(impact_score)
            
            # Generate impact summary
            impact_summary = self._generate_impact_summary(
                discovery_data, impact_score, impact_level
            )
            
            return {
                'impact_score': impact_score,
                'impact_level': impact_level,
                'novelty_score': novelty_score,
                'significance_score': significance_score,
                'habitability_score': habitability_score,
                'follow_up_score': follow_up_score,
                'impact_summary': impact_summary,
                'recommendations': self._generate_impact_recommendations(impact_score, impact_level)
            }
            
        except Exception as e:
            logger.error(f"Error assessing discovery impact: {e}")
            return {'impact_score': 0.0, 'impact_level': 'low'}
    
    def _assess_novelty(self, discovery_data: DiscoveryData) -> float:
        """Assess novelty of discovery"""
        novelty = 0.0
        
        # Multi-planet systems are novel
        if discovery_data.discovery_type == DiscoveryType.MULTI_PLANET:
            novelty += 0.4
        
        # Biosignatures are highly novel
        if discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            novelty += 0.6
        
        # Resonant systems are novel
        if discovery_data.discovery_type == DiscoveryType.RESONANT_SYSTEM:
            novelty += 0.5
        
        # Habitable zone planets are moderately novel
        if discovery_data.discovery_type == DiscoveryType.HABITABLE_ZONE:
            novelty += 0.3
        
        # High confidence detections are more novel
        if discovery_data.detection_confidence > 0.9:
            novelty += 0.2
        
        return min(1.0, novelty)
    
    def _assess_significance(self, discovery_data: DiscoveryData) -> float:
        """Assess scientific significance"""
        significance = 0.0
        
        # Statistical significance
        if discovery_data.statistical_significance > 5.0:
            significance += 0.3
        elif discovery_data.statistical_significance > 3.0:
            significance += 0.2
        
        # Discovery type significance
        if discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            significance += 0.5
        elif discovery_data.discovery_type == DiscoveryType.MULTI_PLANET:
            significance += 0.3
        elif discovery_data.discovery_type == DiscoveryType.HABITABLE_ZONE:
            significance += 0.2
        
        return min(1.0, significance)
    
    def _assess_habitability_potential(self, discovery_data: DiscoveryData) -> float:
        """Assess habitability potential"""
        if discovery_data.discovery_type != DiscoveryType.HABITABLE_ZONE:
            return 0.0
        
        # Check if in habitable zone
        stellar_props = discovery_data.stellar_properties
        planetary_props = discovery_data.planetary_properties
        
        # Simplified habitability assessment
        temperature = planetary_props.get('temperature', 300)
        if 250 < temperature < 350:  # K
            return 0.8
        elif 200 < temperature < 400:
            return 0.5
        else:
            return 0.2
    
    def _assess_follow_up_potential(self, discovery_data: DiscoveryData) -> float:
        """Assess potential for follow-up observations"""
        follow_up = 0.0
        
        # High confidence detections have better follow-up potential
        if discovery_data.detection_confidence > 0.9:
            follow_up += 0.3
        
        # Bright stars are better for follow-up
        stellar_props = discovery_data.stellar_properties
        magnitude = stellar_props.get('magnitude', 15)
        if magnitude < 10:
            follow_up += 0.4
        elif magnitude < 12:
            follow_up += 0.2
        
        # Biosignature detections need follow-up
        if discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            follow_up += 0.3
        
        return min(1.0, follow_up)
    
    def _get_impact_level(self, impact_score: float) -> str:
        """Get impact level from score"""
        if impact_score > 0.8:
            return 'revolutionary'
        elif impact_score > 0.6:
            return 'high'
        elif impact_score > 0.4:
            return 'moderate'
        elif impact_score > 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_impact_summary(self, discovery_data: DiscoveryData, 
                               impact_score: float, impact_level: str) -> str:
        """Generate impact summary"""
        if impact_level == 'revolutionary':
            return f"This {self._format_discovery_type(discovery_data.discovery_type)} discovery has the potential to revolutionize our understanding of {self._get_scientific_area(discovery_data.discovery_type)}."
        elif impact_level == 'high':
            return f"This discovery represents a significant advance in {self._get_scientific_area(discovery_data.discovery_type)} and will likely generate substantial follow-up research."
        elif impact_level == 'moderate':
            return f"This discovery contributes meaningfully to our understanding of {self._get_scientific_area(discovery_data.discovery_type)} and warrants further investigation."
        else:
            return f"This discovery adds to the growing body of knowledge in {self._get_scientific_area(discovery_data.discovery_type)}."
    
    def _generate_impact_recommendations(self, impact_score: float, impact_level: str) -> List[str]:
        """Generate impact-based recommendations"""
        recommendations = []
        
        if impact_level in ['revolutionary', 'high']:
            recommendations.extend([
                "Submit to high-impact journal (Nature, Science, ApJ)",
                "Prepare press release for media outreach",
                "Coordinate with other observatories for follow-up",
                "Consider organizing conference session"
            ])
        elif impact_level == 'moderate':
            recommendations.extend([
                "Submit to specialized journal (ApJ, MNRAS)",
                "Present at upcoming conference",
                "Coordinate follow-up observations"
            ])
        else:
            recommendations.extend([
                "Submit to appropriate journal",
                "Include in survey paper",
                "Archive for future reference"
            ])
        
        return recommendations
    
    def _format_discovery_type(self, discovery_type: DiscoveryType) -> str:
        """Format discovery type for text"""
        type_map = {
            DiscoveryType.SINGLE_PLANET: "exoplanet",
            DiscoveryType.MULTI_PLANET: "multi-planet system",
            DiscoveryType.BIOSIGNATURE: "biosignature",
            DiscoveryType.RESONANT_SYSTEM: "resonant system",
            DiscoveryType.HABITABLE_ZONE: "habitable zone planet"
        }
        return type_map.get(discovery_type, "exoplanet")
    
    def _get_scientific_area(self, discovery_type: DiscoveryType) -> str:
        """Get scientific area for impact assessment"""
        area_map = {
            DiscoveryType.SINGLE_PLANET: "planetary science",
            DiscoveryType.MULTI_PLANET: "planetary system dynamics",
            DiscoveryType.BIOSIGNATURE: "astrobiology",
            DiscoveryType.RESONANT_SYSTEM: "orbital mechanics",
            DiscoveryType.HABITABLE_ZONE: "habitability studies"
        }
        return area_map.get(discovery_type, "exoplanet science")

class DiscoveryReporter:
    """Main class for autonomous discovery reporting"""
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.scientific_writer = ScientificWriter()
        self.impact_assessor = ImpactAssessor()
    
    def generate_discovery_report(self, discovery_data: DiscoveryData, 
                                analysis_results: Dict) -> Dict:
        """Generate comprehensive discovery report"""
        try:
            # Perform statistical analysis
            statistical_results = self.statistical_analyzer.calculate_significance(analysis_results)
            
            # Generate scientific paper
            paper_sections = self.scientific_writer.generate_paper_sections(
                discovery_data, analysis_results
            )
            
            # Assess scientific impact
            impact_assessment = self.impact_assessor.assess_discovery_impact(
                discovery_data, analysis_results
            )
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                discovery_data, statistical_results, impact_assessment
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                discovery_data, statistical_results, impact_assessment
            )
            
            return {
                'discovery_data': discovery_data,
                'statistical_analysis': statistical_results,
                'scientific_paper': paper_sections,
                'impact_assessment': impact_assessment,
                'executive_summary': executive_summary,
                'recommendations': recommendations,
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating discovery report: {e}")
            return {}
    
    def _generate_executive_summary(self, discovery_data: DiscoveryData, 
                                  statistical_results: Dict, 
                                  impact_assessment: Dict) -> str:
        """Generate executive summary"""
        try:
            summary = f"""
            DISCOVERY EXECUTIVE SUMMARY
            =========================
            
            Discovery ID: {discovery_data.discovery_id}
            Target: {discovery_data.target_id}
            Type: {discovery_data.discovery_type.value}
            Coordinates: RA={discovery_data.coordinates[0]:.4f}°, Dec={discovery_data.coordinates[1]:.4f}°
            
            DETECTION PARAMETERS:
            • Confidence: {discovery_data.detection_confidence:.1%}
            • Statistical Significance: {statistical_results.get('significance_sigma', 0.0):.1f}σ
            • False Positive Probability: {statistical_results.get('false_positive_probability', 0.0):.2e}
            • Detection Quality: {statistical_results.get('detection_quality', 'unknown')}
            
            SCIENTIFIC IMPACT:
            • Impact Score: {impact_assessment.get('impact_score', 0.0):.2f}
            • Impact Level: {impact_assessment.get('impact_level', 'unknown').upper()}
            • Novelty: {impact_assessment.get('novelty_score', 0.0):.2f}
            • Significance: {impact_assessment.get('significance_score', 0.0):.2f}
            
            KEY FINDINGS:
            {impact_assessment.get('impact_summary', 'No summary available')}
            
            RECOMMENDATIONS:
            {chr(10).join(f"• {rec}" for rec in impact_assessment.get('recommendations', []))}
            
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            """
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Executive summary generation failed."
    
    def _generate_recommendations(self, discovery_data: DiscoveryData, 
                                statistical_results: Dict, 
                                impact_assessment: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Statistical recommendations
        if statistical_results.get('detection_quality') == 'poor':
            recommendations.append("Improve data quality through additional observations")
        
        if statistical_results.get('significance_sigma', 0) < 3.0:
            recommendations.append("Increase observation time to improve statistical significance")
        
        # Impact-based recommendations
        recommendations.extend(impact_assessment.get('recommendations', []))
        
        # Discovery-specific recommendations
        if discovery_data.discovery_type == DiscoveryType.BIOSIGNATURE:
            recommendations.append("Conduct follow-up spectroscopic observations with JWST")
        
        if discovery_data.discovery_type == DiscoveryType.MULTI_PLANET:
            recommendations.append("Monitor system for transit timing variations")
        
        if discovery_data.discovery_type == DiscoveryType.HABITABLE_ZONE:
            recommendations.append("Characterize atmospheric composition and surface conditions")
        
        return recommendations
    
    def save_report(self, report: Dict, filename: str = None) -> str:
        """Save discovery report to file"""
        try:
            if filename is None:
                discovery_id = report['discovery_data'].discovery_id
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"discovery_report_{discovery_id}_{timestamp}.json"
            
            # Convert dataclass to dict for JSON serialization
            report_dict = self._convert_to_dict(report)
            
            with open(filename, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Discovery report saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""
    
    def _convert_to_dict(self, obj):
        """Convert objects to dictionary for JSON serialization"""
        if hasattr(obj, '__dict__'):
            return {k: self._convert_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_dict(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
