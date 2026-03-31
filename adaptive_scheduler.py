"""
Real-Time Adaptive Observation Scheduler
AI system for optimizing telescope time and coordinating observations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import heapq
import threading
import time

logger = logging.getLogger(__name__)

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class ObservationTarget:
    """Represents an observation target with all relevant parameters"""
    target_id: str
    ra: float
    dec: float
    priority: Priority
    confidence_score: float
    observation_duration: float  # hours
    optimal_time: datetime
    telescope_requirements: List[str]
    weather_dependency: float  # 0-1, higher means more weather sensitive
    scientific_value: float  # 0-1, estimated scientific importance
    follow_up_urgency: float  # 0-1, how urgent is follow-up
    coordinates: Tuple[float, float] = None
    
    def __post_init__(self):
        if self.coordinates is None:
            self.coordinates = (self.ra, self.dec)

@dataclass
class Telescope:
    """Represents a telescope with its capabilities and constraints"""
    telescope_id: str
    location: Tuple[float, float]  # (lat, lon)
    aperture: float  # meters
    capabilities: List[str]  # e.g., ['photometry', 'spectroscopy', 'imaging']
    current_weather: Dict
    operational_hours: Tuple[int, int]  # (start_hour, end_hour) UTC
    maintenance_schedule: List[Tuple[datetime, datetime]]
    efficiency: float  # 0-1, current operational efficiency

class WeatherPredictor:
    """Predict weather conditions for telescope scheduling"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.weather_cache = {}
        
    def get_weather_forecast(self, location: Tuple[float, float], 
                           hours_ahead: int = 24) -> Dict:
        """Get weather forecast for telescope location"""
        try:
            lat, lon = location
            cache_key = f"{lat}_{lon}_{hours_ahead}"
            
            if cache_key in self.weather_cache:
                return self.weather_cache[cache_key]
            
            # Use OpenWeatherMap API (simplified)
            if self.api_key:
                url = f"http://api.openweathermap.org/data/2.5/forecast"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                forecast = self._parse_weather_data(data, hours_ahead)
            else:
                # Fallback to simulated weather
                forecast = self._simulate_weather(location, hours_ahead)
            
            self.weather_cache[cache_key] = forecast
            return forecast
            
        except Exception as e:
            logger.error(f"Error getting weather forecast: {e}")
            return self._simulate_weather(location, hours_ahead)
    
    def _parse_weather_data(self, data: Dict, hours_ahead: int) -> Dict:
        """Parse OpenWeatherMap API response"""
        forecasts = []
        
        for item in data.get('list', [])[:hours_ahead//3]:  # 3-hour intervals
            forecast = {
                'timestamp': datetime.fromtimestamp(item['dt']),
                'cloud_cover': item['clouds']['all'] / 100.0,
                'visibility': item['visibility'] / 1000.0,  # km
                'wind_speed': item['wind']['speed'],
                'humidity': item['main']['humidity'] / 100.0,
                'temperature': item['main']['temp'],
                'seeing_quality': self._calculate_seeing_quality(item)
            }
            forecasts.append(forecast)
        
        return {
            'forecasts': forecasts,
            'overall_quality': np.mean([f['seeing_quality'] for f in forecasts])
        }
    
    def _calculate_seeing_quality(self, weather_data: Dict) -> float:
        """Calculate astronomical seeing quality from weather data"""
        cloud_cover = weather_data['clouds']['all'] / 100.0
        wind_speed = weather_data['wind']['speed']
        humidity = weather_data['main']['humidity'] / 100.0
        
        # Simple seeing quality calculation
        seeing = 1.0
        seeing *= (1.0 - cloud_cover)  # Less clouds = better seeing
        seeing *= max(0.1, 1.0 - wind_speed / 20.0)  # Less wind = better seeing
        seeing *= max(0.1, 1.0 - humidity)  # Less humidity = better seeing
        
        return min(1.0, max(0.0, seeing))
    
    def _simulate_weather(self, location: Tuple[float, float], 
                         hours_ahead: int) -> Dict:
        """Simulate weather data for testing"""
        forecasts = []
        base_time = datetime.now()
        
        for i in range(0, hours_ahead, 3):
            forecast = {
                'timestamp': base_time + timedelta(hours=i),
                'cloud_cover': np.random.uniform(0.1, 0.8),
                'visibility': np.random.uniform(5, 20),
                'wind_speed': np.random.uniform(2, 15),
                'humidity': np.random.uniform(0.3, 0.9),
                'temperature': np.random.uniform(10, 25),
                'seeing_quality': np.random.uniform(0.3, 0.9)
            }
            forecasts.append(forecast)
        
        return {
            'forecasts': forecasts,
            'overall_quality': np.mean([f['seeing_quality'] for f in forecasts])
        }

class TargetPrioritizer:
    """AI system for prioritizing observation targets"""
    
    def __init__(self):
        self.priority_weights = {
            'confidence': 0.3,
            'scientific_value': 0.25,
            'follow_up_urgency': 0.2,
            'weather_dependency': 0.15,
            'telescope_efficiency': 0.1
        }
    
    def calculate_priority_score(self, target: ObservationTarget, 
                               telescope: Telescope, 
                               weather_forecast: Dict) -> float:
        """Calculate priority score for observation target"""
        try:
            # Base confidence score
            confidence_score = target.confidence_score
            
            # Scientific value
            scientific_score = target.scientific_value
            
            # Follow-up urgency (time decay)
            time_since_detection = datetime.now() - target.optimal_time
            urgency_decay = max(0.1, 1.0 - time_since_detection.total_seconds() / (24 * 3600))
            urgency_score = target.follow_up_urgency * urgency_decay
            
            # Weather compatibility
            weather_score = self._calculate_weather_compatibility(target, weather_forecast)
            
            # Telescope efficiency
            telescope_score = telescope.efficiency
            
            # Calculate weighted score
            total_score = (
                self.priority_weights['confidence'] * confidence_score +
                self.priority_weights['scientific_value'] * scientific_score +
                self.priority_weights['follow_up_urgency'] * urgency_score +
                self.priority_weights['weather_dependency'] * weather_score +
                self.priority_weights['telescope_efficiency'] * telescope_score
            )
            
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            logger.error(f"Error calculating priority score: {e}")
            return 0.0
    
    def _calculate_weather_compatibility(self, target: ObservationTarget, 
                                       weather_forecast: Dict) -> float:
        """Calculate how well weather conditions suit the target"""
        if not weather_forecast.get('forecasts'):
            return 0.5  # Neutral if no weather data
        
        # Get weather for optimal observation time
        optimal_time = target.optimal_time
        best_weather = None
        min_time_diff = float('inf')
        
        for forecast in weather_forecast['forecasts']:
            time_diff = abs((forecast['timestamp'] - optimal_time).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_weather = forecast
        
        if best_weather is None:
            return 0.5
        
        # Calculate compatibility based on target requirements
        compatibility = best_weather['seeing_quality']
        
        # Adjust for weather dependency
        if target.weather_dependency > 0.5:  # Weather sensitive
            compatibility *= (1.0 - best_weather['cloud_cover'])
        
        return min(1.0, max(0.0, compatibility))

class ObservationScheduler:
    """Main scheduler for coordinating observations across telescopes"""
    
    def __init__(self):
        self.telescopes = {}
        self.targets = []
        self.schedule = {}
        self.weather_predictor = WeatherPredictor()
        self.prioritizer = TargetPrioritizer()
        self.running = False
        self.schedule_lock = threading.Lock()
        
    def add_telescope(self, telescope: Telescope):
        """Add telescope to the network"""
        self.telescopes[telescope.telescope_id] = telescope
        self.schedule[telescope.telescope_id] = []
        logger.info(f"Added telescope {telescope.telescope_id}")
    
    def add_target(self, target: ObservationTarget):
        """Add observation target to queue"""
        self.targets.append(target)
        logger.info(f"Added target {target.target_id}")
    
    def start_scheduling(self, update_interval: int = 300):  # 5 minutes
        """Start continuous scheduling process"""
        self.running = True
        scheduler_thread = threading.Thread(
            target=self._scheduling_loop,
            args=(update_interval,),
            daemon=True
        )
        scheduler_thread.start()
        logger.info("Observation scheduling started")
    
    def stop_scheduling(self):
        """Stop scheduling process"""
        self.running = False
        logger.info("Observation scheduling stopped")
    
    def _scheduling_loop(self, update_interval: int):
        """Main scheduling loop"""
        while self.running:
            try:
                self._update_schedule()
                time.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                time.sleep(60)
    
    def _update_schedule(self):
        """Update observation schedule"""
        with self.schedule_lock:
            # Get current weather for all telescopes
            weather_forecasts = {}
            for telescope_id, telescope in self.telescopes.items():
                weather_forecasts[telescope_id] = self.weather_predictor.get_weather_forecast(
                    telescope.location, 24
                )
            
            # Calculate priority scores for all targets
            target_scores = []
            for target in self.targets:
                for telescope_id, telescope in self.telescopes.items():
                    score = self.prioritizer.calculate_priority_score(
                        target, telescope, weather_forecasts[telescope_id]
                    )
                    target_scores.append((score, target, telescope_id))
            
            # Sort by priority score
            target_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Schedule observations
            self._assign_observations(target_scores, weather_forecasts)
    
    def _assign_observations(self, target_scores: List[Tuple], 
                           weather_forecasts: Dict):
        """Assign observations to telescopes using optimization"""
        # Clear current schedule
        for telescope_id in self.schedule:
            self.schedule[telescope_id] = []
        
        # Track scheduled times
        scheduled_times = {}
        
        for score, target, telescope_id in target_scores:
            if self._can_schedule(target, telescope_id, scheduled_times, weather_forecasts):
                self._schedule_observation(target, telescope_id, scheduled_times)
    
    def _can_schedule(self, target: ObservationTarget, telescope_id: str,
                     scheduled_times: Dict, weather_forecasts: Dict) -> bool:
        """Check if target can be scheduled on telescope"""
        telescope = self.telescopes[telescope_id]
        
        # Check telescope capabilities
        if not any(req in telescope.capabilities for req in target.telescope_requirements):
            return False
        
        # Check operational hours
        current_hour = datetime.now().hour
        if not (telescope.operational_hours[0] <= current_hour <= telescope.operational_hours[1]):
            return False
        
        # Check maintenance schedule
        for maintenance_start, maintenance_end in telescope.maintenance_schedule:
            if maintenance_start <= target.optimal_time <= maintenance_end:
                return False
        
        # Check weather conditions
        weather = weather_forecasts.get(telescope_id, {})
        if weather.get('overall_quality', 0.5) < 0.3:  # Poor weather
            return False
        
        # Check for conflicts
        telescope_schedule = scheduled_times.get(telescope_id, [])
        for scheduled_start, scheduled_end in telescope_schedule:
            if self._times_overlap(
                target.optimal_time, 
                target.optimal_time + timedelta(hours=target.observation_duration),
                scheduled_start, 
                scheduled_end
            ):
                return False
        
        return True
    
    def _times_overlap(self, start1: datetime, end1: datetime,
                      start2: datetime, end2: datetime) -> bool:
        """Check if two time intervals overlap"""
        return start1 < end2 and start2 < end1
    
    def _schedule_observation(self, target: ObservationTarget, telescope_id: str,
                            scheduled_times: Dict):
        """Schedule observation on telescope"""
        start_time = target.optimal_time
        end_time = start_time + timedelta(hours=target.observation_duration)
        
        # Add to schedule
        self.schedule[telescope_id].append({
            'target_id': target.target_id,
            'start_time': start_time,
            'end_time': end_time,
            'priority': target.priority.value,
            'confidence': target.confidence_score
        })
        
        # Track scheduled time
        if telescope_id not in scheduled_times:
            scheduled_times[telescope_id] = []
        scheduled_times[telescope_id].append((start_time, end_time))
        
        logger.info(f"Scheduled {target.target_id} on {telescope_id} at {start_time}")
    
    def get_schedule(self, telescope_id: str = None) -> Dict:
        """Get current observation schedule"""
        with self.schedule_lock:
            if telescope_id:
                return {telescope_id: self.schedule.get(telescope_id, [])}
            return self.schedule.copy()
    
    def get_schedule_summary(self) -> Dict:
        """Get summary of current schedule"""
        summary = {
            'total_observations': 0,
            'telescopes_utilized': 0,
            'high_priority_observations': 0,
            'next_observation': None
        }
        
        next_observation_time = None
        
        for telescope_id, observations in self.schedule.items():
            if observations:
                summary['telescopes_utilized'] += 1
                summary['total_observations'] += len(observations)
                
                # Count high priority observations
                summary['high_priority_observations'] += sum(
                    1 for obs in observations if obs['priority'] <= 2
                )
                
                # Find next observation
                for obs in observations:
                    if (next_observation_time is None or 
                        obs['start_time'] < next_observation_time):
                        next_observation_time = obs['start_time']
                        summary['next_observation'] = obs
        
        return summary

class FollowUpCoordinator:
    """Coordinate follow-up observations across multiple telescopes"""
    
    def __init__(self, scheduler: ObservationScheduler):
        self.scheduler = scheduler
        self.follow_up_queue = []
        
    def trigger_follow_up(self, target_id: str, urgency: float, 
                         observation_type: str = "photometry"):
        """Trigger follow-up observation for high-confidence detection"""
        try:
            # Create follow-up target
            follow_up_target = ObservationTarget(
                target_id=f"{target_id}_followup",
                ra=0.0,  # Would be filled from original detection
                dec=0.0,
                priority=Priority.HIGH if urgency > 0.7 else Priority.MEDIUM,
                confidence_score=urgency,
                observation_duration=2.0,  # hours
                optimal_time=datetime.now() + timedelta(hours=1),
                telescope_requirements=[observation_type],
                weather_dependency=0.3,
                scientific_value=0.8,
                follow_up_urgency=urgency
            )
            
            # Add to scheduler
            self.scheduler.add_target(follow_up_target)
            
            logger.info(f"Triggered follow-up for {target_id} with urgency {urgency}")
            
        except Exception as e:
            logger.error(f"Error triggering follow-up: {e}")
    
    def coordinate_multi_telescope_observation(self, target_id: str, 
                                             telescopes: List[str]):
        """Coordinate observation across multiple telescopes"""
        # This would implement sophisticated coordination logic
        # For now, just add to each telescope's queue
        for telescope_id in telescopes:
            if telescope_id in self.scheduler.telescopes:
                # Create coordinated observation target
                coordinated_target = ObservationTarget(
                    target_id=f"{target_id}_coord_{telescope_id}",
                    ra=0.0,
                    dec=0.0,
                    priority=Priority.HIGH,
                    confidence_score=0.9,
                    observation_duration=1.0,
                    optimal_time=datetime.now() + timedelta(minutes=30),
                    telescope_requirements=["photometry"],
                    weather_dependency=0.2,
                    scientific_value=0.9,
                    follow_up_urgency=0.8
                )
                
                self.scheduler.add_target(coordinated_target)
        
        logger.info(f"Coordinated observation for {target_id} across {len(telescopes)} telescopes")

# Example usage and configuration
def create_sample_telescopes() -> List[Telescope]:
    """Create sample telescopes for testing"""
    telescopes = [
        Telescope(
            telescope_id="TESS",
            location=(28.5, -80.6),  # Florida
            aperture=0.1,
            capabilities=["photometry"],
            current_weather={"cloud_cover": 0.2, "seeing": 0.8},
            operational_hours=(0, 24),
            maintenance_schedule=[],
            efficiency=0.95
        ),
        Telescope(
            telescope_id="Kepler",
            location=(40.0, -105.0),  # Colorado
            aperture=0.95,
            capabilities=["photometry"],
            current_weather={"cloud_cover": 0.1, "seeing": 0.9},
            operational_hours=(18, 6),  # Night time
            maintenance_schedule=[],
            efficiency=0.88
        ),
        Telescope(
            telescope_id="JWST",
            location=(0.0, 0.0),  # Space
            aperture=6.5,
            capabilities=["photometry", "spectroscopy", "imaging"],
            current_weather={"cloud_cover": 0.0, "seeing": 1.0},
            operational_hours=(0, 24),
            maintenance_schedule=[],
            efficiency=0.99
        )
    ]
    return telescopes
