#!/usr/bin/env python3
"""
Kubernetes HPA Optimizer
This script analyzes Prometheus metrics to optimize HPA settings.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
import logging
from requests.auth import HTTPBasicAuth
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HPAConfig:
    """Configuration for HPA optimization"""
    namespace: str
    deployment_name: str
    target_cpu_utilization: float
    prometheus_url: str
    prometheus_user: str
    prometheus_pass: str

class PrometheusHPACollector:
    """Collects HPA-related metrics from Prometheus"""
    
    def __init__(self, config: HPAConfig):
        self.config = config
        self.auth = HTTPBasicAuth(config.prometheus_user, config.prometheus_pass)
    
    def query_range(self, query: str, start_time: datetime, end_time: datetime, step: str = "5m") -> dict:
        """Execute a range query against Prometheus"""
        params = {
            'query': query,
            'start': start_time.isoformat('T') + 'Z',
            'end': end_time.isoformat('T') + 'Z',
            'step': step
        }
        
        try:
            response = requests.get(
                f"{self.config.prometheus_url}/api/v1/query_range",
                params=params,
                auth=self.auth,
                verify=True,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Prometheus query failed: {str(e)}")
            raise

    def get_cpu_metrics(self, start_time: datetime, end_time: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get detailed CPU metrics for each pod"""
        
        # Query CPU usage for each pod
        per_pod_cpu_usage_query = f"""
        sum by (pod) (
            rate(container_cpu_usage_seconds_total{{
                namespace="{self.config.namespace}",
                pod=~"{self.config.deployment_name}-[a-zA-Z0-9]+",
                container!=""
            }}[5m])
        )
        """
        
        # Query CPU requests for each pod
        per_pod_cpu_requests_query = f"""
        sum by (pod) (
            kube_pod_container_resource_requests{{
                namespace="{self.config.namespace}",
                pod=~"{self.config.deployment_name}-[a-zA-Z0-9]+",
                resource="cpu"
            }}
        )
        """
        
        try:
            logger.info("Fetching metrics from Prometheus...")
            cpu_usage_data = self.query_range(per_pod_cpu_usage_query, start_time, end_time)
            cpu_requests_data = self.query_range(per_pod_cpu_requests_query, start_time, end_time)
            
            # Process metrics
            usage_df = self._process_per_pod_metrics(cpu_usage_data, 'cpu_usage')
            requests_df = self._process_per_pod_metrics(cpu_requests_data, 'cpu_requests')
            
            # Merge metrics
            merged_df = pd.merge(usage_df, requests_df, on=['timestamp', 'pod'], how='outer')
            merged_df['cpu_utilization'] = (merged_df['cpu_usage'] / merged_df['cpu_requests']) * 100
            
            # Calculate aggregates
            agg_df = merged_df.groupby('timestamp').agg({
                'cpu_usage': 'sum',
                'cpu_requests': 'sum',
                'cpu_utilization': 'mean',
                'pod': 'count'
            }).reset_index()
            
            agg_df.rename(columns={'pod': 'replica_count'}, inplace=True)
            
            return merged_df, agg_df
            
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {str(e)}")
            raise

    def _process_per_pod_metrics(self, metric_data: dict, metric_name: str) -> pd.DataFrame:
        """Process raw Prometheus metrics into a DataFrame"""
        processed_data = []
        
        if 'data' not in metric_data or 'result' not in metric_data['data']:
            raise ValueError(f"Invalid metric data format for {metric_name}")
            
        for result in metric_data['data']['result']:
            pod_name = result['metric'].get('pod', 'unknown')
            
            for timestamp, value in result['values']:
                processed_data.append({
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'pod': pod_name,
                    metric_name: float(value)
                })
        
        return pd.DataFrame(processed_data)

def plot_metrics(per_pod_df: pd.DataFrame, agg_df: pd.DataFrame, config: HPAConfig):
    """Create comprehensive visualizations"""
    fig = plt.figure(figsize=(15, 20))
    gs = plt.GridSpec(4, 1, height_ratios=[1.5, 1, 1, 1])

    # 1. Distribution Plot
    ax1 = fig.add_subplot(gs[0])
    hourly_stats = per_pod_df.set_index('timestamp').resample('1H').agg({
        'cpu_utilization': ['mean', 'std', 'min', 'max', 
                           lambda x: np.percentile(x, 25),
                           lambda x: np.percentile(x, 75)]
    })
    hourly_stats.columns = ['mean', 'std', 'min', 'max', '25th', '75th']
    
    ax1.fill_between(hourly_stats.index, 
                    hourly_stats['25th'], 
                    hourly_stats['75th'],
                    alpha=0.3, label='25th-75th Percentile')
    ax1.plot(hourly_stats.index, hourly_stats['mean'], 
             label='Mean', color='blue', linewidth=2)
    ax1.plot(hourly_stats.index, hourly_stats['min'], 
             label='Min', color='green', alpha=0.5)
    ax1.plot(hourly_stats.index, hourly_stats['max'], 
             label='Max', color='red', alpha=0.5)
    ax1.axhline(y=config.target_cpu_utilization, color='red', linestyle='--',
                label=f'Target ({config.target_cpu_utilization}%)')
    ax1.set_title('CPU Utilization Distribution Across All Pods')
    ax1.set_ylabel('Utilization %')
    ax1.legend()

    # 2. Heatmap
    ax2 = fig.add_subplot(gs[1])
    pivot_df = per_pod_df.pivot_table(
        values='cpu_utilization',
        index=pd.Grouper(key='timestamp', freq='1H'),
        columns='pod',
        aggfunc='mean'
    )
    im = ax2.imshow(pivot_df.T, aspect='auto', cmap='YlOrRd')
    ax2.set_title('Pod CPU Utilization Heatmap')
    ax2.set_ylabel('Pod Index')
    plt.colorbar(im, ax=ax2, label='CPU Utilization %')

    # 3. Total CPU Usage
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(agg_df['timestamp'], agg_df['cpu_usage'], 
             label='Total CPU Usage (cores)')
    ax3.plot(agg_df['timestamp'], agg_df['cpu_requests'], 
             label='Total CPU Requests (cores)')
    ax3.set_title('Total CPU Usage vs Requests')
    ax3.set_ylabel('CPU Cores')
    ax3.legend()

    # 4. Replica Count
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(agg_df['timestamp'], agg_df['replica_count'], 
             label='Replica Count')
    ax4.set_title('Replica Count Over Time')
    ax4.set_ylabel('Count')
    ax4.legend()

    plt.tight_layout()
    return fig

def analyze_metrics(per_pod_df: pd.DataFrame, agg_df: pd.DataFrame, config: HPAConfig) -> Dict:
    """Analyze metrics and generate recommendations"""
    
    # Calculate key statistics
    stats = {
        'current': {
            'avg_utilization': agg_df['cpu_utilization'].mean(),
            'p95_utilization': agg_df['cpu_utilization'].quantile(0.95),
            'max_utilization': agg_df['cpu_utilization'].max(),
            'min_utilization': agg_df['cpu_utilization'].min(),
            'avg_replicas': agg_df['replica_count'].mean(),
            'max_replicas': agg_df['replica_count'].max(),
            'min_replicas': agg_df['replica_count'].min()
        }
    }
    
    # Generate recommendations
    recommended_target = min(max(stats['current']['p95_utilization'], 
                               config.target_cpu_utilization), 80)
    
    min_cores_needed = agg_df['cpu_usage'].quantile(0.05)
    avg_pod_request = agg_df['cpu_requests'].mean() / agg_df['replica_count'].mean()
    
    recommended_min_replicas = max(1, int(np.ceil(
        min_cores_needed / (avg_pod_request * (recommended_target / 100))
    )))
    
    stats['recommendations'] = {
        'target_utilization': recommended_target,
        'min_replicas': recommended_min_replicas,
        'reason': f"Based on P95 utilization of {stats['current']['p95_utilization']:.1f}% "
                 f"and minimum cores needed of {min_cores_needed:.2f}"
    }
    
    return stats

def main():
    """Main execution function"""
    if len(sys.argv) < 7:
        print("""
Usage: python3 hpa_optimizer.py <namespace> <deployment_name> <target_cpu> 
       <prometheus_url> <prometheus_user> <prometheus_pass>
Example: python3 hpa_optimizer.py default my-app 50 
         http://prometheus:9090 admin password
        """)
        sys.exit(1)

    # Parse command line arguments
    config = HPAConfig(
        namespace=sys.argv[1],
        deployment_name=sys.argv[2],
        target_cpu_utilization=float(sys.argv[3]),
        prometheus_url=sys.argv[4],
        prometheus_user=sys.argv[5],
        prometheus_pass=sys.argv[6]
    )

    try:
        # Initialize collector
        collector = PrometheusHPACollector(config)
        
        # Set time range for analysis (last 14 days)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=14)
        
        # Collect metrics
        logger.info("Collecting metrics for the last 14 days...")
        per_pod_df, agg_df = collector.get_cpu_metrics(start_time, end_time)
        
        # Analyze metrics
        logger.info("Analyzing metrics...")
        stats = analyze_metrics(per_pod_df, agg_df, config)
        
        # Create visualizations
        logger.info("Generating visualizations...")
        fig = plot_metrics(per_pod_df, agg_df, config)
        
        # Print results
        print("\nCurrent Statistics:")
        for key, value in stats['current'].items():
            print(f"{key}: {value:.2f}")
        
        print("\nRecommendations:")
        print(f"Target CPU Utilization: {stats['recommendations']['target_utilization']:.1f}%")
        print(f"Minimum Replicas: {stats['recommendations']['min_replicas']}")
        print(f"Reasoning: {stats['recommendations']['reason']}")
        
        # Save plot
        plot_filename = f"hpa_analysis_{config.namespace}_{config.deployment_name}.png"
        fig.savefig(plot_filename)
        print(f"\nPlot saved as: {plot_filename}")
        
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
