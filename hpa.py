import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
import logging
from requests.auth import HTTPBasicAuth
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HPAConfig:
    """Minimal configuration for HPA optimization"""
    namespace: str
    deployment_name: str
    target_cpu_utilization: float

class PrometheusHPACollector:
    """Collects HPA-related metrics from Prometheus"""
    
    def __init__(self, 
                 config: HPAConfig,
                 prometheus_url: str = "http://prometheus.example.com",
                 username: str = "prometheus",
                 password: str = "secret"):
        self.config = config
        self.prometheus_url = prometheus_url
        self.auth = HTTPBasicAuth(username, password)
        
    def query_range(self, query: str, start_time: datetime, end_time: datetime, step: str = "5m") -> dict:
        """Execute a range query against Prometheus"""
        params = {
            'query': query,
            'start': start_time.isoformat('T') + 'Z',
            'end': end_time.isoformat('T') + 'Z',
            'step': step
        }
        
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query_range",
            params=params,
            auth=self.auth,
            verify=True
        )
        
        if response.status_code != 200:
            raise Exception(f"Prometheus query failed: {response.text}")
        
        return response.json()

    def get_cpu_utilization(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get CPU utilization metrics for the deployment"""
        
        # 1. Container CPU Usage Rate (actual CPU cores used)
        cpu_usage_query = f"""
        sum(
          rate(container_cpu_usage_seconds_total{{
            namespace="{self.config.namespace}",
            pod=~"{self.config.deployment_name}-[a-zA-Z0-9]+",
            container!=""
          }}[5m])
        ) by (pod)
        """
        
        # 2. CPU Requests (allocated CPU cores)
        cpu_requests_query = f"""
        sum(
          kube_pod_container_resource_requests{{
            namespace="{self.config.namespace}",
            pod=~"{self.config.deployment_name}-[a-zA-Z0-9]+",
            resource="cpu"
          }}
        ) by (pod)
        """
        
        # 3. Current HPA target utilization
        hpa_target_query = f"""
        kube_horizontalpodautoscaler_spec_target_metric{{
          namespace="{self.config.namespace}",
          horizontalpodautoscaler=~"{self.config.deployment_name}-.*",
          metric_name="cpu"
        }}
        """
        
        # 4. Current replica count
        replica_query = f"""
        kube_deployment_spec_replicas{{
          namespace="{self.config.namespace}",
          deployment="{self.config.deployment_name}"
        }}
        """
        
        # 5. HPA min replicas
        min_replicas_query = f"""
        kube_horizontalpodautoscaler_spec_min_replicas{{
          namespace="{self.config.namespace}",
          horizontalpodautoscaler=~"{self.config.deployment_name}-.*"
        }}
        """
        
        # 6. HPA max replicas
        max_replicas_query = f"""
        kube_horizontalpodautoscaler_spec_max_replicas{{
          namespace="{self.config.namespace}",
          horizontalpodautoscaler=~"{self.config.deployment_name}-.*"
        }}
        """
        
        # 7. Pod Ready status (for actual running pods)
        ready_pods_query = f"""
        sum(
          kube_pod_status_ready{{
            namespace="{self.config.namespace}",
            pod=~"{self.config.deployment_name}-[a-zA-Z0-9]+",
            condition="true"
          }}
        )
        """
        
        # 8. Container Memory Usage (complementary metric)
        memory_usage_query = f"""
        sum(
          container_memory_usage_bytes{{
            namespace="{self.config.namespace}",
            pod=~"{self.config.deployment_name}-[a-zA-Z0-9]+",
            container!=""
          }}
        ) by (pod)
        """
        
        # Collect all metrics
        try:
            metrics = {
                'cpu_usage': self.query_range(cpu_usage_query, start_time, end_time),
                'cpu_requests': self.query_range(cpu_requests_query, start_time, end_time),
                'hpa_target': self.query_range(hpa_target_query, start_time, end_time),
                'replicas': self.query_range(replica_query, start_time, end_time),
                'min_replicas': self.query_range(min_replicas_query, start_time, end_time),
                'max_replicas': self.query_range(max_replicas_query, start_time, end_time),
                'ready_pods': self.query_range(ready_pods_query, start_time, end_time),
                'memory_usage': self.query_range(memory_usage_query, start_time, end_time)
            }
            
            return self._process_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            raise

    def _process_metrics(self, metrics: dict) -> pd.DataFrame:
        """Process raw Prometheus metrics into a pandas DataFrame"""
        processed_data = []
        
        for metric_name, metric_data in metrics.items():
            if 'data' not in metric_data or 'result' not in metric_data['data']:
                logger.warning(f"No data found for {metric_name}")
                continue
                
            results = metric_data['data']['result']
            if not results:
                logger.warning(f"Empty result set for {metric_name}")
                continue
                
            # Take the first result if multiple series exist
            result = results[0]
            
            for timestamp, value in result['values']:
                processed_data.append({
                    'timestamp': datetime.fromtimestamp(timestamp),
                    metric_name: float(value)
                })
        
        # Convert to DataFrame and handle missing values
        df = pd.DataFrame(processed_data)
        df = df.groupby('timestamp').first().reset_index()
        
        # Calculate CPU utilization percentage
        if 'cpu_usage' in df.columns and 'cpu_requests' in df.columns:
            df['cpu_utilization'] = (df['cpu_usage'] / df['cpu_requests']) * 100
        
        return df

def analyze_hpa_metrics(namespace: str, deployment_name: str, target_cpu: float):
    """Main function to analyze HPA metrics and provide recommendations"""
    
    config = HPAConfig(
        namespace=namespace,
        deployment_name=deployment_name,
        target_cpu_utilization=target_cpu
    )
    
    collector = PrometheusHPACollector(config)
    
    # Get last 14 days of metrics
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=14)
    
    try:
        # Collect metrics
        df = collector.get_cpu_utilization(start_time, end_time)
        
        # Calculate key statistics
        stats = {
            'avg_cpu_util': df['cpu_utilization'].mean(),
            'p95_cpu_util': df['cpu_utilization'].quantile(0.95),
            'min_replicas': df['min_replicas'].iloc[-1],
            'max_replicas': df['max_replicas'].iloc[-1],
            'current_replicas': df['replicas'].iloc[-1]
        }
        
        # Generate recommendations
        recommendations = {
            'suggested_target_cpu': min(max(stats['p95_cpu_util'], target_cpu), 80),
            'suggested_min_replicas': max(1, int(np.ceil(
                df['cpu_usage'].quantile(0.05) / df['cpu_requests'].mean()
            ))),
            'suggested_max_replicas': max(int(np.ceil(
                df['cpu_usage'].max() / df['cpu_requests'].mean()
            )), stats['current_replicas'])
        }
        
        # Plot metrics
        plot_metrics(df)
        
        return stats, recommendations
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

def plot_metrics(df: pd.DataFrame):
    """Plot key metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # CPU Utilization
    ax1.plot(df['timestamp'], df['cpu_utilization'], label='CPU Utilization (%)')
    ax1.set_title('CPU Utilization Over Time')
    ax1.set_ylabel('Utilization %')
    ax1.legend()
    
    # Replica Count
    ax2.plot(df['timestamp'], df['replicas'], label='Replica Count')
    ax2.plot(df['timestamp'], df['ready_pods'], label='Ready Pods', linestyle='--')
    ax2.set_title('Replica Count Over Time')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Example usage"""
    stats, recommendations = analyze_hpa_metrics(
        namespace="default",
        deployment_name="example-app",
        target_cpu=50.0
    )
    
    print("\nCurrent Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nRecommendations:")
    for key, value in recommendations.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
