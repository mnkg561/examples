from prometheus_api_client import PrometheusConnect
import numpy as np
from datetime import datetime, timedelta
import math

class EnhancedHPAAnalyzer:
    def __init__(self, prometheus_url, namespace, deployment):
        """
        Initialize Enhanced HPA Analyzer with Prometheus connection.
        
        Args:
            prometheus_url (str): Prometheus server URL
            namespace (str): Kubernetes namespace
            deployment (str): Deployment name
        """
        self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        self.namespace = namespace
        self.deployment = deployment

    def get_pod_cpu_request(self):
        """Get CPU request for pods in the deployment"""
        query = f'''
        max(
          kube_pod_container_resource_requests{{
            namespace="{self.namespace}",
            pod=~"{self.deployment}-.*",
            resource="cpu"
          }}
        )
        '''
        result = self.prom.custom_query(query)
        return float(result[0]['value'][1]) if result else None

    def get_current_replicas(self):
        """Get current number of replicas"""
        query = f'kube_deployment_spec_replicas{{namespace="{self.namespace}", deployment="{self.deployment}"}}'
        result = self.prom.custom_query(query)
        return int(result[0]['value'][1]) if result else None

    def get_cpu_metrics(self, days=14):
        """
        Get detailed CPU metrics including absolute usage and utilization
        
        Returns:
            dict: CPU usage statistics and current HPA target
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get absolute CPU usage
        query = f'''
        sum(
          rate(
            container_cpu_usage_seconds_total{{
              namespace="{self.namespace}",
              pod=~"{self.deployment}-.*",
              container!=""
            }}[5m]
          )
        )
        '''
        
        result = self.prom.custom_query_range(
            query,
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            step='5m'
        )
        
        if not result:
            return None
            
        # Extract CPU usage values
        cpu_usage = [float(v[1]) for v in result[0]['values']]
        
        # Get current HPA target utilization
        hpa_query = f'''
        kube_horizontalpodautoscaler_spec_target_metric{{
          namespace="{self.namespace}",
          name=~"{self.deployment}-.*",
          metric="cpu"
        }}
        '''
        hpa_result = self.prom.custom_query(hpa_query)
        current_hpa_target = float(hpa_result[0]['value'][1]) if hpa_result else None
        
        return {
            'cpu_usage': cpu_usage,
            'current_hpa_target': current_hpa_target
        }

    def analyze_and_recommend(self, standard_threshold=45):
        """
        Analyze metrics and recommend optimal HPA settings
        
        Args:
            standard_threshold (float): Standard CPU utilization threshold percentage
            
        Returns:
            dict: Detailed analysis and recommendations
        """
        current_replicas = self.get_current_replicas()
        cpu_request = self.get_pod_cpu_request()
        metrics = self.get_cpu_metrics()
        
        if not all([current_replicas, cpu_request, metrics]):
            return {"error": "Could not fetch required metrics"}

        # Calculate statistics
        cpu_usage_array = np.array(metrics['cpu_usage'])
        total_cpu_capacity = current_replicas * cpu_request
        utilization_percentages = (cpu_usage_array / total_cpu_capacity) * 100
        
        stats = {
            'min_utilization': np.min(utilization_percentages),
            'max_utilization': np.max(utilization_percentages),
            'p95_utilization': np.percentile(utilization_percentages, 95),
            'mean_utilization': np.mean(utilization_percentages),
            'current_hpa_target': metrics['current_hpa_target']
        }
        
        # Calculate new target based on p95 if current utilization is low
        new_target = None
        if stats['mean_utilization'] < standard_threshold and stats['p95_utilization'] > stats['mean_utilization']:
            new_target = min(stats['p95_utilization'], standard_threshold)
        
        # Calculate minimum replicas based on lowest utilization period
        min_cpu_usage = np.min(cpu_usage_array)
        if new_target:
            recommended_replicas = math.ceil(
                min_cpu_usage / (cpu_request * (new_target / 100))
            )
        else:
            recommended_replicas = current_replicas
        
        return {
            'current_state': {
                'replicas': current_replicas,
                'cpu_request_cores': cpu_request,
                'total_cpu_capacity_cores': total_cpu_capacity,
                'utilization_stats': stats
            },
            'recommendation': {
                'current_hpa_target': stats['current_hpa_target'],
                'recommended_hpa_target': new_target if new_target else stats['current_hpa_target'],
                'recommended_min_replicas': recommended_replicas,
                'potential_savings': (
                    (current_replicas - recommended_replicas) / current_replicas * 100
                    if recommended_replicas < current_replicas else 0
                )
            },
            'calculation_explanation': self._generate_calculation_explanation(
                min_cpu_usage, cpu_request, new_target if new_target else stats['current_hpa_target'],
                current_replicas, recommended_replicas
            )
        }
    
    def _generate_calculation_explanation(self, min_cpu, cpu_request, target_utilization, 
                                       current_replicas, recommended_replicas):
        """Generate detailed explanation of calculations"""
        return [
            f"Minimum CPU usage observed: {min_cpu:.2f} cores",
            f"CPU request per pod: {cpu_request:.2f} cores",
            f"Target utilization: {target_utilization:.1f}%",
            f"Calculation: {min_cpu:.2f} / ({cpu_request:.2f} * {target_utilization/100:.2f}) = {recommended_replicas}",
            f"Current replicas: {current_replicas} -> Recommended: {recommended_replicas}",
            "Note: Final recommendation rounded up to ensure capacity"
        ]

def main():
    """Example usage with test data"""
    class TestPrometheusConnect:
        def custom_query(self, query):
            # Simulate test data
            if 'resource_requests' in query:
                return [{'value': [0, '10']}]  # 10 CPU cores request
            elif 'replicas' in query:
                return [{'value': [0, '100']}]  # 100 current replicas
            else:
                return [{'value': [0, '25']}]  # 25% current HPA target

        def custom_query_range(self, query, **kwargs):
            # Simulate 14 days of metrics with cyclical pattern
            hours = np.arange(0, 14 * 24)
            base_usage = 70  # 70 cores base usage
            daily_pattern = 30 * np.sin(2 * np.pi * hours / 24)  # ±30 cores variation
            usage_values = base_usage + daily_pattern
            
            return [{
                'values': [[h * 3600, str(v)] for h, v in zip(hours, usage_values)]
            }]

    # Initialize analyzer with test data
    analyzer = EnhancedHPAAnalyzer(
        prometheus_url="http://prometheus:9090",
        namespace="test-namespace",
        deployment="test-deployment"
    )
    analyzer.prom = TestPrometheusConnect()
    
    # Get recommendations
    results = analyzer.analyze_and_recommend(standard_threshold=45)
    
    # Print results
    print("\nCurrent State:")
    current = results['current_state']
    print(f"Replicas: {current['replicas']}")
    print(f"CPU Request per Pod: {current['cpu_request_cores']} cores")
    print(f"Total CPU Capacity: {current['total_cpu_capacity_cores']} cores")
    
    print("\nUtilization Statistics:")
    stats = current['utilization_stats']
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}%")
    
    print("\nRecommendation:")
    rec = results['recommendation']
    print(f"Current HPA Target: {rec['current_hpa_target']:.1f}%")
    print(f"Recommended HPA Target: {rec['recommended_hpa_target']:.1f}%")
    print(f"Recommended Min Replicas: {rec['recommended_min_replicas']}")
    print(f"Potential Resource Savings: {rec['potential_savings']:.2f}%")
    
    print("\nCalculation Explanation:")
    for line in results['calculation_explanation']:
        print(f"- {line}")

if __name__ == "__main__":
    main()
