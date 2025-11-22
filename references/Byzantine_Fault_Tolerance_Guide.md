# Byzantine Fault Tolerance Strategies for Federated Learning

## Overview

This guide documents Byzantine fault tolerance strategies for robust federated learning systems. These defense mechanisms protect against malicious clients that may attempt to poison the global model through adversarial updates.

## Problem Statement

In federated learning, Byzantine clients can:
- Submit poisoned gradients to degrade model performance
- Coordinate attacks to overwhelm defensive measures
- Mimic legitimate updates while subtly introducing bias
- Launch targeted attacks against specific data classes

## Defense Strategies

### 1. Trust-Based Byzantine Detection

**Principle**: Track client behavior over time to build trust scores.

**Approach**:
- Initialize all clients with equal trust scores
- Monitor gradient submission patterns and model contributions
- Penalize clients whose updates consistently diverge from expected behavior
- Gradually exclude low-trust clients from aggregation

**Implementation Points**:
```python
# In framework/server.py
class TrustTracker:
    def __init__(self, num_clients):
        self.trust_scores = {i: 1.0 for i in range(num_clients)}
        self.contribution_history = {i: [] for i in range(num_clients)}
    
    def update_trust(self, client_id, gradient_quality_score):
        # Update trust based on gradient alignment with consensus
        pass
```

**Pros**: 
- Adaptive to client behavior patterns
- Maintains historical context
- Can detect sophisticated attacks

**Cons**:
- Requires initialization period
- May penalize legitimate but atypical clients

### 2. Krum Algorithm

**Principle**: Select the client update closest to the majority of other updates.

**Mathematical Foundation**:
- For each client gradient, calculate sum of squared distances to k nearest neighbors
- Select gradient with minimum sum (most central in parameter space)
- Robust against up to f < (n-k-2)/2 Byzantine clients

**Implementation Points**:
```python
# In framework/server.py
def krum_aggregation(client_gradients, num_byzantines):
    """
    Select most representative gradient using Krum algorithm
    """
    scores = []
    n = len(client_gradients)
    k = n - num_byzantines - 2
    
    for i, grad_i in enumerate(client_gradients):
        distances = []
        for j, grad_j in enumerate(client_gradients):
            if i != j:
                dist = torch.norm(grad_i - grad_j).item() ** 2
                distances.append(dist)
        
        # Sum of k smallest distances
        k_smallest = sorted(distances)[:k]
        scores.append((sum(k_smallest), i))
    
    # Return gradient with minimum score
    _, best_idx = min(scores)
    return client_gradients[best_idx]
```

**Pros**:
- Strong theoretical guarantees
- Works well with known Byzantine count
- Single-gradient selection reduces attack surface

**Cons**:
- Requires knowledge of Byzantine client count
- May be overly conservative in gradient selection

### 3. Bulyan Algorithm

**Principle**: Combine Krum with coordinate-wise median for enhanced robustness.

**Two-Phase Approach**:
1. **Selection Phase**: Use Krum to select θ closest gradients
2. **Aggregation Phase**: Apply coordinate-wise median to selected gradients

**Implementation Points**:
```python
# In framework/server.py
def bulyan_aggregation(client_gradients, num_byzantines):
    """
    Bulyan: Krum selection + coordinate-wise median
    """
    n = len(client_gradients)
    theta = n - 2 * num_byzantines
    
    # Phase 1: Select theta closest gradients using modified Krum
    selected_gradients = []
    remaining_gradients = client_gradients.copy()
    
    for _ in range(theta):
        scores = []
        for i, grad_i in enumerate(remaining_gradients):
            distances = []
            for j, grad_j in enumerate(remaining_gradients):
                if i != j:
                    dist = torch.norm(grad_i - grad_j).item() ** 2
                    distances.append(dist)
            
            k = len(remaining_gradients) - num_byzantines - 2
            k_smallest = sorted(distances)[:k]
            scores.append((sum(k_smallest), i))
        
        _, best_idx = min(scores)
        selected_gradients.append(remaining_gradients.pop(best_idx))
    
    # Phase 2: Coordinate-wise median
    stacked_gradients = torch.stack(selected_gradients)
    return torch.median(stacked_gradients, dim=0)[0]
```

**Pros**:
- Combines selection robustness with aggregation robustness
- Better performance than pure Krum under diverse attacks
- Maintains gradient diversity in aggregation

**Cons**:
- Computationally more expensive than single-phase methods
- Still requires Byzantine count estimation

### 4. PID-Based Defense (Project Focus)

**Principle**: Use Proportional-Integral-Derivative control theory to detect anomalous gradient behavior.

**Control System Analogy**:
- **Proportional (P)**: Current gradient deviation from expected norm
- **Integral (I)**: Cumulative deviation over time (persistent attacks)
- **Derivative (D)**: Rate of change in deviation (sudden attacks)

**Implementation Framework**:
```python
# In framework/server.py
class PIDDefense:
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.05):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        
        self.previous_error = 0
        self.integral = 0
        self.gradient_history = []
    
    def detect_anomaly(self, client_gradient, expected_gradient):
        """
        PID-based anomaly detection for client gradient
        """
        # Proportional term: current deviation
        error = torch.norm(client_gradient - expected_gradient).item()
        
        # Integral term: cumulative deviation
        self.integral += error
        
        # Derivative term: rate of change
        derivative = error - self.previous_error
        
        # PID output (anomaly score)
        pid_output = (self.Kp * error + 
                     self.Ki * self.integral + 
                     self.Kd * derivative)
        
        self.previous_error = error
        return pid_output
```

**Pros**:
- Theoretically grounded in control systems
- Adapts to different attack patterns (sudden, persistent, varying)
- Can be tuned for specific threat models

**Cons**:
- Requires careful parameter tuning (Kp, Ki, Kd)
- May need baseline establishment period

### 5. Robust Federated Aggregation (RFA)

**Principle**: Use robust statistical methods to handle outlier gradients.

**Geometric Median Approach**:
- Find gradient that minimizes sum of distances to all client gradients
- More robust than arithmetic mean against outliers
- Computationally tractable with iterative algorithms

**Implementation Points**:
```python
# In framework/server.py
def geometric_median_aggregation(client_gradients, max_iterations=100, tolerance=1e-6):
    """
    Compute geometric median of client gradients
    """
    # Initialize with arithmetic mean
    current_median = torch.mean(torch.stack(client_gradients), dim=0)
    
    for iteration in range(max_iterations):
        # Compute weights (inverse distances)
        weights = []
        total_weight = 0
        
        for grad in client_gradients:
            distance = torch.norm(grad - current_median).item()
            if distance > tolerance:
                weight = 1.0 / distance
            else:
                weight = 1.0 / tolerance  # Avoid division by zero
            weights.append(weight)
            total_weight += weight
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Update median
        new_median = torch.zeros_like(current_median)
        for grad, weight in zip(client_gradients, weights):
            new_median += weight * grad
        
        # Check convergence
        if torch.norm(new_median - current_median).item() < tolerance:
            break
        
        current_median = new_median
    
    return current_median
```

**Trimmed Mean Alternative**:
```python
def trimmed_mean_aggregation(client_gradients, trim_ratio=0.2):
    """
    Aggregate using trimmed mean (remove extreme values)
    """
    n_gradients = len(client_gradients)
    n_trim = int(n_gradients * trim_ratio)
    
    # Stack gradients for coordinate-wise processing
    stacked_gradients = torch.stack(client_gradients)
    
    # Sort each coordinate and trim extremes
    sorted_gradients, _ = torch.sort(stacked_gradients, dim=0)
    trimmed_gradients = sorted_gradients[n_trim:-n_trim] if n_trim > 0 else sorted_gradients
    
    # Return mean of trimmed gradients
    return torch.mean(trimmed_gradients, dim=0)
```

**Pros**:
- Statistically robust against outliers
- Parameter-free (geometric median) or simple tuning (trimmed mean)
- Well-established mathematical foundation

**Cons**:
- May remove legitimate diversity in gradients
- Geometric median computation can be expensive

## Integration Strategy

### Hybrid Defense Architecture

Combine multiple strategies for enhanced robustness:

```python
# In framework/server.py
class HybridByzantineDefense:
    def __init__(self, strategy='adaptive'):
        self.trust_tracker = TrustTracker()
        self.pid_defense = PIDDefense()
        self.strategy = strategy
    
    def aggregate_gradients(self, client_gradients, client_ids, round_num):
        """
        Multi-layer Byzantine defense aggregation
        """
        # Layer 1: Trust-based filtering
        trusted_gradients = []
        trusted_ids = []
        
        for grad, client_id in zip(client_gradients, client_ids):
            if self.trust_tracker.is_trusted(client_id):
                trusted_gradients.append(grad)
                trusted_ids.append(client_id)
        
        if len(trusted_gradients) < 3:  # Fallback to all clients
            trusted_gradients = client_gradients
            trusted_ids = client_ids
        
        # Layer 2: Algorithm selection based on threat assessment
        if self.strategy == 'conservative':
            return krum_aggregation(trusted_gradients, estimated_byzantines=1)
        elif self.strategy == 'balanced':
            return bulyan_aggregation(trusted_gradients, estimated_byzantines=1)
        elif self.strategy == 'adaptive':
            # Use PID to assess threat level and choose accordingly
            threat_level = self.assess_threat_level(trusted_gradients)
            if threat_level > 0.7:
                return krum_aggregation(trusted_gradients, estimated_byzantines=2)
            elif threat_level > 0.3:
                return bulyan_aggregation(trusted_gradients, estimated_byzantines=1)
            else:
                return geometric_median_aggregation(trusted_gradients)
        
        # Default: robust aggregation
        return geometric_median_aggregation(trusted_gradients)
```

### Configuration Integration

Update `cfg/project.yaml` to include Byzantine defense parameters:

```yaml
# Byzantine fault tolerance settings
byzantine_defense:
  enabled: true
  strategy: "adaptive"  # conservative, balanced, adaptive
  trust_threshold: 0.5
  pid_gains:
    kp: 1.0
    ki: 0.1
    kd: 0.05
  krum_params:
    max_byzantines: 2
  bulyan_params:
    max_byzantines: 2
  rfa_params:
    trim_ratio: 0.2
    geometric_median_tolerance: 1e-6
```

## Evaluation Metrics

### Robustness Metrics
- **Attack Success Rate**: Percentage of successful Byzantine attacks
- **Model Accuracy Degradation**: Performance loss under Byzantine clients
- **Detection Accuracy**: True positive/false positive rates for Byzantine identification
- **Convergence Rate**: Training rounds to reach target accuracy

### Efficiency Metrics
- **Computational Overhead**: Additional processing time per round
- **Communication Overhead**: Extra bandwidth for defense mechanisms
- **Memory Usage**: Additional memory requirements for defense state

### Implementation Roadmap

1. **Phase 1**: Implement basic trust tracking and PID defense
2. **Phase 2**: Add Krum and Bulyan algorithms with configurable parameters
3. **Phase 3**: Integrate RFA methods (geometric median, trimmed mean)
4. **Phase 4**: Develop hybrid defense with adaptive strategy selection
5. **Phase 5**: Comprehensive evaluation against various attack scenarios

## References

- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. NIPS.
- Mhamdi, E. M. E., Guerraoui, R., & Rouault, S. (2018). The hidden vulnerability of distributed learning in Byzantium. ICML.
- Yin, D., Chen, Y., Ramchandran, K., & Bartlett, P. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. ICML.
- Xie, C., Koyejo, O., & Gupta, I. (2019). Generalized Byzantine-tolerant SGD. arXiv preprint arXiv:1802.10116.

## Integration with PID Defense

The Byzantine strategies complement the core PID defense by:
1. **Pre-filtering**: Remove obviously malicious clients before PID analysis
2. **Multi-layer validation**: Use PID signals to validate Byzantine detection results
3. **Adaptive response**: Adjust PID parameters based on detected Byzantine activity
4. **Robustness enhancement**: Provide fallback mechanisms when PID defense is insufficient

This comprehensive approach ensures robust federated learning even under sophisticated adversarial conditions.