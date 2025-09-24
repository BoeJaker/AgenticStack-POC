# Request Queue and Resource Management

The enhanced Ollama API Shim now includes sophisticated request queuing and resource management to ensure optimal performance under load and prevent system overload.

## Overview

The queue system automatically manages incoming requests based on:
- **System Resource Availability** (CPU, Memory, Response Times)
- **Request Priority** (user-defined or automatically calculated)
- **Processing Capacity** (concurrent request limits)
- **Queue Limits** (maximum queued requests)

## Key Features

### 🚦 Intelligent Request Queuing
- **Priority-based scheduling**: Requests are processed by priority (1=highest, 10=lowest)
- **Automatic priority calculation**: Based on request characteristics and system state
- **Queue overflow protection**: Rejects requests when queue is full rather than crashing
- **Background processing**: Queued requests are processed asynchronously

### 📊 Resource Monitoring
- **Real-time system monitoring**: CPU, memory, and Ollama response times
- **Automatic overload detection**: Defers requests when system is struggling
- **Resource-aware scheduling**: Adjusts priorities based on system load
- **Performance metrics**: Comprehensive queue and resource metrics

### ⚡ Performance Optimization
- **Streaming request bypass**: Streaming requests get immediate processing for better UX
- **Context preparation pipelining**: Prepares context while starting response stream
- **Intelligent caching**: Reduces processing overhead for repeated operations
- **Graceful degradation**: Maintains service quality under load

## Configuration

### Environment Variables

```env
# Queue Configuration
MAX_CONCURRENT_REQUESTS=10      # Maximum simultaneous processing
MAX_QUEUE_SIZE=100             # Maximum queued requests
QUEUE_TIMEOUT=300              # Queue timeout in seconds

# Resource Thresholds
CPU_THRESHOLD=80.0             # CPU percentage before overload
MEMORY_THRESHOLD=85.0          # Memory percentage before overload
RESPONSE_TIME_THRESHOLD=30.0   # Ollama response time threshold

# Performance Tuning
ENABLE_CACHING=true           # Enable response/embedding caching
REQUEST_TIMEOUT=60            # Individual request timeout
```

### Development vs Production

```python
# Development (fewer resources)
MAX_CONCURRENT_REQUESTS = 3
MAX_QUEUE_SIZE = 20
CPU_THRESHOLD = 70.0

# Production (more resources)
MAX_CONCURRENT_REQUESTS = 20
MAX_QUEUE_SIZE = 500
CPU_THRESHOLD = 80.0
```

## API Usage

### Basic Request with Priority
```bash
curl -X POST http://localhost:8082/api/generate \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: user-123" \
  -H "X-Priority: 3" \
  -d '{
    "model": "llama2",
    "prompt": "Urgent: What is the capital of France?",
    "stream": false
  }'
```

### Queue Status Monitoring
```bash
# Get current queue status
curl http://localhost:8082/api/shim/queue/status

# Response example:
{
  "queue": {
    "active_requests": 3,
    "queue_size": 7,
    "max_concurrent": 10,
    "max_queue_size": 100,
    "queued": 7,
    "processing": 3,
    "completed": 145,
    "failed": 2,
    "queue_full_rejections": 0
  },
  "resources": {
    "cpu_percent": 65.2,
    "memory_percent": 72.1,
    "system_overloaded": false,
    "ollama_avg_response_time": 8.5
  }
}
```

### Resource Status
```bash
curl http://localhost:8082/api/shim/resources

# Response includes current usage and recommendations
{
  "resources": {
    "cpu_percent": 85.3,
    "memory_percent": 78.2,
    "system_overloaded": true
  },
  "thresholds": {
    "cpu_threshold": 80.0,
    "memory_threshold": 85.0,
    "response_time_threshold": 30.0
  },
  "recommendations": [
    "High CPU usage detected. Consider reducing concurrent requests.",
    "System is overloaded. Requests are being queued automatically."
  ]
}
```

### Queue Management (Admin Operations)
```bash
# Flush all queued requests
curl -X POST http://localhost:8082/api/shim/queue/flush

# Cancel specific request (if supported)
curl -X DELETE http://localhost:8082/api/shim/queue/cancel/request-id-123

# Update request priority (if supported)
curl -X POST http://localhost:8082/api/shim/queue/priority \
  -d '{"request_id": "abc-123", "new_priority": 1}'
```

## Priority System

### Automatic Priority Calculation

The system automatically calculates request priority based on:

1. **Request Type**
   - Streaming: Higher priority (immediate user interaction)
   - Short prompts (< 100 chars): Higher priority
   - Long prompts (> 1000 chars): Lower priority

2. **User Context**
   - Returning users: Higher priority
   - New sessions: Normal priority

3. **System Load**
   - High load: Increases all priorities
   - Normal load: No adjustment

### Manual Priority Override

```bash
# High priority (1-3): Urgent requests, short queries
curl -H "X-Priority: 1" ...

# Normal priority (4-6): Regular conversations
curl -H "X-Priority: 5" ...

# Low priority (7-10): Batch processing, long documents
curl -H "X-Priority: 8" ...
```

## Monitoring and Metrics

### Prometheus Metrics

The system exposes comprehensive metrics for monitoring:

```prometheus
# Queue metrics
ollama_shim_queue_size                    # Current queue size
ollama_shim_queue_wait_seconds           # Time requests spend in queue
ollama_shim_active_requests              # Currently processing requests
ollama_shim_queued_requests_total        # Total queued requests by priority
ollama_shim_queue_rejections_total       # Requests rejected due to full queue

# Resource metrics
ollama_shim_system_cpu_percent           # System CPU usage
ollama_shim_system_memory_percent        # System memory usage
ollama_shim_system_overloaded           # System overload status (0/1)
ollama_shim_deferred_requests_total     # Requests deferred due to resource constraints
```

### Grafana Dashboard Panels

Add these panels to your Grafana dashboard:

1. **Queue Status** - Current queue size and active requests
2. **Queue Wait Times** - Distribution of time requests spend queued
3. **System Resources** - CPU, memory, and overload status
4. **Request Throughput** - Requests processed per second by priority
5. **Queue Rejections** - Rate of rejected requests due to capacity

### Health Check Integration

The health endpoint now includes queue and resource status:

```json
{
  "status": "degraded",
  "queue": {
    "size": 45,
    "active_requests": 8,
    "max_concurrent": 10,
    "processing_capacity": "8/10"
  },
  "resources": {
    "cpu_percent": 85.3,
    "memory_percent": 72.1,
    "system_overloaded": true
  }
}
```

## Behavior Under Load

### Normal Operations
- Requests processed immediately if resources available
- Queue remains empty or small
- Response times consistent

### Moderate Load
- Some requests queued during peak times
- Priority-based processing ensures important requests handled first
- Queue drains quickly during low-traffic periods

### High Load / Overload
- Most requests queued automatically
- System monitors resources and defers new requests
- Streaming requests still get immediate processing for UX
- Oldest queued requests may timeout and be cancelled

### Recovery
- As resources become available, queue drains automatically
- System gradually returns to normal processing
- Metrics track recovery progress

## Best Practices

### For Application Developers
1. **Use appropriate priorities**: Don't mark everything as high priority
2. **Implement retry logic**: Handle 503 responses gracefully
3. **Monitor queue status**: Check queue health before sending batches
4. **Set reasonable timeouts**: Allow time for queuing delays

### For System Administrators
1. **Monitor resource metrics**: Set up alerts for high CPU/memory
2. **Tune thresholds**: Adjust based on your hardware capabilities
3. **Scale horizontally**: Add more shim instances if needed
4. **Regular maintenance**: Clear old sessions and caches periodically

### For Load Balancers
```nginx
# Nginx example with queue-aware health checks
upstream ollama_shim {
    server shim1:8082 max_fails=3 fail_timeout=30s;
    server shim2:8082 max_fails=3 fail_timeout=30s;
}

location /health {
    # Custom health check that considers queue status
    proxy_pass http://ollama_shim/api/shim/health;
    proxy_read_timeout 5s;
}
```

## Troubleshooting

### Common Issues

1. **High Queue Times**
   - Check system resources (CPU/memory)
   - Verify Ollama is responding quickly
   - Consider increasing MAX_CONCURRENT_REQUESTS

2. **Queue Full Rejections**
   - Increase MAX_QUEUE_SIZE if you have memory
   - Implement client-side retry with backoff
   - Scale horizontally with multiple shim instances

3. **System Overload False Positives**
   - Adjust CPU_THRESHOLD and MEMORY_THRESHOLD
   - Check for other processes consuming resources
   - Verify psutil is working correctly

4. **Streaming Requests Slow**
   - Streaming bypasses queue but still needs resources
   - Check Ollama performance directly
   - Monitor context preparation time

### Debug Commands
```bash
# Check queue status
curl -s http://localhost:8082/api/shim/queue/status | jq .

# Monitor resources
curl -s http://localhost:8082/api/shim/resources | jq .

# Check health with queue info
curl -s http://localhost:8082/api/shim/health | jq .

# View Prometheus metrics
curl http://localhost:8082/metrics | grep queue
```

This queue system ensures your Ollama API Shim remains responsive and stable even under heavy load, providing a much better user experience than simple request overload failures.