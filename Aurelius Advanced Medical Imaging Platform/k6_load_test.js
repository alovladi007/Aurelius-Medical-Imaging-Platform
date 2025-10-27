/**
 * k6 Load Testing Script for Aurelius API
 * 
 * Installation:
 *   brew install k6  (macOS)
 *   sudo apt install k6  (Ubuntu)
 * 
 * Usage:
 *   # Basic smoke test (1 VU, 30s)
 *   k6 run k6_load_test.js
 * 
 *   # Load test (10 VUs, 5 minutes)
 *   k6 run --vus 10 --duration 5m k6_load_test.js
 * 
 *   # Stress test (ramp up to 100 VUs)
 *   k6 run --stage 1m:10,5m:50,2m:100,5m:100,2m:0 k6_load_test.js
 * 
 *   # With thresholds
 *   k6 run --vus 50 --duration 10m --out json=results.json k6_load_test.js
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const searchLatency = new Trend('search_latency');
const mlLatency = new Trend('ml_latency');
const apiCalls = new Counter('api_calls_total');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const SEARCH_URL = __ENV.SEARCH_URL || 'http://localhost:8004';

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 10 },   // Stay at 10 for 5 minutes
    { duration: '2m', target: 50 },   // Ramp to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 for 5 minutes
    { duration: '2m', target: 0 },    // Ramp down to 0
  ],
  thresholds: {
    // SLO: 99.5% requests must complete within 500ms
    'http_req_duration': ['p(95)<500', 'p(99)<1000'],
    
    // SLO: Error rate must be below 1%
    'errors': ['rate<0.01'],
    
    // SLO: 95% of requests must succeed
    'http_req_failed': ['rate<0.05'],
    
    // Custom metrics thresholds
    'search_latency': ['p(95)<1000'],
    'ml_latency': ['p(95)<2000'],
  },
};

// Authentication helper
function login() {
  const loginRes = http.post(`${BASE_URL}/auth/login`, JSON.stringify({
    username: 'doctor',
    password: 'doctor123',
  }), {
    headers: { 'Content-Type': 'application/json' },
  });

  check(loginRes, {
    'login successful': (r) => r.status === 200,
  });

  if (loginRes.status === 200) {
    const body = JSON.parse(loginRes.body);
    return body.access_token;
  }
  
  return null;
}

// Main test scenario
export default function () {
  // Get authentication token
  const token = login();
  if (!token) {
    errorRate.add(1);
    return;
  }

  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  // Scenario 1: Browse Studies
  group('Browse Studies', function () {
    const listRes = http.get(`${BASE_URL}/studies?page=1&page_size=20`, { headers });
    
    apiCalls.add(1);
    
    check(listRes, {
      'status is 200': (r) => r.status === 200,
      'response time < 500ms': (r) => r.timings.duration < 500,
    }) || errorRate.add(1);
    
    sleep(1);
  });

  // Scenario 2: Search Studies (Keyword)
  group('Search - Keyword', function () {
    const queries = ['chest', 'brain', 'abdomen', 'spine'];
    const query = queries[Math.floor(Math.random() * queries.length)];
    
    const searchRes = http.post(`${SEARCH_URL}/search`, JSON.stringify({
      query: query,
      page: 1,
      page_size: 20,
    }), { headers });
    
    apiCalls.add(1);
    searchLatency.add(searchRes.timings.duration);
    
    check(searchRes, {
      'search status is 200': (r) => r.status === 200,
      'search has results': (r) => {
        const body = JSON.parse(r.body);
        return body.total !== undefined;
      },
    }) || errorRate.add(1);
    
    sleep(2);
  });

  // Scenario 3: Search with Filters
  group('Search - Filtered', function () {
    const searchRes = http.post(`${SEARCH_URL}/search`, JSON.stringify({
      query: 'xray',
      modalities: ['X-Ray', 'CT'],
      date_from: '2024-01-01',
      page: 1,
      page_size: 20,
    }), { headers });
    
    apiCalls.add(1);
    searchLatency.add(searchRes.timings.duration);
    
    check(searchRes, {
      'filtered search ok': (r) => r.status === 200,
    }) || errorRate.add(1);
    
    sleep(1);
  });

  // Scenario 4: Semantic Search
  group('Search - Semantic', function () {
    const searchRes = http.post(`${SEARCH_URL}/search`, JSON.stringify({
      query: 'lung nodules with high density',
      semantic_search: true,
      page: 1,
      page_size: 20,
    }), { headers });
    
    apiCalls.add(1);
    searchLatency.add(searchRes.timings.duration);
    
    check(searchRes, {
      'semantic search ok': (r) => r.status === 200,
      'semantic search time < 2s': (r) => r.timings.duration < 2000,
    }) || errorRate.add(1);
    
    sleep(2);
  });

  // Scenario 5: ML Prediction
  group('ML Inference', function () {
    const predictRes = http.post(`${BASE_URL}/ml/predict`, JSON.stringify({
      model_name: 'chest-xray-classifier',
      model_version: 'latest',
      input_data: {
        study_id: `study-${Math.floor(Math.random() * 1000)}`,
      },
    }), { headers });
    
    apiCalls.add(1);
    mlLatency.add(predictRes.timings.duration);
    
    check(predictRes, {
      'ml predict accepted': (r) => r.status === 200 || r.status === 202,
      'ml time reasonable': (r) => r.timings.duration < 5000,
    }) || errorRate.add(1);
    
    sleep(3);
  });

  // Scenario 6: Health Check
  group('Health Check', function () {
    const healthRes = http.get(`${BASE_URL}/health`);
    
    check(healthRes, {
      'health check ok': (r) => r.status === 200,
      'health check fast': (r) => r.timings.duration < 100,
    });
    
    sleep(1);
  });

  // Random sleep between 1-3 seconds
  sleep(Math.random() * 2 + 1);
}

// Test lifecycle hooks
export function setup() {
  console.log('='.repeat(60));
  console.log('K6 LOAD TEST STARTING');
  console.log('='.repeat(60));
  console.log(`Target: ${BASE_URL}`);
  console.log(`Search: ${SEARCH_URL}`);
  console.log('='.repeat(60));
}

export function teardown(data) {
  console.log('='.repeat(60));
  console.log('K6 LOAD TEST COMPLETED');
  console.log('='.repeat(60));
}

// Handle summary
export function handleSummary(data) {
  console.log('='.repeat(60));
  console.log('LOAD TEST SUMMARY');
  console.log('='.repeat(60));
  
  const { metrics } = data;
  
  // Print key metrics
  console.log('\n--- HTTP Metrics ---');
  console.log(`Total Requests: ${metrics.http_reqs.values.count}`);
  console.log(`Request Rate: ${metrics.http_reqs.values.rate.toFixed(2)}/s`);
  console.log(`Failed Requests: ${metrics.http_req_failed.values.rate * 100}%`);
  console.log(`Avg Duration: ${metrics.http_req_duration.values.avg.toFixed(2)}ms`);
  console.log(`P95 Duration: ${metrics.http_req_duration.values['p(95)'].toFixed(2)}ms`);
  console.log(`P99 Duration: ${metrics.http_req_duration.values['p(99)'].toFixed(2)}ms`);
  
  console.log('\n--- Custom Metrics ---');
  if (metrics.search_latency) {
    console.log(`Search Avg: ${metrics.search_latency.values.avg.toFixed(2)}ms`);
    console.log(`Search P95: ${metrics.search_latency.values['p(95)'].toFixed(2)}ms`);
  }
  if (metrics.ml_latency) {
    console.log(`ML Avg: ${metrics.ml_latency.values.avg.toFixed(2)}ms`);
    console.log(`ML P95: ${metrics.ml_latency.values['p(95)'].toFixed(2)}ms`);
  }
  if (metrics.api_calls_total) {
    console.log(`Total API Calls: ${metrics.api_calls_total.values.count}`);
  }
  
  console.log('\n--- SLO Compliance ---');
  const p95 = metrics.http_req_duration.values['p(95)'];
  const errorRate = metrics.http_req_failed.values.rate;
  
  console.log(`P95 Latency SLO (<500ms): ${p95 < 500 ? '✅ PASS' : '❌ FAIL'} (${p95.toFixed(2)}ms)`);
  console.log(`Error Rate SLO (<1%): ${errorRate < 0.01 ? '✅ PASS' : '❌ FAIL'} (${(errorRate * 100).toFixed(2)}%)`);
  
  console.log('='.repeat(60));
  
  return {
    'stdout': '', // Don't print default summary
    'results.json': JSON.stringify(data, null, 2),
  };
}
