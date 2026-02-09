import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const analyticsService = {
  // Get comprehensive analytics data
  getAnalytics: async (params) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analytics`, { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching analytics:', error);
      // Return mock data
      return {
        accuracyChart: {
          labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
          datasets: [
            {
              label: 'Accuracy %',
              data: [92, 94, 91, 95, 93, 96, 94],
              borderColor: 'rgb(75, 192, 192)',
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              tension: 0.1
            }
          ]
        },
        regionalChart: {
          labels: ['North America', 'Europe', 'Asia', 'Africa', 'South America'],
          datasets: [
            {
              label: 'Performance Score',
              data: [94, 91, 89, 87, 92],
              backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)'
              ],
              borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)'
              ],
              borderWidth: 1
            }
          ]
        },
        predictionTypes: {
          labels: ['Hourly', 'Daily', 'Weekly', 'Monthly'],
          datasets: [
            {
              label: 'Prediction Distribution',
              data: [45, 30, 20, 5],
              backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)'
              ],
              borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)'
              ],
              borderWidth: 1
            }
          ]
        },
        performance: [
          {
            date: '2024-01-20',
            predictions: 342,
            accuracy: 94,
            avgLoss: 0.045,
            status: 'excellent'
          },
          {
            date: '2024-01-19',
            predictions: 298,
            accuracy: 91,
            avgLoss: 0.052,
            status: 'good'
          },
          {
            date: '2024-01-18',
            predictions: 276,
            accuracy: 89,
            avgLoss: 0.061,
            status: 'good'
          },
          {
            date: '2024-01-17',
            predictions: 312,
            accuracy: 93,
            avgLoss: 0.048,
            status: 'excellent'
          },
          {
            date: '2024-01-16',
            predictions: 289,
            accuracy: 87,
            avgLoss: 0.067,
            status: 'fair'
          }
        ],
        regional: [
          {
            region: 'North America',
            totalPredictions: 1247,
            successRate: 94,
            avgConfidence: 91,
            performance: 94
          },
          {
            region: 'Europe',
            totalPredictions: 892,
            successRate: 91,
            avgConfidence: 89,
            performance: 91
          },
          {
            region: 'Asia Pacific',
            totalPredictions: 756,
            successRate: 89,
            avgConfidence: 87,
            performance: 89
          },
          {
            region: 'South America',
            totalPredictions: 423,
            successRate: 92,
            avgConfidence: 90,
            performance: 92
          },
          {
            region: 'Africa',
            totalPredictions: 367,
            successRate: 87,
            avgConfidence: 85,
            performance: 87
          }
        ]
      };
    }
  },

  // Get detailed performance metrics
  getPerformanceMetrics: async (timeRange) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analytics/performance`, {
        params: { timeRange }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
      return {
        avgAccuracy: 92.5,
        avgLoss: 0.052,
        totalPredictions: 3685,
        successRate: 91.2,
        modelUptime: 99.8
      };
    }
  },

  // Get regional analytics
  getRegionalAnalytics: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analytics/regional`);
      return response.data;
    } catch (error) {
      console.error('Error fetching regional analytics:', error);
      return [];
    }
  },

  // Get usage statistics
  getUsageStats: async (timeRange) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analytics/usage`, {
        params: { timeRange }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching usage stats:', error);
      return {
        apiCalls: 15420,
        dataProcessed: '2.4TB',
        activeUsers: 342,
        avgResponseTime: '1.2s'
      };
    }
  },

  // Export analytics data
  exportAnalytics: async (params) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analytics/export`, {
        params,
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `analytics_export_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      return { success: true };
    } catch (error) {
      console.error('Error exporting analytics:', error);
      throw error;
    }
  },

  // Get real-time metrics
  getRealTimeMetrics: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analytics/realtime`);
      return response.data;
    } catch (error) {
      console.error('Error fetching real-time metrics:', error);
      return {
        currentAccuracy: 94.2,
        activePredictions: 12,
        queueLength: 3,
        systemLoad: 67
      };
    }
  },

  // Get model comparison data
  getModelComparison: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analytics/model-comparison`);
      return response.data;
    } catch (error) {
      console.error('Error fetching model comparison:', error);
      return {
        models: [
          {
            name: 'DCGAN v4.0',
            accuracy: 94.5,
            speed: 2.3,
            reliability: 98.2
          },
          {
            name: 'DCGAN v3.0',
            accuracy: 91.2,
            speed: 3.1,
            reliability: 95.8
          }
        ]
      };
    }
  }
};

export default analyticsService;
