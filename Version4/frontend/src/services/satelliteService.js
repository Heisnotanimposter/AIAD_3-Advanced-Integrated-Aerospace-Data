import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const satelliteService = {
  // Get dashboard statistics
  getStats: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/stats`);
      return response.data;
    } catch (error) {
      console.error('Error fetching stats:', error);
      // Return mock data for development
      return {
        activeSatellites: 12,
        predictionsToday: 248,
        dataProcessed: 1.2,
        modelAccuracy: 94.5
      };
    }
  },

  // Get recent predictions
  getRecentPredictions: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/predictions/recent`);
      return response.data;
    } catch (error) {
      console.error('Error fetching recent predictions:', error);
      // Return mock data
      return [
        {
          id: 1,
          location: 'North America',
          timestamp: '2024-01-20 14:30:00',
          accuracy: 94,
          status: 'completed'
        },
        {
          id: 2,
          location: 'Europe',
          timestamp: '2024-01-20 14:15:00',
          accuracy: 91,
          status: 'completed'
        },
        {
          id: 3,
          location: 'Asia Pacific',
          timestamp: '2024-01-20 14:00:00',
          accuracy: 89,
          status: 'processing'
        }
      ];
    }
  },

  // Get chart data for dashboard
  getChartData: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/charts/performance`);
      return response.data;
    } catch (error) {
      console.error('Error fetching chart data:', error);
      // Return mock chart data
      return {
        labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
        datasets: [
          {
            label: 'Generator Loss',
            data: [0.5, 0.45, 0.4, 0.35, 0.3, 0.25],
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.1
          },
          {
            label: 'Discriminator Loss',
            data: [0.6, 0.55, 0.5, 0.4, 0.35, 0.3],
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.1
          }
        ]
      };
    }
  },

  // Get satellite imagery
  getSatelliteImagery: async (params) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/satellite/imagery`, { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching satellite imagery:', error);
      // Return mock data
      return {
        images: [
          {
            id: 1,
            title: 'GOES-16 Full Disk',
            url: 'https://picsum.photos/seed/sat1/400/300.jpg',
            timestamp: '2024-01-20 14:30:00',
            resolution: '2km',
            filename: 'goes16_full_disk_20240120_1430.jpg'
          },
          {
            id: 2,
            title: 'North America View',
            url: 'https://picsum.photos/seed/sat2/400/300.jpg',
            timestamp: '2024-01-20 14:25:00',
            resolution: '1km',
            filename: 'north_america_20240120_1425.jpg'
          },
          {
            id: 3,
            title: 'Pacific Region',
            url: 'https://picsum.photos/seed/sat3/400/300.jpg',
            timestamp: '2024-01-20 14:20:00',
            resolution: '4km',
            filename: 'pacific_region_20240120_1420.jpg'
          }
        ],
        markers: [
          { lat: 40.7128, lng: -74.0060, title: 'New York', description: 'Clear skies', timestamp: '14:30 UTC' },
          { lat: 34.0522, lng: -118.2437, title: 'Los Angeles', description: 'Partly cloudy', timestamp: '14:25 UTC' },
          { lat: 51.5074, lng: -0.1278, title: 'London', description: 'Overcast', timestamp: '14:20 UTC' }
        ],
        gallery: [
          {
            id: 1,
            title: 'Hurricane Formation',
            thumbnail: 'https://picsum.photos/seed/thumb1/150/150.jpg',
            fullSize: 'https://picsum.photos/seed/full1/800/600.jpg'
          },
          {
            id: 2,
            title: 'Cloud Patterns',
            thumbnail: 'https://picsum.photos/seed/thumb2/150/150.jpg',
            fullSize: 'https://picsum.photos/seed/full2/800/600.jpg'
          }
        ]
      };
    }
  },

  // Get live video feed
  getLiveVideoFeed: async (satellite) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/satellite/live-feed`, {
        params: { satellite }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching live video feed:', error);
      // Return mock video URL
      return {
        url: 'https://www.w3schools.com/html/mov_bbb.mp4'
      };
    }
  },

  // Download satellite image
  downloadImage: async (imageId, filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/satellite/download/${imageId}`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading image:', error);
    }
  }
};

export default satelliteService;
