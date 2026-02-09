import React, { useState, useEffect } from 'react';
import { 
  Row, 
  Col, 
  Card, 
  Statistic, 
  Progress, 
  List, 
  Typography, 
  Space,
  Spin,
  Alert
} from 'antd';
import { 
  CloudOutlined, 
  ThunderboltOutlined, 
  EyeOutlined, 
  ClockCircleOutlined 
} from '@ant-design/icons';
import { Line } from 'react-chartjs-2';
import { satelliteService } from '../services/satelliteService';

const { Title, Text } = Typography;

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({});
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [chartData, setChartData] = useState({});

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [statsData, predictionsData, chartResponse] = await Promise.all([
        satelliteService.getStats(),
        satelliteService.getRecentPredictions(),
        satelliteService.getChartData()
      ]);
      
      setStats(statsData);
      setRecentPredictions(predictionsData);
      setChartData(chartResponse);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Over Time',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  if (loading) {
    return (
      <div className="loading-spinner">
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2} className="page-title">Satellite Data Analysis Dashboard</Title>
      
      <Alert
        message="System Status: Active"
        description="All satellite data processing systems are operational. Real-time predictions are running."
        type="success"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={6}>
          <Card className="stat-card">
            <Statistic
              title="Active Satellites"
              value={stats.activeSatellites || 12}
              prefix={<SatelliteOutlined />}
              valueStyle={{ color: '#fff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card className="stat-card">
            <Statistic
              title="Predictions Today"
              value={stats.predictionsToday || 248}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#fff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card className="stat-card">
            <Statistic
              title="Data Processed"
              value={stats.dataProcessed || 1.2}
              suffix="TB"
              prefix={<CloudOutlined />}
              valueStyle={{ color: '#fff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card className="stat-card">
            <Statistic
              title="Model Accuracy"
              value={stats.modelAccuracy || 94.5}
              suffix="%"
              prefix={<EyeOutlined />}
              valueStyle={{ color: '#fff' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} lg={16}>
          <Card title="Performance Metrics" className="dashboard-card">
            <div className="chart-container">
              {chartData.labels ? (
                <Line data={chartData} options={chartOptions} />
              ) : (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <Text type="secondary">No chart data available</Text>
                </div>
              )}
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="System Health" className="dashboard-card">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>CPU Usage</Text>
                <Progress percent={65} status="active" />
              </div>
              <div>
                <Text>Memory Usage</Text>
                <Progress percent={78} status="active" />
              </div>
              <div>
                <Text>GPU Usage</Text>
                <Progress percent={45} status="active" />
              </div>
              <div>
                <Text>Storage</Text>
                <Progress percent={82} status="active" />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24}>
          <Card title="Recent Predictions" className="dashboard-card">
            <List
              itemLayout="horizontal"
              dataSource={recentPredictions.slice(0, 5)}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={<ClockCircleOutlined style={{ fontSize: '20px', color: '#1890ff' }} />}
                    title={`Prediction for ${item.location}`}
                    description={`${item.timestamp} - Accuracy: ${item.accuracy}%`}
                  />
                  <div>
                    <span className={`status-badge ${item.status === 'completed' ? 'status-active' : 'status-processing'}`}>
                      {item.status}
                    </span>
                  </div>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
