import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Select, 
  DatePicker, 
  Button, 
  Space, 
  Typography, 
  Spin,
  Table,
  Tag,
  Progress
} from 'antd';
import { 
  BarChartOutlined, 
  LineChartOutlined, 
  PieChartOutlined,
  DownloadOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { Line, Bar, Pie } from 'react-chartjs-2';
import { analyticsService } from '../services/analyticsService';

const { Title, Text } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;

const Analytics = () => {
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState('7d');
  const [selectedMetric, setSelectedMetric] = useState('all');
  const [analyticsData, setAnalyticsData] = useState({});
  const [performanceData, setPerformanceData] = useState([]);
  const [regionalData, setRegionalData] = useState([]);

  useEffect(() => {
    fetchAnalyticsData();
  }, [timeRange, selectedMetric]);

  const fetchAnalyticsData = async () => {
    try {
      setLoading(true);
      const data = await analyticsService.getAnalytics({
        timeRange,
        metric: selectedMetric
      });
      
      setAnalyticsData(data);
      setPerformanceData(data.performance || []);
      setRegionalData(data.regional || []);
    } catch (error) {
      console.error('Error fetching analytics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Prediction Accuracy Over Time',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
      },
    },
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Regional Performance Comparison',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const pieChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
      },
      title: {
        display: true,
        text: 'Prediction Types Distribution',
      },
    },
  };

  const performanceColumns = [
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
      sorter: (a, b) => new Date(a.date) - new Date(b.date),
    },
    {
      title: 'Predictions',
      dataIndex: 'predictions',
      key: 'predictions',
      sorter: (a, b) => a.predictions - b.predictions,
    },
    {
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy) => (
        <Progress 
          percent={accuracy} 
          size="small" 
          status={accuracy > 90 ? 'success' : accuracy > 75 ? 'normal' : 'exception'}
        />
      ),
      sorter: (a, b) => a.accuracy - b.accuracy,
    },
    {
      title: 'Avg Loss',
      dataIndex: 'avgLoss',
      key: 'avgLoss',
      render: (loss) => loss.toFixed(4),
      sorter: (a, b) => a.avgLoss - b.avgLoss,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'excellent' ? 'green' : status === 'good' ? 'blue' : 'orange'}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
  ];

  const regionalColumns = [
    {
      title: 'Region',
      dataIndex: 'region',
      key: 'region',
    },
    {
      title: 'Total Predictions',
      dataIndex: 'totalPredictions',
      key: 'totalPredictions',
      sorter: (a, b) => a.totalPredictions - b.totalPredictions,
    },
    {
      title: 'Success Rate',
      dataIndex: 'successRate',
      key: 'successRate',
      render: (rate) => `${rate}%`,
      sorter: (a, b) => a.successRate - b.successRate,
    },
    {
      title: 'Avg Confidence',
      dataIndex: 'avgConfidence',
      key: 'avgConfidence',
      render: (confidence) => `${confidence}%`,
      sorter: (a, b) => a.avgConfidence - b.avgConfidence,
    },
    {
      title: 'Performance',
      dataIndex: 'performance',
      key: 'performance',
      render: (performance) => (
        <Progress 
          percent={performance} 
          size="small" 
          status={performance > 85 ? 'success' : performance > 70 ? 'normal' : 'exception'}
        />
      ),
      sorter: (a, b) => a.performance - b.performance,
    },
  ];

  const exportData = () => {
    // Export functionality
    const csvContent = [
      ['Date', 'Predictions', 'Accuracy', 'Avg Loss', 'Status'],
      ...performanceData.map(row => [
        row.date,
        row.predictions,
        row.accuracy,
        row.avgLoss,
        row.status
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analytics_${timeRange}_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
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
      <Title level={2} className="page-title">Analytics & Performance Metrics</Title>
      
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={6}>
            <Text strong>Time Range:</Text>
            <Select
              value={timeRange}
              onChange={setTimeRange}
              style={{ width: '100%', marginTop: 4 }}
            >
              <Option value="24h">Last 24 Hours</Option>
              <Option value="7d">Last 7 Days</Option>
              <Option value="30d">Last 30 Days</Option>
              <Option value="90d">Last 90 Days</Option>
            </Select>
          </Col>
          <Col xs={24} sm={6}>
            <Text strong>Metric:</Text>
            <Select
              value={selectedMetric}
              onChange={setSelectedMetric}
              style={{ width: '100%', marginTop: 4 }}
            >
              <Option value="all">All Metrics</Option>
              <Option value="accuracy">Accuracy</Option>
              <Option value="performance">Performance</Option>
              <Option value="usage">Usage</Option>
            </Select>
          </Col>
          <Col xs={24} sm={6}>
            <Text strong>Custom Range:</Text>
            <RangePicker
              style={{ width: '100%', marginTop: 4 }}
              onChange={(dates) => {
                if (dates) {
                  // Handle custom date range
                }
              }}
            />
          </Col>
          <Col xs={24} sm={6}>
            <Space>
              <Button 
                icon={<ReloadOutlined />} 
                onClick={fetchAnalyticsData}
                loading={loading}
              >
                Refresh
              </Button>
              <Button 
                type="primary" 
                icon={<DownloadOutlined />}
                onClick={exportData}
              >
                Export
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <LineChartOutlined />
                <span>Accuracy Trends</span>
              </Space>
            } 
            className="dashboard-card"
          >
            <div className="chart-container">
              {analyticsData.accuracyChart ? (
                <Line data={analyticsData.accuracyChart} options={lineChartOptions} />
              ) : (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <Text type="secondary">No accuracy data available</Text>
                </div>
              )}
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <BarChartOutlined />
                <span>Regional Performance</span>
              </Space>
            } 
            className="dashboard-card"
          >
            <div className="chart-container">
              {analyticsData.regionalChart ? (
                <Bar data={analyticsData.regionalChart} options={barChartOptions} />
              ) : (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <Text type="secondary">No regional data available</Text>
                </div>
              )}
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} lg={8}>
          <Card 
            title={
              <Space>
                <PieChartOutlined />
                <span>Prediction Types</span>
              </Space>
            } 
            className="dashboard-card"
          >
            <div className="chart-container">
              {analyticsData.predictionTypes ? (
                <Pie data={analyticsData.predictionTypes} options={pieChartOptions} />
              ) : (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <Text type="secondary">No prediction type data available</Text>
                </div>
              )}
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={16}>
          <Card title="Performance Metrics" className="dashboard-card">
            <Table
              columns={performanceColumns}
              dataSource={performanceData}
              rowKey="date"
              pagination={{ pageSize: 10 }}
              scroll={{ x: true }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24}>
          <Card title="Regional Analysis" className="dashboard-card">
            <Table
              columns={regionalColumns}
              dataSource={regionalData}
              rowKey="region"
              pagination={{ pageSize: 10 }}
              scroll={{ x: true }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Analytics;
