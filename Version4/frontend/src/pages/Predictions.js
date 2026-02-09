import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Select, 
  Button, 
  Space, 
  Typography, 
  Spin,
  Image,
  Progress,
  Statistic,
  Alert,
  Slider,
  Switch,
  Tag,
  Timeline
} from 'antd';
import { 
  ThunderboltOutlined, 
  DownloadOutlined, 
  PlayCircleOutlined,
  EyeOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  SyncOutlined
} from '@ant-design/icons';
import { Line } from 'react-chartjs-2';
import { predictionService } from '../services/predictionService';

const { Title, Text } = Typography;
const { Option } = Select;

const Predictions = () => {
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [modelStats, setModelStats] = useState({});
  const [selectedTimeHorizon, setSelectedTimeHorizon] = useState(24);
  const [selectedRegion, setSelectedRegion] = useState('global');
  const [predictionType, setPredictionType] = useState('hourly');
  const [autoGenerate, setAutoGenerate] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [lossData, setLossData] = useState({});

  useEffect(() => {
    fetchPredictions();
    fetchModelStats();
  }, []);

  useEffect(() => {
    if (autoGenerate) {
      const interval = setInterval(() => {
        generatePrediction();
      }, 30000); // Generate every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoGenerate]);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const data = await predictionService.getPredictions({
        timeHorizon: selectedTimeHorizon,
        region: selectedRegion,
        type: predictionType
      });
      setPredictions(data.predictions || []);
      setLossData(data.lossData || {});
    } catch (error) {
      console.error('Error fetching predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchModelStats = async () => {
    try {
      const stats = await predictionService.getModelStats();
      setModelStats(stats);
    } catch (error) {
      console.error('Error fetching model stats:', error);
    }
  };

  const generatePrediction = async () => {
    try {
      setGenerating(true);
      setTrainingProgress(0);
      
      const progressInterval = setInterval(() => {
        setTrainingProgress(prev => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + Math.random() * 10;
        });
      }, 500);

      const data = await predictionService.generatePrediction({
        timeHorizon: selectedTimeHorizon,
        region: selectedRegion,
        type: predictionType
      });

      clearInterval(progressInterval);
      setTrainingProgress(100);
      
      setTimeout(() => {
        setPredictions(prev => [data.prediction, ...prev.slice(0, 11)]);
        setTrainingProgress(0);
        setGenerating(false);
      }, 500);

    } catch (error) {
      console.error('Error generating prediction:', error);
      setGenerating(false);
      setTrainingProgress(0);
    }
  };

  const handleDownload = async (imageUrl, filename) => {
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading image:', error);
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
        text: 'Training Loss Progress',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2} className="page-title">AI-Powered Weather Predictions</Title>
      
      <Alert
        message="DCGAN Model Active"
        description="Advanced Deep Convolutional Generative Adversarial Network is generating weather predictions based on satellite data."
        type="success"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Model Accuracy"
              value={modelStats.accuracy || 94.5}
              suffix="%"
              prefix={<EyeOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Predictions"
              value={modelStats.totalPredictions || 1247}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Avg Generation Time"
              value={modelStats.avgGenTime || 2.3}
              suffix="sec"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      <Card style={{ margin: '24px 0' }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={6}>
            <Text strong>Time Horizon:</Text>
            <Slider
              min={1}
              max={168}
              value={selectedTimeHorizon}
              onChange={setSelectedTimeHorizon}
              marks={{
                24: '24h',
                72: '3d',
                168: '7d'
              }}
              style={{ marginTop: 8 }}
            />
            <Text type="secondary">{selectedTimeHorizon} hours</Text>
          </Col>
          <Col xs={24} sm={6}>
            <Text strong>Region:</Text>
            <Select
              value={selectedRegion}
              onChange={setSelectedRegion}
              style={{ width: '100%', marginTop: 4 }}
            >
              <Option value="global">Global</Option>
              <Option value="north-america">North America</Option>
              <Option value="europe">Europe</Option>
              <Option value="asia">Asia</Option>
            </Select>
          </Col>
          <Col xs={24} sm={6}>
            <Text strong>Prediction Type:</Text>
            <Select
              value={predictionType}
              onChange={setPredictionType}
              style={{ width: '100%', marginTop: 4 }}
            >
              <Option value="hourly">Hourly</Option>
              <Option value="daily">Daily</Option>
              <Option value="weekly">Weekly</Option>
            </Select>
          </Col>
          <Col xs={24} sm={6}>
            <Space direction="vertical">
              <Space>
                <Button 
                  type="primary" 
                  icon={<ThunderboltOutlined />} 
                  onClick={generatePrediction}
                  loading={generating}
                  size="large"
                >
                  Generate Prediction
                </Button>
                <Switch
                  checked={autoGenerate}
                  onChange={setAutoGenerate}
                  checkedChildren="Auto"
                  unCheckedChildren="Manual"
                />
              </Space>
              {generating && (
                <Progress percent={trainingProgress} status="active" />
              )}
            </Space>
          </Col>
        </Row>
      </Card>

      {generating && (
        <Alert
          message="Generating Prediction"
          description="DCGAN model is processing satellite data and generating weather predictions..."
          type="info"
          showIcon
          icon={<SyncOutlined spin />}
          style={{ marginBottom: 24 }}
        />
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title="Generated Predictions" className="dashboard-card">
            {loading ? (
              <div className="loading-spinner">
                <Spin size="large" />
              </div>
            ) : (
              <div className="prediction-grid">
                {predictions.map((prediction, index) => (
                  <div key={index} className="prediction-item">
                    <Image
                      src={prediction.imageUrl}
                      alt={`Prediction ${index + 1}`}
                      className="prediction-image"
                      preview={{
                        src: prediction.highResUrl,
                      }}
                    />
                    <div>
                      <Text strong>{prediction.timeLabel}</Text>
                      <br />
                      <Text type="secondary">{prediction.timestamp}</Text>
                      <br />
                      <Tag color={prediction.confidence > 0.8 ? 'green' : 'orange'}>
                        {Math.round(prediction.confidence * 100)}% confidence
                      </Tag>
                      <br />
                      <Space style={{ marginTop: 8 }}>
                        <Button 
                          size="small" 
                          icon={<DownloadOutlined />}
                          onClick={() => handleDownload(prediction.imageUrl, `prediction_${index + 1}.png`)}
                        />
                        <Button 
                          size="small" 
                          icon={<EyeOutlined />}
                          onClick={() => fetchPredictions()}
                        />
                      </Space>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="Training Progress" className="dashboard-card">
            <div className="chart-container">
              {lossData.labels ? (
                <Line data={lossData} options={chartOptions} />
              ) : (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <Text type="secondary">No training data available</Text>
                </div>
              )}
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Recent Generation History" className="dashboard-card">
            <Timeline>
              {predictions.slice(0, 6).map((prediction, index) => (
                <Timeline.Item
                  key={index}
                  color={prediction.confidence > 0.8 ? 'green' : 'orange'}
                  dot={prediction.confidence > 0.8 ? <CheckCircleOutlined /> : <ClockCircleOutlined />}
                >
                  <Text strong>{prediction.timeLabel}</Text>
                  <br />
                  <Text type="secondary">{prediction.timestamp}</Text>
                  <br />
                  <Tag color={prediction.confidence > 0.8 ? 'green' : 'orange'}>
                    Confidence: {Math.round(prediction.confidence * 100)}%
                  </Tag>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Model Information" className="dashboard-card">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>Architecture:</Text>
                <br />
                <Text>Deep Convolutional GAN (DCGAN)</Text>
              </div>
              <div>
                <Text strong>Input Shape:</Text>
                <br />
                <Text>180x180x3 (RGB satellite images)</Text>
              </div>
              <div>
                <Text strong>Training Data:</Text>
                <br />
                <Text>60,000+ satellite frames</Text>
              </div>
              <div>
                <Text strong>Model Version:</Text>
                <br />
                <Tag color="blue">v4.0</Tag>
              </div>
              <div>
                <Text strong>Last Training:</Text>
                <br />
                <Text>2 hours ago</Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Predictions;
