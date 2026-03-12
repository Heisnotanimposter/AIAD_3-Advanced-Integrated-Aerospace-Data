import React, { useState, useEffect } from 'react';
import { Layout, Menu, Button, Row, Col, Card, Statistic, Space, Tag } from 'antd';
import { 
  CloudOutlined, 
  ThunderboltOutlined, 
  LineChartOutlined, 
  InfoCircleOutlined,
  PlayCircleOutlined,
  CompassOutlined
} from '@ant-design/icons';
import { motion } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import './style.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const { Header, Content, Footer } = Layout;

const Dashboard: React.FC = () => {
    const [prediction, setPrediction] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const runPrediction = async () => {
        setLoading(true);
        try {
            const response = await axios.get('http://localhost:8000/weather/predict');
            setPrediction(response.data);
        } catch (error) {
            console.error("Prediction failed", error);
        } finally {
            setLoading(false);
        }
    };

    const chartData = {
        labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
        datasets: [
            {
                label: 'Prediction Confidence',
                data: [85, 88, 92, 90, 94, 96],
                borderColor: '#00f2fe',
                backgroundColor: 'rgba(0, 242, 254, 0.2)',
                tension: 0.4,
                fill: true,
            },
        ],
    };

    return (
        <Layout className="layout" style={{ background: 'transparent' }}>
            <Header className="nav-bar">
                <div className="logo" style={{ color: '#00f2fe', fontSize: '1.5rem', fontWeight: 800 }}>
                    AIAD <span style={{ color: '#fff', fontWeight: 300 }}>WEATHER v5</span>
                </div>
                <Menu
                    theme="dark"
                    mode="horizontal"
                    defaultSelectedKeys={['1']}
                    style={{ background: 'transparent', border: 'none', minWidth: '400px' }}
                    items={[
                        { key: '1', label: 'Dashboard', icon: <CompassOutlined /> },
                        { key: '2', label: 'Analysis', icon: <ThunderboltOutlined /> },
                        { key: '3', label: 'Satellites', icon: <CloudOutlined /> },
                        { key: '4', label: 'History', icon: <LineChartOutlined /> },
                    ]}
                />
                <Button type="primary" shape="round" icon={<PlayCircleOutlined />} style={{ background: 'linear-gradient(90deg, #00f2fe, #4facfe)', border: 'none' }}>
                    Live Feed
                </Button>
            </Header>

            <Content style={{ padding: '0 50px' }}>
                <div className="hero-section">
                    <motion.h1 
                        className="hero-title gradient-text"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        Intelligence Integrated.
                    </motion.h1>
                    <motion.p 
                        style={{ fontSize: '1.2rem', color: '#94a3b8', marginBottom: '40px' }}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 }}
                    >
                        The next evolution of satellite weather prediction powered by multimodal AI reasoning.
                    </motion.p>
                    <Button 
                        size="large" 
                        type="primary" 
                        icon={<ThunderboltOutlined />} 
                        onClick={runPrediction}
                        loading={loading}
                        style={{ height: '50px', padding: '0 40px', fontSize: '1.1rem', borderRadius: '25px', background: 'linear-gradient(90deg, #00f2fe, #4facfe)', border: 'none' }}
                    >
                        Analyze Atmospheric Data
                    </Button>
                </div>

                <Row gutter={[24, 24]}>
                    <Col span={16}>
                        <Card className="glass-panel prediction-card" bordered={false}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                                <h2 style={{ color: '#fff' }}><CloudOutlined /> AI Weather Prediction</h2>
                                <Tag color="cyan">REAL-TIME</Tag>
                            </div>
                            <div style={{ 
                                height: '400px', 
                                background: 'rgba(0,0,0,0.3)', 
                                borderRadius: '16px', 
                                display: 'flex', 
                                justifyContent: 'center', 
                                alignItems: 'center',
                                border: '1px dashed #4facfe'
                            }}>
                                {prediction ? (
                                    <img src={`http://localhost:8000${prediction.image_url}`} alt="Weather Prediction" style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: '12px' }} />
                                ) : (
                                    <p style={{ color: '#4facfe' }}>[ Click "Analyze Atmospheric Data" to predict ]</p>
                                )}
                            </div>
                            <div style={{ marginTop: '20px' }}>
                                <Space size="large">
                                    <Statistic title={<span style={{color: '#94a3b8'}}>Accuracy</span>} value={prediction?.confidence * 100 || 96.4} precision={1} suffix="%" valueStyle={{ color: '#00f2fe' }} />
                                    <Statistic title={<span style={{color: '#94a3b8'}}>Inference</span>} value={1.8} precision={2} suffix="s" valueStyle={{ color: '#fff' }} />
                                    <Statistic title={<span style={{color: '#94a3b8'}}>Data Points</span>} value={12400} valueStyle={{ color: '#fff' }} />
                                </Space>
                            </div>
                        </Card>
                    </Col>
                    
                    <Col span={8}>
                        <Card className="glass-panel" bordered={false} style={{ height: '100%' }}>
                            <h2 style={{ color: '#fff' }}><ThunderboltOutlined /> Gemini AI Insights</h2>
                            <div style={{ background: 'rgba(255,255,255,0.05)', padding: '15px', borderRadius: '12px', marginTop: '20px', minHeight: '150px' }}>
                                <p style={{ color: '#cbd5e1', lineHeight: '1.6' }}>
                                    {prediction ? prediction.analysis : "System standby. Waiting for prediction data..."}
                                </p>
                            </div>
                            <div style={{ marginTop: '30px' }}>
                                <Line data={chartData} options={{ responsive: true, scales: { y: { grid: { color: 'rgba(255,255,255,0.1)' } } } }} />
                            </div>
                            <Button block type="default" style={{ marginTop: '20px', backgroundColor: 'rgba(255,255,255,0.1)', color: '#fff', border: '1px solid #4facfe' }}>
                                Detailed AI Report
                            </Button>
                        </Card>
                    </Col>
                </Row>
            </Content>

            <Footer style={{ textAlign: 'center', background: 'transparent', color: '#64748b' }}>
                AIAD Weather Platform ©2026 Created by Seungwon Lee | Satellite Data powered by NASA & ESA
            </Footer>
        </Layout>
    );
};

export default Dashboard;
