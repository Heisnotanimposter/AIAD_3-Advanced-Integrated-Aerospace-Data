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
  Image,
  Tabs,
  Alert
} from 'antd';
import { 
  SatelliteOutlined, 
  DownloadOutlined, 
  PlayCircleOutlined,
  ReloadOutlined 
} from '@ant-design/icons';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import ReactPlayer from 'react-player';
import { satelliteService } from '../services/satelliteService';
import 'leaflet/dist/leaflet.css';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { RangePicker } = DatePicker;

// Fix for default markers in Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const SatelliteImagery = () => {
  const [loading, setLoading] = useState(false);
  const [satelliteData, setSatelliteData] = useState([]);
  const [selectedSatellite, setSelectedSatellite] = useState('GOES-16');
  const [selectedRegion, setSelectedRegion] = useState('global');
  const [dateRange, setDateRange] = useState(null);
  const [mapCenter, setMapCenter] = useState([39.8283, -98.5795]); // Center of USA
  const [markers, setMarkers] = useState([]);
  const [liveVideoUrl, setLiveVideoUrl] = useState('');
  const [imageGallery, setImageGallery] = useState([]);

  useEffect(() => {
    fetchSatelliteData();
    fetchLiveVideo();
  }, [selectedSatellite, selectedRegion]);

  const fetchSatelliteData = async () => {
    try {
      setLoading(true);
      const data = await satelliteService.getSatelliteImagery({
        satellite: selectedSatellite,
        region: selectedRegion,
        dateRange
      });
      setSatelliteData(data.images || []);
      setMarkers(data.markers || []);
      setImageGallery(data.gallery || []);
    } catch (error) {
      console.error('Error fetching satellite data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchLiveVideo = async () => {
    try {
      const videoData = await satelliteService.getLiveVideoFeed(selectedSatellite);
      setLiveVideoUrl(videoData.url);
    } catch (error) {
      console.error('Error fetching live video:', error);
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

  const handleRefresh = () => {
    fetchSatelliteData();
    fetchLiveVideo();
  };

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2} className="page-title">Satellite Imagery & Live Feeds</Title>
      
      <Alert
        message="Real-time Satellite Data"
        description="Access live satellite imagery and video feeds from multiple satellites worldwide."
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Card style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={6}>
            <Text strong>Satellite:</Text>
            <Select
              value={selectedSatellite}
              onChange={setSelectedSatellite}
              style={{ width: '100%', marginTop: 4 }}
            >
              <Option value="GOES-16">GOES-16 (Americas)</Option>
              <Option value="GOES-17">GOES-17 (Pacific)</Option>
              <Option value="HIMAWARI-8">Himawari-8 (Asia)</Option>
              <Option value="METEOSAT">Meteosat (Europe/Africa)</Option>
            </Select>
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
              <Option value="south-america">South America</Option>
              <Option value="europe">Europe</Option>
              <Option value="asia">Asia</Option>
              <Option value="africa">Africa</Option>
            </Select>
          </Col>
          <Col xs={24} sm={8}>
            <Text strong>Date Range:</Text>
            <RangePicker
              value={dateRange}
              onChange={setDateRange}
              style={{ width: '100%', marginTop: 4 }}
            />
          </Col>
          <Col xs={24} sm={4}>
            <Space>
              <Button 
                type="primary" 
                icon={<ReloadOutlined />} 
                onClick={handleRefresh}
                loading={loading}
              >
                Refresh
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Live Satellite Map" className="dashboard-card">
            <div className="satellite-map">
              <MapContainer
                center={mapCenter}
                zoom={4}
                style={{ height: '500px', width: '100%' }}
              >
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                {markers.map((marker, index) => (
                  <Marker key={index} position={[marker.lat, marker.lng]}>
                    <Popup>
                      <div>
                        <strong>{marker.title}</strong><br />
                        {marker.description}<br />
                        <small>{marker.timestamp}</small>
                      </div>
                    </Popup>
                  </Marker>
                ))}
              </MapContainer>
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Live Video Feed" className="dashboard-card">
            <div className="video-container">
              {liveVideoUrl ? (
                <ReactPlayer
                  url={liveVideoUrl}
                  width="100%"
                  height="400px"
                  playing={true}
                  controls={true}
                  light={true}
                />
              ) : (
                <div style={{ textAlign: 'center', padding: '100px' }}>
                  <Spin size="large" />
                  <Text style={{ display: 'block', marginTop: 16 }}>Loading live feed...</Text>
                </div>
              )}
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24}>
          <Card title="Recent Satellite Images" className="dashboard-card">
            <Tabs defaultActiveKey="1">
              <TabPane tab="Latest Images" key="1">
                {loading ? (
                  <div className="loading-spinner">
                    <Spin size="large" />
                  </div>
                ) : (
                  <Row gutter={[16, 16]}>
                    {satelliteData.map((image, index) => (
                      <Col xs={24} sm={12} md={8} lg={6} key={index}>
                        <Card
                          hoverable
                          cover={
                            <Image
                              src={image.url}
                              alt={image.title}
                              style={{ height: '200px', objectFit: 'cover' }}
                            />
                          }
                          actions={[
                            <DownloadOutlined 
                              key="download" 
                              onClick={() => handleDownload(image.url, image.filename)}
                            />,
                            <SatelliteOutlined key="satellite" />
                          ]}
                        >
                          <Card.Meta
                            title={image.title}
                            description={
                              <Space direction="vertical" size="small">
                                <Text type="secondary">{image.timestamp}</Text>
                                <Text type="secondary">Resolution: {image.resolution}</Text>
                              </Space>
                            }
                          />
                        </Card>
                      </Col>
                    ))}
                  </Row>
                )}
              </TabPane>
              <TabPane tab="Image Gallery" key="2">
                <Row gutter={[16, 16]}>
                  {imageGallery.map((image, index) => (
                    <Col xs={24} sm={12} md={8} lg={4} key={index}>
                      <Image
                        src={image.thumbnail}
                        alt={image.title}
                        style={{ width: '100%', cursor: 'pointer' }}
                        preview={{
                          src: image.fullSize,
                        }}
                      />
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {image.title}
                      </Text>
                    </Col>
                  ))}
                </Row>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SatelliteImagery;
