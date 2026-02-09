import React from 'react';
import { Menu, Typography, Space } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  DashboardOutlined, 
  SatelliteOutlined, 
  ThunderboltOutlined, 
  BarChartOutlined 
} from '@ant-design/icons';

const { Title } = Typography;

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/satellite',
      icon: <SatelliteOutlined />,
      label: 'Satellite Imagery',
    },
    {
      key: '/predictions',
      icon: <ThunderboltOutlined />,
      label: 'Predictions',
    },
    {
      key: '/analytics',
      icon: <BarChartOutlined />,
      label: 'Analytics',
    },
  ];

  const handleMenuClick = ({ key }) => {
    navigate(key);
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
      <Title level={3} className="logo" style={{ margin: 0, color: 'white' }}>
        🛰️ SatData v4.0
      </Title>
      <Menu
        theme="dark"
        mode="horizontal"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        className="nav-menu"
        style={{ flex: 1, minWidth: 0, borderBottom: 'none' }}
      />
    </div>
  );
};

export default Navbar;
