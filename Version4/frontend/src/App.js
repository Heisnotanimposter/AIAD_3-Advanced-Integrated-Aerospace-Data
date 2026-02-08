import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout, theme } from 'antd';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import SatelliteImagery from './pages/SatelliteImagery';
import Predictions from './pages/Predictions';
import Analytics from './pages/Analytics';
import 'antd/dist/reset.css';
import './App.css';

const { Header, Content } = Layout;

function App() {
  const {
    token: { colorBgContainer },
  } = theme.useToken();

  return (
    <Router>
      <Layout className="min-h-screen">
        <Header className="header">
          <Navbar />
        </Header>
        <Content className="content">
          <div
            style={{
              background: colorBgContainer,
              minHeight: 'calc(100vh - 64px)',
            }}
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/satellite" element={<SatelliteImagery />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/analytics" element={<Analytics />} />
            </Routes>
          </div>
        </Content>
      </Layout>
    </Router>
  );
}

export default App;
