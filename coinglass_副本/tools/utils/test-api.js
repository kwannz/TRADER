#!/usr/bin/env node

// 测试CoinGlass API连接
import axios from 'axios';

const testAPI = async () => {
  const baseURL = 'https://open-api-v4.coinglass.com';
  
  console.log('🔍 测试CoinGlass API连接...');
  
  try {
    // 测试基础端点
    // 尝试多种API密钥传递方式
    const apiKey = '51e89d90bf31473384e7e6c61b75afe7';
    
    // 方式1: 作为查询参数
    let response;
    try {
      response = await axios.get(`${baseURL}/api/futures/supported-coins?apiKey=${apiKey}`);
    } catch (err1) {
      console.log('方式1失败:', err1.response?.data?.msg || err1.message);
      
      // 方式2: 作为header
      try {
        response = await axios.get(`${baseURL}/api/futures/supported-coins`, {
          headers: { 'X-API-KEY': apiKey }
        });
      } catch (err2) {
        console.log('方式2失败:', err2.response?.data?.msg || err2.message);
        
        // 方式3: 作为Authorization header
        try {
          response = await axios.get(`${baseURL}/api/futures/supported-coins`, {
            headers: { 'Authorization': `Bearer ${apiKey}` }
          });
        } catch (err3) {
          console.log('方式3失败:', err3.response?.data?.msg || err3.message);
          throw err3;
        }
      }
    }
    console.log('✅ API连接成功');
    console.log('📊 响应数据：', JSON.stringify(response.data, null, 2).substring(0, 500) + '...');
  } catch (error) {
    console.log('❌ API连接失败：', error.message);
    console.log('📍 URL：', error.config?.url);
    console.log('🔢 状态码：', error.response?.status);
  }
  
  // 测试其他可能的端点
  const endpoints = [
    '/api/futures/supported-coins',
    '/api/spot/supported-coins',
    '/api/etf/bitcoin-etf-list',
    '/api/indicator/fear-greed-index'
  ];
  
  for (const endpoint of endpoints) {
    try {
      console.log(`\n🔍 测试端点：${endpoint}`);
      const response = await axios.get(`${baseURL}${endpoint}`, {
        params: { apiKey: '51e89d90bf31473384e7e6c61b75afe7' }
      });
      console.log(`✅ ${endpoint} - 成功`);
    } catch (error) {
      console.log(`❌ ${endpoint} - 失败 (${error.response?.status || error.message})`);
    }
  }
};

testAPI();