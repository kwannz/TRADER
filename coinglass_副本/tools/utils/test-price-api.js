#!/usr/bin/env node

import axios from 'axios';

const API_BASE = 'https://open-api-v4.coinglass.com';
const API_KEY = '51e89d90bf31473384e7e6c61b75afe7';

async function testPriceAPI() {
  console.log('测试价格历史API...\n');
  
  // 测试不同的参数组合
  const tests = [
    {
      name: '测试1: 只有symbol参数',
      params: { symbol: 'BTC' }
    },
    {
      name: '测试2: symbol + interval',
      params: { symbol: 'BTC', interval: '1d' }
    },
    {
      name: '测试3: symbol + interval + limit',
      params: { symbol: 'BTC', interval: '1d', limit: 100 }
    },
    {
      name: '测试4: 不带时间戳的请求',
      params: { symbol: 'BTC', interval: '1h', limit: 24 }
    },
    {
      name: '测试5: 使用exchange参数',
      params: { symbol: 'BTC', exchange: 'Binance', interval: '1d' }
    }
  ];
  
  for (const test of tests) {
    console.log(`\n${test.name}`);
    console.log('参数:', test.params);
    
    try {
      const response = await axios.get(`${API_BASE}/api/futures/price/history`, {
        params: test.params,
        headers: {
          'CG-API-KEY': API_KEY,
          'accept': 'application/json'
        }
      });
      
      console.log('✅ 成功');
      console.log('响应结构:', Object.keys(response.data));
      console.log('code:', response.data.code);
      console.log('msg:', response.data.msg);
      console.log('data长度:', response.data.data?.length || 0);
      
      if (response.data.data && response.data.data.length > 0) {
        console.log('第一个数据点:', response.data.data[0]);
        console.log('最后一个数据点:', response.data.data[response.data.data.length - 1]);
      }
    } catch (error) {
      console.log('❌ 失败:', error.message);
      if (error.response?.data) {
        console.log('错误详情:', error.response.data);
      }
    }
    
    // 延迟避免速率限制
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  // 测试现货价格历史
  console.log('\n\n测试现货价格历史API...');
  try {
    const spotResponse = await axios.get(`${API_BASE}/api/spot/price/history`, {
      params: { symbol: 'BTC', interval: '1d', limit: 10 },
      headers: {
        'CG-API-KEY': API_KEY,
        'accept': 'application/json'
      }
    });
    
    console.log('✅ 现货API成功');
    console.log('数据长度:', spotResponse.data.data?.length || 0);
    if (spotResponse.data.data && spotResponse.data.data.length > 0) {
      console.log('示例数据:', spotResponse.data.data[0]);
    }
  } catch (error) {
    console.log('❌ 现货API失败:', error.message);
  }
}

testPriceAPI().catch(console.error);