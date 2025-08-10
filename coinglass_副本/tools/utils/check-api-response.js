#!/usr/bin/env node

import axios from 'axios';

const testEndpoint = async () => {
  try {
    const response = await axios.get('https://open-api-v4.coinglass.com/api/futures/supported-coins');
    console.log('📊 完整响应数据：');
    console.log(JSON.stringify(response.data, null, 2));
  } catch (error) {
    console.log('❌ 错误：', error.response?.data || error.message);
  }
};

testEndpoint();