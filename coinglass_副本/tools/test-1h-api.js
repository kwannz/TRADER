#!/usr/bin/env node

import axios from 'axios';

async function test1HData() {
  const url = 'https://www.okx.com/api/v5/market/candles';
  
  console.log('Testing OKX 1H data availability for BTC-USDT...\n');
  
  try {
    // Test 1: Get recent 1H candles
    console.log('Test 1: Getting recent 1H candles (no before parameter)');
    const response1 = await axios.get(url, {
      params: {
        instId: 'BTC-USDT',
        bar: '1H',
        limit: 5
      }
    });
    
    console.log('Response:', response1.data);
    
    if (response1.data.data && response1.data.data.length > 0) {
      console.log('\nFirst candle:');
      const firstCandle = response1.data.data[0];
      console.log(`  Timestamp: ${firstCandle[0]} (${new Date(parseInt(firstCandle[0])).toISOString()})`);
      console.log(`  OHLCV: ${firstCandle.slice(1).join(', ')}`);
      
      console.log('\nLast candle:');
      const lastCandle = response1.data.data[response1.data.data.length - 1];
      console.log(`  Timestamp: ${lastCandle[0]} (${new Date(parseInt(lastCandle[0])).toISOString()})`);
      console.log(`  OHLCV: ${lastCandle.slice(1).join(', ')}`);
    }
    
    // Test 2: Get historical 1H candles
    console.log('\n\nTest 2: Getting historical 1H candles (with before parameter)');
    const beforeTime = Date.now() - (24 * 60 * 60 * 1000); // 24 hours ago
    const response2 = await axios.get('https://www.okx.com/api/v5/market/history-candles', {
      params: {
        instId: 'BTC-USDT',
        bar: '1H',
        before: beforeTime,
        limit: 5
      }
    });
    
    console.log('Response:', response2.data);
    
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

test1HData();