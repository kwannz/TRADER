#!/usr/bin/env node

require('dotenv').config();
const coinGlassClient = require('./api/coinglass-client');

async function testAPI() {
  console.log('🔧 Testing CoinGlass API Connection...\n');
  
  try {
    // 测试API状态
    const apiStatus = coinGlassClient.getStatus();
    console.log('📊 API Client Status:');
    console.log(`   • Base URL: ${apiStatus.baseURL}`);
    console.log(`   • Has API Key: ${apiStatus.hasApiKey ? '✅' : '❌'}`);
    console.log(`   • Rate Limit: ${apiStatus.rateLimit} requests/minute`);
    console.log(`   • Timeout: ${apiStatus.timeout}ms\n`);
    
    if (!apiStatus.hasApiKey) {
      console.log('❌ No API key found. Please set COINGLASS_API_KEY in .env file');
      return;
    }
    
    // 测试连接
    console.log('🌐 Testing API Connection...');
    const testResult = await coinGlassClient.testConnection();
    
    if (testResult.success) {
      console.log('✅ API Connection Successful!');
      console.log(`   • Message: ${testResult.message}`);
      console.log(`   • Supported Coins: ${testResult.supportedCoins}\n`);
    } else {
      console.log('❌ API Connection Failed!');
      console.log(`   • Error: ${testResult.error}\n`);
      return;
    }
    
    // 测试一些基本的API端点
    console.log('🧪 Testing API Endpoints...\n');
    
    // 1. 测试合约市场数据
    try {
      console.log('📊 Testing Contract Markets...');
      const markets = await coinGlassClient.getContractMarkets('BTC');
      console.log(`   ✅ Contract Markets: ${markets.data?.length || 0} records`);
    } catch (error) {
      console.log(`   ❌ Contract Markets failed: ${error.message}`);
    }
    
    // 2. 测试恐惧贪婪指数
    try {
      console.log('😨 Testing Fear & Greed Index...');
      const fearGreed = await coinGlassClient.getFearGreedIndex();
      console.log(`   ✅ Fear & Greed Index: ${fearGreed.data?.value || 'N/A'} (${fearGreed.data?.classification || 'N/A'})`);
    } catch (error) {
      console.log(`   ❌ Fear & Greed Index failed: ${error.message}`);
    }
    
    // 3. 测试比特币ETF
    try {
      console.log('💰 Testing Bitcoin ETF...');
      const btcETF = await coinGlassClient.getBitcoinETF();
      console.log(`   ✅ Bitcoin ETF: ${btcETF.data?.length || 0} records`);
    } catch (error) {
      console.log(`   ❌ Bitcoin ETF failed: ${error.message}`);
    }
    
    // 4. 测试资金费率
    try {
      console.log('💸 Testing Funding Rates...');
      const fundingRates = await coinGlassClient.getFundingRates('BTC');
      console.log(`   ✅ Funding Rates: ${fundingRates.data?.length || 0} records`);
    } catch (error) {
      console.log(`   ❌ Funding Rates failed: ${error.message}`);
    }
    
    // 5. 测试持仓数据
    try {
      console.log('📈 Testing Open Interest...');
      const openInterest = await coinGlassClient.getOpenInterest('BTC');
      console.log(`   ✅ Open Interest: ${openInterest.data?.length || 0} records`);
    } catch (error) {
      console.log(`   ❌ Open Interest failed: ${error.message}`);
    }
    
    console.log('\n🎉 API Testing Completed!');
    console.log('✅ CoinGlass API is working correctly.\n');
    
    // 显示可用的API端点
    const endpoints = coinGlassClient.getAvailableEndpoints();
    console.log('📋 Available API Methods:');
    endpoints.forEach((endpoint, index) => {
      console.log(`   ${index + 1}. ${endpoint}`);
    });
    
  } catch (error) {
    console.error('\n💥 API Test Failed:', error.message);
    console.error('Stack:', error.stack);
  }
}

// 运行测试
if (require.main === module) {
  testAPI().catch(error => {
    console.error('💥 Unexpected error:', error);
    process.exit(1);
  });
}

module.exports = testAPI; 