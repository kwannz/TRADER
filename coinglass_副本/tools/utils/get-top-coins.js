#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';

class TopCoinsGetter {
  constructor() {
    // 使用CoinGecko免费API获取市值排名
    this.COINGECKO_API = 'https://api.coingecko.com/api/v3';
    
    // 加载OKX支持的代币列表
    this.okxSupportedCoins = null;
  }

  async loadOKXCoins() {
    try {
      const data = await fs.readFile('./historical/storage/raw/spot-supported-coins.json', 'utf8');
      const json = JSON.parse(data);
      this.okxSupportedCoins = new Set(json.data);
      console.log(`✅ 加载了 ${this.okxSupportedCoins.size} 个OKX支持的代币`);
    } catch (error) {
      console.error('❌ 加载OKX代币列表失败:', error.message);
      return false;
    }
    return true;
  }

  async getTopCoinsByMarketCap(limit = 200) {
    try {
      console.log('📊 从CoinGecko获取市值排名前200的代币...');
      
      const response = await axios.get(`${this.COINGECKO_API}/coins/markets`, {
        params: {
          vs_currency: 'usd',
          order: 'market_cap_desc',
          per_page: limit,
          page: 1,
          sparkline: false
        }
      });

      return response.data;
    } catch (error) {
      console.error('❌ 获取市值排名失败:', error.message);
      
      // 备用方案：使用预定义的主流代币列表
      console.log('⚠️  使用备用主流代币列表...');
      return this.getFallbackTopCoins();
    }
  }

  getFallbackTopCoins() {
    // 备用的主流代币列表（按一般市值排序）
    return [
      { symbol: 'BTC', name: 'Bitcoin' },
      { symbol: 'ETH', name: 'Ethereum' },
      { symbol: 'USDT', name: 'Tether' },
      { symbol: 'BNB', name: 'Binance Coin' },
      { symbol: 'XRP', name: 'Ripple' },
      { symbol: 'USDC', name: 'USD Coin' },
      { symbol: 'SOL', name: 'Solana' },
      { symbol: 'ADA', name: 'Cardano' },
      { symbol: 'DOGE', name: 'Dogecoin' },
      { symbol: 'AVAX', name: 'Avalanche' },
      { symbol: 'TRX', name: 'TRON' },
      { symbol: 'DOT', name: 'Polkadot' },
      { symbol: 'MATIC', name: 'Polygon' },
      { symbol: 'LINK', name: 'Chainlink' },
      { symbol: 'TON', name: 'Toncoin' },
      { symbol: 'ICP', name: 'Internet Computer' },
      { symbol: 'SHIB', name: 'Shiba Inu' },
      { symbol: 'DAI', name: 'Dai' },
      { symbol: 'LTC', name: 'Litecoin' },
      { symbol: 'BCH', name: 'Bitcoin Cash' },
      { symbol: 'UNI', name: 'Uniswap' },
      { symbol: 'ATOM', name: 'Cosmos' },
      { symbol: 'OKB', name: 'OKB' },
      { symbol: 'ETC', name: 'Ethereum Classic' },
      { symbol: 'XMR', name: 'Monero' },
      { symbol: 'XLM', name: 'Stellar' },
      { symbol: 'FIL', name: 'Filecoin' },
      { symbol: 'APT', name: 'Aptos' },
      { symbol: 'ARB', name: 'Arbitrum' },
      { symbol: 'VET', name: 'VeChain' },
      { symbol: 'MKR', name: 'Maker' },
      { symbol: 'NEAR', name: 'NEAR Protocol' },
      { symbol: 'OP', name: 'Optimism' },
      { symbol: 'AAVE', name: 'Aave' },
      { symbol: 'INJ', name: 'Injective' },
      { symbol: 'GRT', name: 'The Graph' },
      { symbol: 'ALGO', name: 'Algorand' },
      { symbol: 'SAND', name: 'The Sandbox' },
      { symbol: 'MANA', name: 'Decentraland' },
      { symbol: 'AXS', name: 'Axie Infinity' },
      { symbol: 'EGLD', name: 'MultiversX' },
      { symbol: 'THETA', name: 'Theta' },
      { symbol: 'FTM', name: 'Fantom' },
      { symbol: 'XTZ', name: 'Tezos' },
      { symbol: 'FLOW', name: 'Flow' },
      { symbol: 'CHZ', name: 'Chiliz' },
      { symbol: 'GALA', name: 'Gala' },
      { symbol: 'CRV', name: 'Curve' },
      { symbol: 'KAVA', name: 'Kava' },
      { symbol: 'RNDR', name: 'Render' }
    ].map((coin, index) => ({
      ...coin,
      symbol: coin.symbol.toUpperCase(),
      market_cap_rank: index + 1
    }));
  }

  async filterOKXSupported(coins) {
    const supported = [];
    const unsupported = [];

    for (const coin of coins) {
      const symbol = coin.symbol.toUpperCase();
      
      if (this.okxSupportedCoins.has(symbol)) {
        supported.push({
          rank: coin.market_cap_rank || supported.length + 1,
          symbol: symbol,
          name: coin.name,
          market_cap: coin.market_cap || 0,
          price: coin.current_price || 0
        });
      } else {
        unsupported.push(symbol);
      }
    }

    return { supported, unsupported };
  }

  async saveResults(data) {
    const timestamp = new Date().toISOString();
    
    const output = {
      timestamp,
      totalCoins: data.supported.length,
      coins: data.supported,
      unsupportedCoins: data.unsupported.slice(0, 20), // 只保存前20个不支持的
      note: 'Top coins by market cap that are supported on OKX'
    };

    await fs.writeFile(
      './historical/storage/raw/okx/top150-coins.json',
      JSON.stringify(output, null, 2)
    );

    // 只保存代币符号列表，方便收集器使用
    const symbolsList = data.supported.slice(0, 150).map(c => c.symbol);
    await fs.writeFile(
      './historical/storage/raw/okx/top150-symbols.json',
      JSON.stringify(symbolsList, null, 2)
    );

    console.log(`💾 保存了 ${symbolsList.length} 个代币到 top150-symbols.json`);
    
    return symbolsList;
  }

  async run() {
    console.log('🚀 获取市值前150代币列表\n');

    // 加载OKX支持的代币
    if (!await this.loadOKXCoins()) {
      return;
    }

    // 获取市值排名
    const topCoins = await this.getTopCoinsByMarketCap();
    
    // 过滤出OKX支持的代币
    const { supported, unsupported } = await this.filterOKXSupported(topCoins);

    console.log(`\n📊 统计结果:`);
    console.log(`- 获取了前 ${topCoins.length} 个代币`);
    console.log(`- OKX支持: ${supported.length} 个`);
    console.log(`- OKX不支持: ${unsupported.length} 个`);

    // 保存结果
    const top150 = await this.saveResults({ supported, unsupported });

    // 显示前20个
    console.log('\n🏆 市值前20且OKX支持的代币:');
    supported.slice(0, 20).forEach((coin, i) => {
      console.log(`${i + 1}. ${coin.symbol} - ${coin.name}`);
    });

    console.log('\n✅ 完成！代币列表已保存到:');
    console.log('- top150-coins.json (详细信息)');
    console.log('- top150-symbols.json (仅符号列表)');

    return top150;
  }
}

// 主函数
async function main() {
  const getter = new TopCoinsGetter();
  await getter.run();
}

// 运行
main().catch(console.error);