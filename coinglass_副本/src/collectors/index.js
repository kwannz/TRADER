// 导入所有收集器
import { BaseCollector } from './base-collector.js';
import { ContractMarketCollector } from './contract-market-collector.js';
import { OpenInterestCollector } from './open-interest-collector.js';
import { FundingRateCollector } from './funding-rate-collector.js';
import { LongShortRatioCollector } from './long-short-ratio-collector.js';
import { LiquidationCollector } from './liquidation-collector.js';
import { ActiveTradeCollector } from './active-trade-collector.js';
import { ETFCollector } from './etf-collector.js';
import { FearGreedIndexCollector } from './fear-greed-index-collector.js';
import { IndicatorCollector } from './indicator-collector.js';
import { StablecoinMarketCapCollector } from './stablecoin-market-cap-collector.js';

// 导入新的收集器
import { OrderbookCollector } from './contract/orderbook-collector.js';
import { AggregateOpenInterestCollector } from './contract/aggregate-open-interest-collector.js';
import { LiquidationHeatmapCollector } from './contract/liquidation-heatmap-collector.js';
import { MacroIndicatorCollector } from './indicator/macro-indicator-collector.js';
import { ExchangeBalanceCollector } from './onchain/exchange-balance-collector.js';

import logger from "../utils/logger.js";
const { log } = logger;
import { config } from '../config/index.js';

// 收集器管理类
export class CollectorManager {
  constructor() {
    this.collectors = new Map();
    this.isRunning = false;
    this.collectionInterval = null;
    this.initializeCollectors();
  }

  // 初始化所有收集器
  initializeCollectors() {
    const collectorClasses = [
      ContractMarketCollector,
      OpenInterestCollector,
      FundingRateCollector,
      LongShortRatioCollector,
      LiquidationCollector,
      ActiveTradeCollector,
      ETFCollector,
      FearGreedIndexCollector,
      IndicatorCollector,
      StablecoinMarketCapCollector,
      // 新增的收集器
      OrderbookCollector,
      AggregateOpenInterestCollector,
      LiquidationHeatmapCollector,
      MacroIndicatorCollector,
      ExchangeBalanceCollector
    ];

    collectorClasses.forEach(CollectorClass => {
      const collector = new CollectorClass();
      this.collectors.set(collector.name, collector);
      log.info(`已初始化收集器: ${collector.name}`);
    });

    log.info(`收集器管理器初始化完成，共 ${this.collectors.size} 个收集器`);
  }

  // 获取所有收集器
  getAllCollectors() {
    return Array.from(this.collectors.values());
  }

  // 获取特定收集器
  getCollector(name) {
    return this.collectors.get(name);
  }

  // 运行单个收集器
  async runCollector(name) {
    const collector = this.collectors.get(name);
    if (!collector) {
      throw new Error(`收集器 ${name} 不存在`);
    }

    try {
      const result = await collector.run();
      log.info(`收集器 ${name} 执行完成`);
      return result;
    } catch (error) {
      log.error(`收集器 ${name} 执行失败`, error);
      throw error;
    }
  }

  // 并发运行所有可运行的收集器
  async runAllCollectors() {
    const collectors = this.getAllCollectors();
    const runnableCollectors = collectors.filter(collector => 
      collector.shouldCollect() && collector.errorCount < collector.maxErrors
    );

    if (runnableCollectors.length === 0) {
      log.debug('没有需要运行的收集器');
      return { success: 0, failed: 0 };
    }

    log.info(`开始运行 ${runnableCollectors.length} 个收集器`);

    // 使用 Promise.allSettled 来并发运行，避免单个失败影响其他
    const results = await Promise.allSettled(
      runnableCollectors.map(collector => collector.run())
    );

    let successCount = 0;
    let failedCount = 0;

    results.forEach((result, index) => {
      const collectorName = runnableCollectors[index].name;
      if (result.status === 'fulfilled') {
        successCount++;
        log.info(`收集器 ${collectorName} 执行成功，保存 ${result.value || 0} 条数据`);
      } else {
        failedCount++;
        log.error(`收集器 ${collectorName} 执行失败`, result.reason);
      }
    });

    log.info(`收集器批量执行完成: 成功 ${successCount}，失败 ${failedCount}`);
    return { success: successCount, failed: failedCount };
  }

  // 启动定时收集
  startScheduledCollection() {
    if (this.isRunning) {
      log.warn('定时收集已经在运行中');
      return;
    }

    const intervalMs = config.collection.intervalMinutes * 60 * 1000;
    
    this.collectionInterval = setInterval(async () => {
      try {
        await this.runAllCollectors();
      } catch (error) {
        log.error('定时收集执行失败', error);
      }
    }, intervalMs);

    this.isRunning = true;
    log.info(`定时收集已启动，间隔 ${config.collection.intervalMinutes} 分钟`);

    // 立即执行一次
    setTimeout(() => {
      this.runAllCollectors().catch(error => {
        log.error('初始收集执行失败', error);
      });
    }, 5000); // 5秒后开始第一次收集
  }

  // 停止定时收集
  stopScheduledCollection() {
    if (this.collectionInterval) {
      clearInterval(this.collectionInterval);
      this.collectionInterval = null;
    }
    
    this.isRunning = false;
    log.info('定时收集已停止');
  }

  // 手动触发收集所有数据
  async forceCollectAll() {
    log.info('开始强制收集所有数据');
    
    const collectors = this.getAllCollectors();
    const results = await Promise.allSettled(
      collectors.map(collector => collector.forceCollect())
    );

    let successCount = 0;
    let failedCount = 0;

    results.forEach((result, index) => {
      const collectorName = collectors[index].name;
      if (result.status === 'fulfilled') {
        successCount++;
        log.info(`强制收集 ${collectorName} 成功，保存 ${result.value || 0} 条数据`);
      } else {
        failedCount++;
        log.error(`强制收集 ${collectorName} 失败`, result.reason);
      }
    });

    log.info(`强制收集完成: 成功 ${successCount}，失败 ${failedCount}`);
    return { success: successCount, failed: failedCount };
  }

  // 清理所有收集器的旧数据
  async cleanAllOldData() {
    log.info('开始清理所有收集器的旧数据');
    
    const collectors = this.getAllCollectors();
    let totalDeleted = 0;

    for (const collector of collectors) {
      try {
        const deletedCount = await collector.cleanOldData();
        totalDeleted += deletedCount;
      } catch (error) {
        log.error(`清理 ${collector.name} 旧数据失败`, error);
      }
    }

    log.info(`旧数据清理完成，共删除 ${totalDeleted} 条记录`);
    return totalDeleted;
  }

  // 获取所有收集器状态
  getCollectorsStatus() {
    const collectors = this.getAllCollectors();
    return collectors.map(collector => collector.getStatus());
  }

  // 重置所有收集器错误计数
  resetAllErrors() {
    const collectors = this.getAllCollectors();
    collectors.forEach(collector => collector.resetErrors());
    log.info('已重置所有收集器的错误计数');
  }

  // 获取系统状态
  getSystemStatus() {
    const status = this.getCollectorsStatus();
    const healthyCount = status.filter(s => s.isHealthy).length;
    const runningCount = status.filter(s => s.isRunning).length;
    
    return {
      isScheduledRunning: this.isRunning,
      totalCollectors: status.length,
      healthyCollectors: healthyCount,
      runningCollectors: runningCount,
      collectionInterval: config.collection.intervalMinutes,
      collectors: status
    };
  }
}

// 创建全局收集器管理器实例
export const collectorManager = new CollectorManager();

// 导出所有收集器类
export {
  BaseCollector,
  ContractMarketCollector,
  OpenInterestCollector,
  FundingRateCollector,
  LongShortRatioCollector,
  LiquidationCollector,
  ActiveTradeCollector,
  ETFCollector,
  FearGreedIndexCollector,
  IndicatorCollector,
  StablecoinMarketCapCollector,
  // 新增的收集器
  OrderbookCollector,
  AggregateOpenInterestCollector,
  LiquidationHeatmapCollector,
  MacroIndicatorCollector,
  ExchangeBalanceCollector
}; 