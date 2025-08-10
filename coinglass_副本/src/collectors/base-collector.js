const { logDataCollection, logError } = require('../utils/logger');
const JsonStorageManager = require('../database/json-storage');

class BaseCollector {
  constructor(name, collectionName, coinGlassClient = null, options = {}) {
    this.name = name;
    this.collectionName = collectionName;
    this.coinGlassClient = coinGlassClient;
    this.isRunning = false;
    this.lastRunTime = null;
    this.errorCount = 0;
    this.successCount = 0;
    this.totalDocuments = 0;
    this.jsonStorage = null;
    
    // 配置选项
    this.options = {
      batchSize: 1000,
      retryAttempts: 3,
      retryDelay: 5000,
      enableDuplicateCheck: true,
      ...options
    };
  }

  /**
   * 开始数据收集流程
   */
  async collect() {
    if (this.isRunning) {
      throw new Error(`Collector ${this.name} is already running`);
    }

    const startTime = Date.now();
    this.isRunning = true;
    
    try {
      // 收集前的准备工作
      await this.beforeCollect();
      
      // 执行数据收集
      const result = await this.performCollection();
      
      // 收集后的清理工作
      await this.afterCollect(result);
      
      const duration = Date.now() - startTime;
      this.lastRunTime = new Date();
      this.successCount++;
      
      const finalResult = {
        success: true,
        count: result.count || 0,
        duration,
        errors: 0,
        ...result
      };
      
      logDataCollection(this.name, 'collect', finalResult);
      return finalResult;
      
    } catch (error) {
      const duration = Date.now() - startTime;
      this.errorCount++;
      
      const errorResult = {
        success: false,
        count: 0,
        duration,
        errors: 1,
        error: error.message
      };
      
      logDataCollection(this.name, 'collect', errorResult, { error: error.message });
      throw error;
    } finally {
      this.isRunning = false;
    }
  }

  /**
   * 收集前的准备工作（子类可重写）
   */
  async beforeCollect() {
    // 默认实现为空
  }

  /**
   * 执行实际的数据收集（子类必须实现）
   */
  async performCollection() {
    throw new Error('performCollection method must be implemented by subclass');
  }

  /**
   * 收集后的清理工作（子类可重写）
   */
  async afterCollect(result) {
    // 默认实现为空
  }

  /**
   * 保存数据到JSON文件
   */
  async saveData(data, options = {}) {
    if (!data || data.length === 0) {
      return { insertedCount: 0 };
    }

    const { 
      enableDuplicateCheck = this.options.enableDuplicateCheck,
      batchSize = this.options.batchSize
    } = options;

    try {
      // 初始化JSON存储
      if (!this.jsonStorage) {
        this.jsonStorage = new JsonStorageManager();
        await this.jsonStorage.connect();
      }
      
      // 数据预处理
      const processedData = await this.preprocessData(data);
      
      // 去重检查
      let uniqueData = processedData;
      if (enableDuplicateCheck) {
        uniqueData = await this.removeDuplicates(processedData);
      }

      if (uniqueData.length === 0) {
        return { insertedCount: 0 };
      }

      // 保存到JSON文件
      const result = await this.jsonStorage.saveToCollection(
        this.collectionName, 
        uniqueData,
        {
          upsert: enableDuplicateCheck,
          uniqueField: 'timestamp'
        }
      );

      this.totalDocuments += result.insertedCount;
      
      return { 
        insertedCount: result.insertedCount,
        totalProcessed: data.length,
        duplicatesRemoved: processedData.length - uniqueData.length
      };
      
    } catch (error) {
      logError(`Failed to save data in ${this.name}`, error, {
        collectionName: this.collectionName,
        dataCount: data.length
      });
      throw error;
    }
  }

  /**
   * 数据预处理（子类可重写）
   */
  async preprocessData(data) {
    return data.map(item => ({
      ...item,
      timestamp: item.timestamp || new Date(),
      collector_name: this.name,
      collection_time: new Date()
    }));
  }

  /**
   * 去除重复数据
   */
  async removeDuplicates(data) {
    // 默认实现：基于时间戳和符号去重
    const seen = new Set();
    return data.filter(item => {
      const key = this.generateDuplicateKey(item);
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  /**
   * 生成去重键（子类可重写）
   */
  generateDuplicateKey(item) {
    const timestamp = item.timestamp ? new Date(item.timestamp).getTime() : Date.now();
    const symbol = item.symbol || 'unknown';
    const exchange = item.exchange || 'unknown';
    return `${symbol}-${exchange}-${timestamp}`;
  }

  /**
   * 带重试的数据收集
   */
  async collectWithRetry() {
    let lastError;
    
    for (let attempt = 1; attempt <= this.options.retryAttempts; attempt++) {
      try {
        return await this.collect();
      } catch (error) {
        lastError = error;
        
        if (attempt === this.options.retryAttempts) {
          break;
        }
        
        logError(`Collector ${this.name} attempt ${attempt} failed, retrying...`, error);
        await this.sleep(this.options.retryDelay * attempt);
      }
    }
    
    throw lastError;
  }

  /**
   * 获取数据的时间范围
   */
  async getDataTimeRange() {
    try {
      // 初始化JSON存储
      if (!this.jsonStorage) {
        this.jsonStorage = new JsonStorageManager();
        await this.jsonStorage.connect();
      }
      
      const data = await this.jsonStorage.loadFromCollection(this.collectionName);
      
      if (data.length > 0) {
        const timestamps = data.map(item => new Date(item.timestamp)).filter(d => !isNaN(d));
        
        if (timestamps.length > 0) {
          return {
            minTime: new Date(Math.min(...timestamps)),
            maxTime: new Date(Math.max(...timestamps)),
            totalDocuments: data.length
          };
        }
      }
      
      return null;
    } catch (error) {
      logError(`Failed to get data time range for ${this.name}`, error);
      return null;
    }
  }

  /**
   * 清理过期数据
   */
  async cleanupOldData(retentionDays = 365) {
    try {
      // 初始化JSON存储
      if (!this.jsonStorage) {
        this.jsonStorage = new JsonStorageManager();
        await this.jsonStorage.connect();
      }
      
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - retentionDays);
      
      const data = await this.jsonStorage.loadFromCollection(this.collectionName);
      const filteredData = data.filter(item => {
        const itemDate = new Date(item.timestamp);
        return itemDate >= cutoffDate;
      });
      
      const deletedCount = data.length - filteredData.length;
      
      if (deletedCount > 0) {
        await this.jsonStorage.saveToCollection(this.collectionName, filteredData);
        
        logDataCollection(this.name, 'cleanup', {
          success: true,
          count: deletedCount,
          cutoffDate: cutoffDate.toISOString()
        });
      }
      
      return deletedCount;
    } catch (error) {
      logError(`Failed to cleanup old data for ${this.name}`, error);
      return 0;
    }
  }

  /**
   * 获取收集器统计信息
   */
  getStats() {
    return {
      name: this.name,
      collectionName: this.collectionName,
      isRunning: this.isRunning,
      lastRunTime: this.lastRunTime,
      errorCount: this.errorCount,
      successCount: this.successCount,
      totalDocuments: this.totalDocuments,
      successRate: this.successCount + this.errorCount > 0 ? 
        (this.successCount / (this.successCount + this.errorCount) * 100).toFixed(2) : 0
    };
  }

  /**
   * 重置统计信息
   */
  resetStats() {
    this.errorCount = 0;
    this.successCount = 0;
    this.totalDocuments = 0;
    this.lastRunTime = null;
  }

  /**
   * 检查收集器健康状态
   */
  async healthCheck() {
    try {
      // 初始化JSON存储
      if (!this.jsonStorage) {
        this.jsonStorage = new JsonStorageManager();
        await this.jsonStorage.connect();
      }
      
      // 检查JSON存储健康状态
      if (!this.jsonStorage.isHealthy()) {
        return {
          status: 'unhealthy',
          reason: 'JSON storage connection issue'
        };
      }

      // 检查最近的数据
      const timeRange = await this.getDataTimeRange();
      const now = new Date();
      const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
      
      let hasRecentData = false;
      if (timeRange && timeRange.maxTime) {
        hasRecentData = timeRange.maxTime > oneHourAgo;
      }

      return {
        status: hasRecentData ? 'healthy' : 'warning',
        reason: hasRecentData ? 'Operating normally' : 'No recent data',
        stats: this.getStats(),
        dataTimeRange: timeRange
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        reason: 'Health check failed',
        error: error.message
      };
    }
  }

  /**
   * 延迟函数
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * 格式化时间戳
   */
  formatTimestamp(timestamp) {
    if (!timestamp) return new Date();
    
    if (typeof timestamp === 'string') {
      return new Date(timestamp);
    }
    
    if (typeof timestamp === 'number') {
      // 如果是秒级时间戳，转换为毫秒
      if (timestamp < 1e12) {
        timestamp *= 1000;
      }
      return new Date(timestamp);
    }
    
    return timestamp instanceof Date ? timestamp : new Date();
  }

  /**
   * 验证必需字段
   */
  validateRequiredFields(data, requiredFields) {
    const errors = [];
    
    for (const field of requiredFields) {
      if (data[field] === undefined || data[field] === null) {
        errors.push(`Missing required field: ${field}`);
      }
    }
    
    if (errors.length > 0) {
      throw new Error(`Validation failed: ${errors.join(', ')}`);
    }
  }

  /**
   * 安全的数值转换
   */
  safeParseNumber(value, defaultValue = 0) {
    if (value === null || value === undefined) {
      return defaultValue;
    }
    
    const parsed = parseFloat(value);
    return isNaN(parsed) ? defaultValue : parsed;
  }

  /**
   * 安全的字符串转换
   */
  safeParseString(value, defaultValue = '') {
    if (value === null || value === undefined) {
      return defaultValue;
    }
    
    return String(value);
  }
}

module.exports = BaseCollector; 