const winston = require('winston');
const path = require('path');

// 创建logs目录
const logsDir = path.join(process.cwd(), 'logs');

// 日志格式配置
const logFormat = winston.format.combine(
  winston.format.timestamp({
    format: 'YYYY-MM-DD HH:mm:ss'
  }),
  winston.format.errors({ stack: true }),
  winston.format.json()
);

// 控制台日志格式
const consoleFormat = winston.format.combine(
  winston.format.colorize(),
  winston.format.timestamp({
    format: 'YYYY-MM-DD HH:mm:ss'
  }),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    let msg = `${timestamp} [${level}]: ${message}`;
    
    // 添加元数据
    if (Object.keys(meta).length > 0) {
      msg += '\n' + JSON.stringify(meta, null, 2);
    }
    
    return msg;
  })
);

// 创建Winston logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: logFormat,
  defaultMeta: { 
    service: 'coinglass-collector',
    version: '1.0.0'
  },
  transports: [
    // 控制台输出
    new winston.transports.Console({
      format: consoleFormat,
      handleExceptions: true,
      handleRejections: true
    }),
    
    // 文件输出 - 所有日志
    new winston.transports.File({
      filename: path.join(logsDir, 'app.log'),
      format: logFormat,
      handleExceptions: true,
      handleRejections: true,
      maxsize: 50 * 1024 * 1024, // 50MB
      maxFiles: 10,
      tailable: true
    }),
    
    // 文件输出 - 错误日志
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level: 'error',
      format: logFormat,
      handleExceptions: true,
      handleRejections: true,
      maxsize: 20 * 1024 * 1024, // 20MB
      maxFiles: 5,
      tailable: true
    }),
    
    // 文件输出 - 数据收集日志
    new winston.transports.File({
      filename: path.join(logsDir, 'collection.log'),
      format: logFormat,
      maxsize: 30 * 1024 * 1024, // 30MB
      maxFiles: 7,
      tailable: true
    })
  ],
  
  // 退出时的处理
  exitOnError: false
});

// 创建子logger用于特定模块
const createChildLogger = (module) => {
  return logger.child({ module });
};

// 数据收集专用logger
const collectionLogger = logger.child({ 
  module: 'data-collection',
  transport: 'collection.log'
});

// API请求专用logger
const apiLogger = logger.child({ 
  module: 'api-client',
  transport: 'app.log'
});

// 数据库专用logger
const dbLogger = logger.child({ 
  module: 'database',
  transport: 'app.log'
});

// 调度器专用logger
const schedulerLogger = logger.child({ 
  module: 'scheduler',
  transport: 'app.log'
});

// 监控专用logger
const monitorLogger = logger.child({ 
  module: 'monitor',
  transport: 'app.log'
});

// 错误统计
let errorStats = {
  total: 0,
  lastHour: 0,
  lastResetTime: Date.now()
};

// 每小时重置错误计数
setInterval(() => {
  errorStats.lastHour = 0;
  errorStats.lastResetTime = Date.now();
}, 60 * 60 * 1000);

// 增强的错误日志记录
const logError = (message, error, metadata = {}) => {
  errorStats.total++;
  errorStats.lastHour++;
  
  const errorInfo = {
    message,
    error: {
      name: error?.name,
      message: error?.message,
      stack: error?.stack,
      code: error?.code
    },
    timestamp: new Date().toISOString(),
    errorStats: { ...errorStats },
    ...metadata
  };
  
  logger.error(errorInfo);
};

// 性能监控日志
const logPerformance = (operation, duration, metadata = {}) => {
  const perfInfo = {
    operation,
    duration: `${duration}ms`,
    timestamp: new Date().toISOString(),
    ...metadata
  };
  
  if (duration > 5000) {
    logger.warn('Slow operation detected', perfInfo);
  } else {
    logger.debug('Performance metric', perfInfo);
  }
};

// API请求日志
const logApiRequest = (method, url, status, duration, metadata = {}) => {
  const requestInfo = {
    method,
    url: url?.replace(/api_key=[^&]+/, 'api_key=***'), // 隐藏API密钥
    status,
    duration: `${duration}ms`,
    timestamp: new Date().toISOString(),
    ...metadata
  };
  
  if (status >= 400) {
    apiLogger.error('API request failed', requestInfo);
  } else if (duration > 3000) {
    apiLogger.warn('Slow API request', requestInfo);
  } else {
    apiLogger.info('API request', requestInfo);
  }
};

// 数据收集日志
const logDataCollection = (collector, action, result, metadata = {}) => {
  const collectionInfo = {
    collector,
    action,
    result: {
      success: result.success,
      count: result.count || 0,
      errors: result.errors || 0,
      duration: result.duration ? `${result.duration}ms` : undefined
    },
    timestamp: new Date().toISOString(),
    ...metadata
  };
  
  if (result.success) {
    collectionLogger.info('Data collection completed', collectionInfo);
  } else {
    collectionLogger.error('Data collection failed', collectionInfo);
  }
};

// 系统健康日志
const logSystemHealth = (component, status, metrics = {}) => {
  const healthInfo = {
    component,
    status,
    metrics,
    timestamp: new Date().toISOString()
  };
  
  if (status === 'healthy') {
    monitorLogger.info('System health check', healthInfo);
  } else {
    monitorLogger.warn('System health issue', healthInfo);
  }
};

// 获取错误统计
const getErrorStats = () => {
  return { ...errorStats };
};

// 清理日志统计
const resetErrorStats = () => {
  errorStats = {
    total: 0,
    lastHour: 0,
    lastResetTime: Date.now()
  };
};

module.exports = {
  // 主logger
  logger,
  
  // 专用loggers
  createChildLogger,
  collectionLogger,
  apiLogger,
  dbLogger,
  schedulerLogger,
  monitorLogger,
  
  // 增强日志方法
  logError,
  logPerformance,
  logApiRequest,
  logDataCollection,
  logSystemHealth,
  
  // 统计方法
  getErrorStats,
  resetErrorStats,
  
  // 标准日志方法（向后兼容）
  info: logger.info.bind(logger),
  warn: logger.warn.bind(logger),
  error: logger.error.bind(logger),
  debug: logger.debug.bind(logger),
  verbose: logger.verbose.bind(logger),
  
  // ES6 export compatibility
  log: {
    info: logger.info.bind(logger),
    warn: logger.warn.bind(logger),
    error: logger.error.bind(logger),
    debug: logger.debug.bind(logger),
    verbose: logger.verbose.bind(logger)
  }
}; 