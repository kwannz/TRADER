#!/usr/bin/env node
import { dbManager } from '../src/models/index.js';
import { config } from '../src/config/index.js';
import logger from "../src/utils/logger.js";
const { log } = logger;

async function initDatabase() {
  try {
    console.log('🚀 开始初始化数据库...');
    
    // 连接数据库
    console.log(`📡 正在连接数据库: ${config.database.uri}`);
    await dbManager.connect(config.database.uri, config.database.options);
    
    // 创建索引
    console.log('🔧 正在创建数据库索引...');
    await dbManager.createIndexes();
    
    // 健康检查
    console.log('🏥 执行数据库健康检查...');
    const healthCheck = await dbManager.healthCheck();
    
    if (healthCheck.status === 'healthy') {
      console.log('✅ 数据库健康检查通过');
    } else {
      console.log('❌ 数据库健康检查失败:', healthCheck.error);
      process.exit(1);
    }
    
    // 显示连接状态
    const status = dbManager.getConnectionStatus();
    console.log('\n📊 数据库连接状态:');
    console.log(`  连接状态: ${status.isConnected ? '✅ 已连接' : '❌ 未连接'}`);
    console.log(`  数据库主机: ${status.host}:${status.port}`);
    console.log(`  数据库名称: ${status.name}`);
    console.log(`  连接就绪状态: ${status.readyState}`);
    
    console.log('\n🎉 数据库初始化完成！');
    
  } catch (error) {
    console.error('❌ 数据库初始化失败:', error.message);
    log.error('数据库初始化失败', error);
    process.exit(1);
  } finally {
    // 断开连接
    await dbManager.disconnect();
    console.log('👋 数据库连接已断开');
  }
}

// 运行初始化
initDatabase(); 