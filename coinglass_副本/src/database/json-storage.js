const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');

class JsonStorageManager {
  constructor() {
    this.baseDir = path.join(__dirname, '../../data');
    this.isConnected = false;
  }

  /**
   * Initialize JSON storage (create directories)
   */
  async connect() {
    try {
      // Create base directories if they don't exist
      await this.ensureDirectory(this.baseDir);
      await this.ensureDirectory(path.join(this.baseDir, 'coinglass'));
      
      this.isConnected = true;
      
      logger.info('Successfully initialized JSON file storage', {
        baseDirectory: this.baseDir
      });
      
      return true;
    } catch (error) {
      logger.error('Failed to initialize JSON storage', { error: error.message });
      throw error;
    }
  }

  /**
   * Disconnect (cleanup if needed)
   */
  async disconnect() {
    this.isConnected = false;
    logger.info('JSON storage disconnected');
  }

  /**
   * Save data to JSON file
   */
  async saveToCollection(collectionName, data, options = {}) {
    try {
      if (!Array.isArray(data)) {
        data = [data];
      }
      
      const filePath = this.getCollectionPath(collectionName);
      await this.ensureDirectory(path.dirname(filePath));
      
      let existingData = [];
      
      // Load existing data if file exists
      try {
        const existingContent = await fs.readFile(filePath, 'utf8');
        existingData = JSON.parse(existingContent);
      } catch (error) {
        // File doesn't exist or is invalid, start fresh
        existingData = [];
      }
      
      // Merge data (avoid duplicates if specified)
      let finalData;
      if (options.upsert && options.uniqueField) {
        // Remove duplicates based on unique field
        const existingMap = new Map();
        existingData.forEach(item => {
          existingMap.set(item[options.uniqueField], item);
        });
        
        data.forEach(item => {
          existingMap.set(item[options.uniqueField], item);
        });
        
        finalData = Array.from(existingMap.values());
      } else {
        // Simple append
        finalData = [...existingData, ...data];
      }
      
      // Save to file
      await fs.writeFile(filePath, JSON.stringify(finalData, null, 2));
      
      return {
        insertedCount: data.length,
        totalCount: finalData.length,
        duplicatesRemoved: existingData.length + data.length - finalData.length
      };
      
    } catch (error) {
      logger.error('Failed to save data to JSON file', {
        collection: collectionName,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Load data from JSON file
   */
  async loadFromCollection(collectionName, filter = {}) {
    try {
      const filePath = this.getCollectionPath(collectionName);
      
      try {
        const content = await fs.readFile(filePath, 'utf8');
        let data = JSON.parse(content);
        
        // Apply simple filters
        if (Object.keys(filter).length > 0) {
          data = data.filter(item => {
            return Object.entries(filter).every(([key, value]) => {
              return item[key] === value;
            });
          });
        }
        
        return data;
      } catch (error) {
        // File doesn't exist, return empty array
        return [];
      }
      
    } catch (error) {
      logger.error('Failed to load data from JSON file', {
        collection: collectionName,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Get collection file path
   */
  getCollectionPath(collectionName) {
    return path.join(this.baseDir, 'coinglass', `${collectionName}.json`);
  }

  /**
   * Ensure directory exists
   */
  async ensureDirectory(dirPath) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
      if (error.code !== 'EEXIST') {
        throw error;
      }
    }
  }

  /**
   * Get storage statistics
   */
  async getStats() {
    try {
      const coinglassDir = path.join(this.baseDir, 'coinglass');
      const files = await fs.readdir(coinglassDir);
      const jsonFiles = files.filter(f => f.endsWith('.json'));
      
      const stats = {
        collections: jsonFiles.length,
        totalSize: 0,
        collectionSizes: {}
      };
      
      for (const file of jsonFiles) {
        const filePath = path.join(coinglassDir, file);
        const stat = await fs.stat(filePath);
        const collectionName = file.replace('.json', '');
        stats.collectionSizes[collectionName] = {
          size: stat.size,
          modified: stat.mtime
        };
        stats.totalSize += stat.size;
      }
      
      return stats;
    } catch (error) {
      logger.error('Failed to get storage stats', { error: error.message });
      return { collections: 0, totalSize: 0, collectionSizes: {} };
    }
  }

  /**
   * Check if storage is healthy
   */
  isHealthy() {
    return this.isConnected;
  }
}

module.exports = JsonStorageManager;