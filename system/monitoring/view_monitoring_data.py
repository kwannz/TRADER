#!/usr/bin/env python3
"""
查看实时监控数据
"""

import sqlite3
import json
from datetime import datetime

def view_monitoring_data():
    """查看监控数据"""
    db_path = "monitoring_data.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            print("="*80)
            print("📊 实时监控系统数据查看")
            print("="*80)
            
            # 查看市场快照
            print("\n📈 最近的市场快照:")
            cursor.execute("""
                SELECT timestamp, price, volume, funding_rate, whale_activity, 
                       fear_greed_index, sentiment, liquidity_risk, alert_level
                FROM market_snapshots 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            
            snapshots = cursor.fetchall()
            if snapshots:
                print(f"{'时间':<20} {'价格':<10} {'成交量':<10} {'资金费率':<10} {'巨鲸活动':<10} {'恐惧贪婪':<10} {'情绪':<10} {'流动性风险':<10} {'告警级别':<10}")
                print("-" * 120)
                for snapshot in snapshots:
                    timestamp = datetime.fromisoformat(snapshot[0]).strftime('%H:%M:%S')
                    price = f"${snapshot[1]:,.0f}" if snapshot[1] else "N/A"
                    volume = f"{snapshot[2]:,.0f}" if snapshot[2] else "N/A"
                    funding = f"{snapshot[3]:.4f}" if snapshot[3] else "N/A"
                    whale = snapshot[4] or "N/A"
                    fear_greed = f"{snapshot[5]:.1f}" if snapshot[5] else "N/A"
                    sentiment = snapshot[6] or "N/A"
                    liquidity = f"{snapshot[7]:.1f}%" if snapshot[7] else "N/A"
                    alert = snapshot[8] or "N/A"
                    
                    print(f"{timestamp:<20} {price:<10} {volume:<10} {funding:<10} {whale:<10} {fear_greed:<10} {sentiment:<10} {liquidity:<10} {alert:<10}")
            else:
                print("   暂无快照数据")
            
            # 查看告警记录
            print("\n🚨 最近的告警记录:")
            cursor.execute("""
                SELECT timestamp, alert_level, indicator, message, value, threshold
                FROM alerts 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            
            alerts = cursor.fetchall()
            if alerts:
                print(f"{'时间':<20} {'级别':<10} {'指标':<15} {'消息':<50} {'值':<10} {'阈值':<10}")
                print("-" * 120)
                for alert in alerts:
                    timestamp = datetime.fromisoformat(alert[0]).strftime('%H:%M:%S')
                    level = alert[1]
                    indicator = alert[2] or "N/A"
                    message = alert[3][:45] + "..." if len(alert[3]) > 45 else alert[3]
                    value = str(alert[4]) if alert[4] is not None else "N/A"
                    threshold = str(alert[5]) if alert[5] is not None else "N/A"
                    
                    print(f"{timestamp:<20} {level:<10} {indicator:<15} {message:<50} {value:<10} {threshold:<10}")
            else:
                print("   暂无告警记录")
            
            # 统计信息
            print("\n📊 统计信息:")
            
            # 总快照数
            cursor.execute("SELECT COUNT(*) FROM market_snapshots")
            total_snapshots = cursor.fetchone()[0]
            print(f"   总快照数: {total_snapshots}")
            
            # 总告警数
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = cursor.fetchone()[0]
            print(f"   总告警数: {total_alerts}")
            
            # 各级别告警统计
            cursor.execute("""
                SELECT alert_level, COUNT(*) as count 
                FROM alerts 
                GROUP BY alert_level 
                ORDER BY count DESC
            """)
            alert_stats = cursor.fetchall()
            if alert_stats:
                print("   告警级别统计:")
                for level, count in alert_stats:
                    print(f"     {level}: {count}次")
            
            # 最新快照详情
            cursor.execute("""
                SELECT * FROM market_snapshots 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            latest = cursor.fetchone()
            if latest:
                print(f"\n📍 最新市场状态 ({datetime.fromisoformat(latest[1]).strftime('%Y-%m-%d %H:%M:%S')}):")
                print(f"   价格: ${latest[2]:,.0f}" if latest[2] else "   价格: N/A")
                print(f"   成交量: {latest[3]:,.0f}" if latest[3] else "   成交量: N/A")
                print(f"   资金费率: {latest[4]:.4f}" if latest[4] else "   资金费率: N/A")
                print(f"   巨鲸活动: {latest[5]}" if latest[5] else "   巨鲸活动: N/A")
                print(f"   恐惧贪婪指数: {latest[6]:.1f}" if latest[6] else "   恐惧贪婪指数: N/A")
                print(f"   市场情绪: {latest[7]}" if latest[7] else "   市场情绪: N/A")
                print(f"   流动性风险: {latest[8]:.1f}%" if latest[8] else "   流动性风险: N/A")
                print(f"   告警级别: {latest[9]}" if latest[9] else "   告警级别: N/A")
            
    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
    except FileNotFoundError:
        print("监控数据库文件不存在，请先运行监控系统")

if __name__ == "__main__":
    view_monitoring_data()