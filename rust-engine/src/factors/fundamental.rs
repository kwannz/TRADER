//! 基本面因子
//! 
//! 实现基于财务数据的基本面分析因子

use serde::{Deserialize, Serialize};

/// 财务数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialData {
    pub revenue: f64,
    pub net_income: f64,
    pub total_assets: f64,
    pub shareholders_equity: f64,
    pub total_debt: f64,
    pub cash_and_equivalents: f64,
    pub shares_outstanding: f64,
    pub market_cap: f64,
}

/// 计算市盈率 (P/E Ratio)
/// 
/// # 参数
/// * `market_cap` - 市值
/// * `net_income` - 净利润
/// 
/// # 返回
/// 市盈率
pub fn pe_ratio(market_cap: f64, net_income: f64) -> f64 {
    if net_income <= 0.0 {
        f64::INFINITY
    } else {
        market_cap / net_income
    }
}

/// 计算市净率 (P/B Ratio)
/// 
/// # 参数
/// * `market_cap` - 市值
/// * `book_value` - 账面价值（股东权益）
/// 
/// # 返回
/// 市净率
pub fn pb_ratio(market_cap: f64, book_value: f64) -> f64 {
    if book_value <= 0.0 {
        f64::INFINITY
    } else {
        market_cap / book_value
    }
}

/// 计算市销率 (P/S Ratio)
/// 
/// # 参数
/// * `market_cap` - 市值
/// * `revenue` - 营业收入
/// 
/// # 返回
/// 市销率
pub fn ps_ratio(market_cap: f64, revenue: f64) -> f64 {
    if revenue <= 0.0 {
        f64::INFINITY
    } else {
        market_cap / revenue
    }
}

/// 计算资产收益率 (ROA)
/// 
/// # 参数
/// * `net_income` - 净利润
/// * `total_assets` - 总资产
/// 
/// # 返回
/// 资产收益率
pub fn roa(net_income: f64, total_assets: f64) -> f64 {
    if total_assets <= 0.0 {
        f64::NAN
    } else {
        net_income / total_assets
    }
}

/// 计算股东权益收益率 (ROE)
/// 
/// # 参数
/// * `net_income` - 净利润
/// * `shareholders_equity` - 股东权益
/// 
/// # 返回
/// 股东权益收益率
pub fn roe(net_income: f64, shareholders_equity: f64) -> f64 {
    if shareholders_equity <= 0.0 {
        f64::NAN
    } else {
        net_income / shareholders_equity
    }
}

/// 计算债务权益比 (D/E Ratio)
/// 
/// # 参数
/// * `total_debt` - 总债务
/// * `shareholders_equity` - 股东权益
/// 
/// # 返回
/// 债务权益比
pub fn debt_to_equity(total_debt: f64, shareholders_equity: f64) -> f64 {
    if shareholders_equity <= 0.0 {
        f64::INFINITY
    } else {
        total_debt / shareholders_equity
    }
}

/// 计算流动比率
/// 
/// # 参数
/// * `current_assets` - 流动资产
/// * `current_liabilities` - 流动负债
/// 
/// # 返回
/// 流动比率
pub fn current_ratio(current_assets: f64, current_liabilities: f64) -> f64 {
    if current_liabilities <= 0.0 {
        f64::INFINITY
    } else {
        current_assets / current_liabilities
    }
}

/// 计算每股收益 (EPS)
/// 
/// # 参数
/// * `net_income` - 净利润
/// * `shares_outstanding` - 流通股数
/// 
/// # 返回
/// 每股收益
pub fn eps(net_income: f64, shares_outstanding: f64) -> f64 {
    if shares_outstanding <= 0.0 {
        f64::NAN
    } else {
        net_income / shares_outstanding
    }
}

/// 计算每股净资产 (BVPS)
/// 
/// # 参数
/// * `shareholders_equity` - 股东权益
/// * `shares_outstanding` - 流通股数
/// 
/// # 返回
/// 每股净资产
pub fn book_value_per_share(shareholders_equity: f64, shares_outstanding: f64) -> f64 {
    if shares_outstanding <= 0.0 {
        f64::NAN
    } else {
        shareholders_equity / shares_outstanding
    }
}

/// 计算资产负债率
/// 
/// # 参数
/// * `total_debt` - 总债务
/// * `total_assets` - 总资产
/// 
/// # 返回
/// 资产负债率
pub fn debt_ratio(total_debt: f64, total_assets: f64) -> f64 {
    if total_assets <= 0.0 {
        f64::NAN
    } else {
        total_debt / total_assets
    }
}

/// 计算综合基本面评分
/// 
/// # 参数
/// * `financial_data` - 财务数据
/// 
/// # 返回
/// 综合评分（0-100）
pub fn fundamental_score(financial_data: &FinancialData) -> f64 {
    let mut score = 50.0; // 基准分数
    
    // ROE评分 (权重: 25%)
    let roe_val = roe(financial_data.net_income, financial_data.shareholders_equity);
    if !roe_val.is_nan() && roe_val > 0.0 {
        score += (roe_val * 100.0).min(25.0) * 0.25;
    }
    
    // ROA评分 (权重: 20%)
    let roa_val = roa(financial_data.net_income, financial_data.total_assets);
    if !roa_val.is_nan() && roa_val > 0.0 {
        score += (roa_val * 100.0).min(20.0) * 0.20;
    }
    
    // 债务权益比评分 (权重: 20%, 越低越好)
    let de_ratio = debt_to_equity(financial_data.total_debt, financial_data.shareholders_equity);
    if !de_ratio.is_infinite() && de_ratio >= 0.0 {
        let de_score = (1.0 / (1.0 + de_ratio)) * 20.0;
        score += de_score * 0.20;
    }
    
    // P/E比率评分 (权重: 15%, 适中最好)
    let pe = pe_ratio(financial_data.market_cap, financial_data.net_income);
    if !pe.is_infinite() && pe > 0.0 {
        let pe_score = if pe < 15.0 {
            pe
        } else if pe <= 25.0 {
            25.0 - pe
        } else {
            0.0
        };
        score += pe_score.max(0.0) * 0.15;
    }
    
    // P/B比率评分 (权重: 10%, 越低越好)
    let pb = pb_ratio(financial_data.market_cap, financial_data.shareholders_equity);
    if !pb.is_infinite() && pb > 0.0 {
        let pb_score = (5.0 / pb).min(10.0);
        score += pb_score * 0.10;
    }
    
    // 现金比率评分 (权重: 10%)
    let cash_ratio = financial_data.cash_and_equivalents / financial_data.total_assets;
    score += (cash_ratio * 100.0).min(10.0) * 0.10;
    
    score.min(100.0).max(0.0)
}

/// 计算Altman Z-Score（破产预测模型）
/// 
/// # 参数
/// * `working_capital` - 营运资本
/// * `total_assets` - 总资产
/// * `retained_earnings` - 留存收益
/// * `ebit` - 息税前利润
/// * `market_value_equity` - 股权市场价值
/// * `total_liabilities` - 总负债
/// * `sales` - 销售收入
/// 
/// # 返回
/// Altman Z-Score
pub fn altman_z_score(
    working_capital: f64,
    total_assets: f64,
    retained_earnings: f64,
    ebit: f64,
    market_value_equity: f64,
    total_liabilities: f64,
    sales: f64,
) -> f64 {
    if total_assets <= 0.0 {
        return f64::NAN;
    }
    
    let z1 = 1.2 * (working_capital / total_assets);
    let z2 = 1.4 * (retained_earnings / total_assets);
    let z3 = 3.3 * (ebit / total_assets);
    let z4 = 0.6 * (market_value_equity / total_liabilities);
    let z5 = 1.0 * (sales / total_assets);
    
    z1 + z2 + z3 + z4 + z5
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pe_ratio() {
        let pe = pe_ratio(1000000.0, 50000.0);
        assert_relative_eq!(pe, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roe() {
        let roe_val = roe(100000.0, 500000.0);
        assert_relative_eq!(roe_val, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_fundamental_score() {
        let financial_data = FinancialData {
            revenue: 1000000.0,
            net_income: 100000.0,
            total_assets: 800000.0,
            shareholders_equity: 400000.0,
            total_debt: 200000.0,
            cash_and_equivalents: 80000.0,
            shares_outstanding: 10000.0,
            market_cap: 2000000.0,
        };
        
        let score = fundamental_score(&financial_data);
        assert!(score >= 0.0 && score <= 100.0);
    }
}