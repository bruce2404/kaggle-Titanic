#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
泰坦尼克号生存预测 - 快速运行脚本
简化版本，直接执行预测流程
"""

from titanic_survival_prediction import TitanicSurvivalPredictor

def main():
    """
    快速运行泰坦尼克号生存预测
    """
    print("🚢 启动泰坦尼克号生存预测...")
    
    # 创建预测器
    predictor = TitanicSurvivalPredictor()
    
    # 运行完整流水线
    try:
        submission = predictor.run_complete_pipeline('train.csv', 'test.csv')
        print("\n✅ 预测成功完成！")
        print("📁 生成文件:")
        print("   - titanic_submission.csv (Kaggle提交文件)")
        print("   - titanic_eda.png (数据分析图表)")
        print("   - feature_importance.png (特征重要性)")
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        print("请检查数据文件是否存在且格式正确")

if __name__ == "__main__":
    main()