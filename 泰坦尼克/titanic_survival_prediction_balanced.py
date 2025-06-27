#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
泰坦尼克号生存预测 - 平衡优化版本
基于Kaggle实际分数反馈的改进版本
重点解决过拟合问题，平衡模型复杂度与泛化能力
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TitanicSurvivalPredictorBalanced:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.best_model = None
        
    def load_data(self, train_path, test_path):
        """
        加载训练和测试数据
        """
        print("=== 数据加载阶段 ===")
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"训练数据形状: {self.train_data.shape}")
        print(f"测试数据形状: {self.test_data.shape}")
        print(f"训练数据缺失值:\n{self.train_data.isnull().sum()}")
        
    def exploratory_data_analysis(self):
        """
        探索性数据分析
        """
        print("\n=== 探索性数据分析阶段 ===")
        
        # 生存率统计
        survival_rate = self.train_data['Survived'].mean()
        print(f"总体生存率: {survival_rate:.2%}")
        
        # 创建可视化图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('泰坦尼克号数据探索分析 - 平衡版本', fontsize=16)
        
        # 性别与生存率
        sex_survival = self.train_data.groupby('Sex')['Survived'].mean()
        axes[0,0].bar(sex_survival.index, sex_survival.values)
        axes[0,0].set_title('性别与生存率')
        axes[0,0].set_ylabel('生存率')
        
        # 船舱等级与生存率
        class_survival = self.train_data.groupby('Pclass')['Survived'].mean()
        axes[0,1].bar(class_survival.index, class_survival.values)
        axes[0,1].set_title('船舱等级与生存率')
        axes[0,1].set_ylabel('生存率')
        
        # 年龄分布
        axes[0,2].hist(self.train_data['Age'].dropna(), bins=30, alpha=0.7)
        axes[0,2].set_title('年龄分布')
        axes[0,2].set_xlabel('年龄')
        
        # 票价分布
        axes[1,0].hist(self.train_data['Fare'].dropna(), bins=30, alpha=0.7)
        axes[1,0].set_title('票价分布')
        axes[1,0].set_xlabel('票价')
        
        # 登船港口与生存率
        embarked_survival = self.train_data.groupby('Embarked')['Survived'].mean()
        axes[1,1].bar(embarked_survival.index, embarked_survival.values)
        axes[1,1].set_title('登船港口与生存率')
        axes[1,1].set_ylabel('生存率')
        
        # 家庭大小与生存率
        self.train_data['Family_Size'] = self.train_data['SibSp'] + self.train_data['Parch'] + 1
        family_survival = self.train_data.groupby('Family_Size')['Survived'].mean()
        axes[1,2].plot(family_survival.index, family_survival.values, marker='o')
        axes[1,2].set_title('家庭大小与生存率')
        axes[1,2].set_xlabel('家庭大小')
        axes[1,2].set_ylabel('生存率')
        
        plt.tight_layout()
        plt.savefig('titanic_eda_balanced.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def balanced_feature_engineering(self):
        """
        平衡的特征工程 - 简化但有效
        """
        print("\n=== 平衡特征工程阶段 ===")
        
        # 合并训练和测试数据
        all_data = pd.concat([self.train_data, self.test_data], ignore_index=True)
        
        # 1. 基础缺失值处理（保守策略）
        print("处理缺失值...")
        
        # Age: 使用中位数按性别和船舱等级分组填充
        all_data['Age'] = all_data.groupby(['Sex', 'Pclass'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        all_data['Age'].fillna(all_data['Age'].median(), inplace=True)
        
        # Embarked: 使用众数填充
        all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
        
        # Fare: 使用中位数按船舱等级填充
        all_data['Fare'] = all_data.groupby('Pclass')['Fare'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 2. 核心特征工程（经过验证的稳定特征）
        print("创建核心特征...")
        
        # 2.1 船舱特征
        all_data['Has_Cabin'] = all_data['Cabin'].notna().astype(int)
        
        # 2.2 家庭特征
        all_data['Family_Size'] = all_data['SibSp'] + all_data['Parch'] + 1
        all_data['Is_Alone'] = (all_data['Family_Size'] == 1).astype(int)
        
        # 2.3 称谓特征（简化版本）
        all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # 简化的称谓分组
        title_mapping = {
            'Mr': 'Mr',
            'Mrs': 'Mrs', 
            'Miss': 'Miss',
            'Master': 'Master'
        }
        all_data['Title'] = all_data['Title'].map(title_mapping)
        all_data['Title'].fillna('Other', inplace=True)
        
        # 2.4 年龄分组（简化）
        all_data['Age_Group'] = pd.cut(all_data['Age'], 
                                      bins=[0, 16, 32, 48, 80], 
                                      labels=['Child', 'Young', 'Adult', 'Senior'])
        all_data['Age_Group'] = all_data['Age_Group'].astype(str)
        
        # 2.5 票价分组（简化）
        all_data['Fare_Group'] = pd.qcut(all_data['Fare'], 
                                        q=4, 
                                        labels=['Low', 'Medium', 'High', 'Very_High'])
        all_data['Fare_Group'] = all_data['Fare_Group'].astype(str)
        
        # 2.6 家庭类型（简化）
        all_data['Family_Type'] = 'Single'
        all_data.loc[all_data['Family_Size'].between(2, 4), 'Family_Type'] = 'Small'
        all_data.loc[all_data['Family_Size'] > 4, 'Family_Type'] = 'Large'
        
        # 3. 特征编码
        print("编码特征...")
        
        # 性别编码
        all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})
        
        # 登船港口编码
        embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
        all_data['Embarked'] = all_data['Embarked'].map(embarked_mapping)
        
        # 称谓编码
        title_mapping = {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Other': 4}
        all_data['Title'] = all_data['Title'].map(title_mapping)
        
        # 年龄组编码
        age_group_mapping = {'Child': 0, 'Young': 1, 'Adult': 2, 'Senior': 3}
        all_data['Age_Group'] = all_data['Age_Group'].map(age_group_mapping)
        
        # 票价组编码
        fare_group_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3}
        all_data['Fare_Group'] = all_data['Fare_Group'].map(fare_group_mapping)
        
        # 家庭类型编码
        family_type_mapping = {'Single': 0, 'Small': 1, 'Large': 2}
        all_data['Family_Type'] = all_data['Family_Type'].map(family_type_mapping)
        
        # 4. 选择最终特征（保守选择）
        self.feature_columns = [
            # 基础特征
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            # 核心衍生特征
            'Has_Cabin', 'Family_Size', 'Is_Alone', 'Title',
            'Age_Group', 'Fare_Group', 'Family_Type'
        ]
        
        # 分离训练和测试数据
        train_len = len(self.train_data)
        self.train_processed = all_data[:train_len][self.feature_columns + ['Survived']]
        self.test_processed = all_data[train_len:][self.feature_columns]
        
        print(f"平衡特征工程完成，最终特征数量: {len(self.feature_columns)}")
        print(f"特征列表: {self.feature_columns}")
        
    def train_balanced_models(self):
        """
        训练平衡的机器学习模型
        """
        print("\n=== 平衡模型训练阶段 ===")
        
        # 准备训练数据
        X = self.train_processed.drop('Survived', axis=1)
        y = self.train_processed['Survived']
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 定义平衡的模型（减少复杂度，增强泛化）
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=4, 
                subsample=0.8, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0, random_state=42, max_iter=1000
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42
            )
        }
        
        # 训练和评估每个模型
        model_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\n训练 {name} 模型...")
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            model_scores[name] = cv_scores.mean()
            
            print(f"{name} 交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 验证集预测
            val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            print(f"{name} 验证集准确率: {val_accuracy:.4f}")
            
            # 保存模型
            self.models[name] = model
        
        # 创建保守的集成模型
        print("\n创建平衡集成模型...")
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', models['RandomForest']),
                ('gb', models['GradientBoosting']),
                ('lr', models['LogisticRegression']),
                ('svm', models['SVM'])
            ],
            voting='soft'
        )
        
        # 训练集成模型
        voting_clf.fit(X_train, y_train)
        self.models['Balanced_Ensemble'] = voting_clf
        
        # 评估集成模型
        ensemble_cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')
        model_scores['Balanced_Ensemble'] = ensemble_cv_scores.mean()
        
        ensemble_val_pred = voting_clf.predict(X_val)
        ensemble_val_accuracy = accuracy_score(y_val, ensemble_val_pred)
        
        print(f"\nBalanced_Ensemble 交叉验证准确率: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
        print(f"Balanced_Ensemble 验证集准确率: {ensemble_val_accuracy:.4f}")
        
        # 选择最佳模型
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name} (CV分数: {model_scores[best_model_name]:.4f})")
        
        return model_scores
        
    def make_predictions(self):
        """
        使用最佳模型进行预测
        """
        print("\n=== 预测阶段 ===")
        
        # 准备测试数据
        X_test = self.scaler.transform(self.test_processed)
        
        # 使用最佳模型预测
        predictions = self.best_model.predict(X_test)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'PassengerId': self.test_data['PassengerId'],
            'Survived': predictions.astype(int)  # 确保是整数格式
        })
        
        submission.to_csv('titanic_submission_balanced.csv', index=False)
        print(f"预测完成，生存人数: {predictions.sum()}/{len(predictions)}")
        
        return submission
        
    def feature_importance_analysis(self):
        """
        分析特征重要性
        """
        print("\n=== 特征重要性分析 ===")
        
        # 获取随机森林模型的特征重要性
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']
            feature_names = self.feature_columns
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                '特征': feature_names,
                '重要性': rf_model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            print("\n特征重要性排序 (随机森林):")
            for idx, row in importance_df.iterrows():
                print(f"{row['特征']}: {row['重要性']:.4f}")
            
            # 可视化特征重要性
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='重要性', y='特征')
            plt.title('特征重要性分析 (平衡版本)')
            plt.xlabel('重要性分数')
            plt.tight_layout()
            plt.savefig('feature_importance_balanced.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def run_complete_balanced_pipeline(self, train_path, test_path):
        """
        运行完整的平衡机器学习流水线
        """
        print("🚢 泰坦尼克号生存预测 - 平衡版本流水线启动 🚢")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data(train_path, test_path)
        
        # 2. 探索性数据分析
        self.exploratory_data_analysis()
        
        # 3. 平衡特征工程
        self.balanced_feature_engineering()
        
        # 4. 模型训练
        model_scores = self.train_balanced_models()
        
        # 5. 特征重要性分析
        self.feature_importance_analysis()
        
        # 6. 预测
        submission = self.make_predictions()
        
        print("\n" + "=" * 60)
        print("🎉 平衡版本流水线执行完成！")
        print("📊 模型性能总结:")
        for model_name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model_name}: {score:.4f}")
        print("📁 输出文件:")
        print("   - titanic_submission_balanced.csv (平衡预测结果)")
        print("   - titanic_eda_balanced.png (数据分析图表)")
        print("   - feature_importance_balanced.png (特征重要性图表)")
        print("\n💡 平衡版本特点:")
        print("   - 简化特征工程，减少过拟合风险")
        print("   - 保守的模型参数，增强泛化能力")
        print("   - 保留经过验证的核心特征")
        print("   - 平衡复杂度与稳定性")
        
        return submission

# 主程序执行
if __name__ == "__main__":
    # 创建平衡预测器实例
    predictor = TitanicSurvivalPredictorBalanced()
    
    # 运行完整平衡流水线
    submission = predictor.run_complete_balanced_pipeline('train.csv', 'test.csv')
    
    print("\n🏆 平衡版本预测完成！期待更稳定的Kaggle分数！")