# -*- coding: utf-8 -*-
"""
泰坦尼克号生存预测模型 - 优化版本
基于Chris Deotte高分方案的改进实现
目标：通过深度特征工程和模型优化提升预测准确率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class TitanicSurvivalPredictorOptimized:
    """
    泰坦尼克号生存预测器 - 优化版本
    基于高分方案的深度特征工程和模型优化
    """
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, train_path, test_path):
        """
        加载训练和测试数据
        """
        print("=== 数据加载阶段 ===")
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"训练数据形状: {self.train_data.shape}")
        print(f"测试数据形状: {self.test_data.shape}")
        print("\n训练数据基本信息:")
        print(self.train_data.info())
        
    def exploratory_data_analysis(self):
        """
        探索性数据分析
        """
        print("\n=== 探索性数据分析 ===")
        
        # 生存率统计
        survival_rate = self.train_data['Survived'].mean()
        print(f"总体生存率: {survival_rate:.2%}")
        
        # 缺失值分析
        print("\n训练数据缺失值统计:")
        missing_train = self.train_data.isnull().sum()
        missing_train_pct = (missing_train / len(self.train_data)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing_train,
            '缺失比例(%)': missing_train_pct
        })
        print(missing_df[missing_df['缺失数量'] > 0])
        
        # 创建可视化图表
        self._create_visualizations()
        
    def _create_visualizations(self):
        """
        创建数据可视化图表
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('泰坦尼克号数据探索性分析 - 优化版本', fontsize=16, fontweight='bold')
        
        # 1. 生存率分布
        survival_counts = self.train_data['Survived'].value_counts()
        axes[0, 0].pie(survival_counts.values, labels=['死亡', '生存'], autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('生存率分布')
        
        # 2. 性别与生存率
        sns.barplot(data=self.train_data, x='Sex', y='Survived', ax=axes[0, 1])
        axes[0, 1].set_title('性别与生存率')
        axes[0, 1].set_ylabel('生存率')
        
        # 3. 船舱等级与生存率
        sns.barplot(data=self.train_data, x='Pclass', y='Survived', ax=axes[0, 2])
        axes[0, 2].set_title('船舱等级与生存率')
        axes[0, 2].set_ylabel('生存率')
        
        # 4. 年龄分布
        self.train_data['Age'].hist(bins=30, ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('年龄分布')
        axes[1, 0].set_xlabel('年龄')
        axes[1, 0].set_ylabel('频数')
        
        # 5. 票价分布
        self.train_data['Fare'].hist(bins=30, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title('票价分布')
        axes[1, 1].set_xlabel('票价')
        axes[1, 1].set_ylabel('频数')
        
        # 6. 登船港口与生存率
        sns.barplot(data=self.train_data, x='Embarked', y='Survived', ax=axes[1, 2])
        axes[1, 2].set_title('登船港口与生存率')
        axes[1, 2].set_ylabel('生存率')
        
        plt.tight_layout()
        plt.savefig('titanic_eda_optimized.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def advanced_feature_engineering(self):
        """
        高级特征工程 - 基于Chris Deotte方案的优化
        """
        print("\n=== 高级特征工程阶段 ===")
        
        # 合并训练和测试数据进行统一处理
        all_data = pd.concat([self.train_data, self.test_data], ignore_index=True)
        
        # 1. 基础缺失值处理
        print("处理缺失值...")
        
        # Age: 使用更精细的分组填充
        all_data['Age'] = all_data.groupby(['Sex', 'Pclass', 'SibSp', 'Parch'])['Age'].transform(
            lambda x: x.fillna(x.median()) if not x.isna().all() else x.fillna(all_data['Age'].median())
        )
        
        # Embarked: 基于票号信息填充
        all_data.loc[all_data['Ticket'] == '113572', 'Embarked'] = 'S'
        all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
        
        # Fare: 使用更精细的分组填充
        all_data['Fare'] = all_data.groupby(['Pclass', 'Embarked'])['Fare'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 2. 高级特征工程
        print("创建高级特征...")
        
        # 2.1 姓名特征工程
        self._extract_name_features(all_data)
        
        # 2.2 票号特征工程
        self._extract_ticket_features(all_data)
        
        # 2.3 船舱特征工程
        self._extract_cabin_features(all_data)
        
        # 2.4 家庭和群体特征工程
        self._extract_family_group_features(all_data)
        
        # 2.5 群体生存模式特征（仅对训练数据）
        self._extract_group_survival_patterns(all_data)
        
        # 2.6 其他高级特征
        self._extract_additional_features(all_data)
        
        # 3. 特征编码和标准化
        self._encode_features(all_data)
        
        # 4. 选择最终特征
        self._select_final_features(self.all_data_processed)
        
        print(f"高级特征工程完成，最终特征数量: {len(self.feature_columns)}")
        print(f"特征列表: {self.feature_columns}")
        
    def _extract_name_features(self, data):
        """
        从姓名中提取特征
        """
        # 提取称谓
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # 提取姓氏
        data['LastName'] = data['Name'].str.extract('([^,]+),', expand=False)
        
        # 高级称谓分类（基于Chris Deotte方案）
        title_mapping = {
            # 男性称谓
            'Mr': 'Man', 'Sir': 'Man', 'Don': 'Man', 'Rev': 'Man', 
            'Major': 'Man', 'Col': 'Man', 'Capt': 'Man', 'Jonkheer': 'Man',
            'Dr': 'Man',  # 大部分Dr是男性
            
            # 女性称谓
            'Mrs': 'Woman', 'Miss': 'Woman', 'Mme': 'Woman', 'Ms': 'Woman',
            'Lady': 'Woman', 'Mlle': 'Woman', 'the Countess': 'Woman', 'Dona': 'Woman',
            
            # 男孩称谓
            'Master': 'Boy'
        }
        
        data['Title_Group'] = data['Title'].map(title_mapping)
        data['Title_Group'].fillna('Rare', inplace=True)
        
        # 姓名长度特征
        data['Name_Length'] = data['Name'].str.len()
        
    def _extract_ticket_features(self, data):
        """
        从票号中提取特征
        """
        # 清理票号
        data['Ticket_Clean'] = data['Ticket'].str.replace('[^A-Za-z0-9]', '', regex=True)
        
        # 票号前缀
        data['Ticket_Prefix'] = data['Ticket'].str.extract('([A-Za-z]+)', expand=False)
        data['Ticket_Prefix'].fillna('None', inplace=True)
        
        # 票号数字部分
        data['Ticket_Number'] = data['Ticket'].str.extract('(\d+)', expand=False)
        data['Ticket_Number'] = pd.to_numeric(data['Ticket_Number'], errors='coerce')
        
        # 票号频率（同一票号的人数）
        data['Ticket_Freq'] = data.groupby('Ticket')['Ticket'].transform('count')
        
        # 调整后的票价（人均票价）
        data['Fare_Per_Person'] = data['Fare'] / data['Ticket_Freq']
        
        # 群体ID（基于票号前4位和人均票价）
        data['Group_ID'] = (data['Ticket_Clean'].str[:4] + '_' + 
                           data['Fare_Per_Person'].round(2).astype(str))
        
    def _extract_cabin_features(self, data):
        """
        从船舱信息中提取特征
        """
        # 是否有船舱信息
        data['Has_Cabin'] = data['Cabin'].notna().astype(int)
        
        # 船舱类型（首字母）
        data['Cabin_Type'] = data['Cabin'].str[0]
        data['Cabin_Type'].fillna('Missing', inplace=True)
        
        # 船舱数量
        data['Cabin_Count'] = data['Cabin'].str.count(' ') + 1
        data['Cabin_Count'] = data['Cabin_Count'].fillna(0)
        
    def _extract_family_group_features(self, data):
        """
        提取家庭和群体特征
        """
        # 基础家庭特征
        data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
        data['Is_Alone'] = (data['Family_Size'] == 1).astype(int)
        
        # 家庭类型
        data['Family_Type'] = 'Small'
        data.loc[data['Family_Size'].between(2, 4), 'Family_Type'] = 'Medium'
        data.loc[data['Family_Size'] >= 5, 'Family_Type'] = 'Large'
        
        # 群体大小（基于Group_ID）
        data['Group_Size'] = data.groupby('Group_ID')['Group_ID'].transform('count')
        
        # 是否为群体旅行
        data['Is_Group_Travel'] = (data['Group_Size'] > 1).astype(int)
        
    def _extract_group_survival_patterns(self, data):
        """
        提取群体生存模式特征（仅基于训练数据）
        """
        train_len = len(self.train_data)
        train_part = data[:train_len].copy()
        
        # 计算各群体的生存率
        group_survival = train_part.groupby('Group_ID')['Survived'].agg(['mean', 'count']).reset_index()
        group_survival.columns = ['Group_ID', 'Group_Survival_Rate', 'Group_Count']
        
        # 只考虑群体大小>=2的情况
        group_survival = group_survival[group_survival['Group_Count'] >= 2]
        
        # 群体生存模式
        group_survival['Group_Pattern'] = 'Mixed'
        group_survival.loc[group_survival['Group_Survival_Rate'] == 1.0, 'Group_Pattern'] = 'All_Survived'
        group_survival.loc[group_survival['Group_Survival_Rate'] == 0.0, 'Group_Pattern'] = 'All_Died'
        
        # 合并回原数据
        data = data.merge(group_survival[['Group_ID', 'Group_Survival_Rate', 'Group_Pattern']], 
                         on='Group_ID', how='left')
        
        # 填充缺失值
        data['Group_Survival_Rate'].fillna(0.5, inplace=True)  # 中性值
        data['Group_Pattern'].fillna('Unknown', inplace=True)
        
        # 按称谓分组的群体生存模式
        for title in ['Man', 'Woman', 'Boy']:
            title_group_survival = train_part[train_part['Title_Group'] == title].groupby('Group_ID')['Survived'].mean()
            data[f'{title}_Group_Survival'] = data['Group_ID'].map(title_group_survival)
            data[f'{title}_Group_Survival'].fillna(0.5, inplace=True)
        
        # 更新self.all_data_processed
        self.all_data_processed = data
        
    def _extract_additional_features(self, data):
        """
        提取其他高级特征
        """
        # 年龄分组（更细致）
        data['Age_Group'] = 'Adult'
        data.loc[data['Age'] <= 12, 'Age_Group'] = 'Child'
        data.loc[(data['Age'] > 12) & (data['Age'] <= 18), 'Age_Group'] = 'Teen'
        data.loc[(data['Age'] > 18) & (data['Age'] <= 35), 'Age_Group'] = 'Young_Adult'
        data.loc[(data['Age'] > 35) & (data['Age'] <= 60), 'Age_Group'] = 'Middle_Age'
        data.loc[data['Age'] > 60, 'Age_Group'] = 'Senior'
        
        # 票价分组（基于分位数）
        data['Fare_Group'] = pd.qcut(data['Fare'], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
        
        # 社会地位特征
        data['Social_Status'] = 'Low'
        data.loc[(data['Pclass'] == 1) | (data['Title_Group'] == 'Woman'), 'Social_Status'] = 'High'
        data.loc[(data['Pclass'] == 2) | (data['Title_Group'] == 'Boy'), 'Social_Status'] = 'Medium'
        
        # 年龄缺失标记
        data['Age_Missing'] = data['Age'].isna().astype(int)
        
        # 更新处理后的数据
        self.all_data_processed = data
        
    def _encode_features(self, data):
        """
        编码分类特征
        """
        # 使用已处理的数据
        if hasattr(self, 'all_data_processed'):
            data = self.all_data_processed.copy()
        
        # 性别编码
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        
        # 登船港口编码
        embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
        data['Embarked'] = data['Embarked'].map(embarked_mapping)
        
        # 称谓组编码
        title_group_mapping = {'Man': 0, 'Woman': 1, 'Boy': 2, 'Rare': 3}
        data['Title_Group'] = data['Title_Group'].map(title_group_mapping)
        
        # 年龄组编码
        age_group_mapping = {'Child': 0, 'Teen': 1, 'Young_Adult': 2, 'Adult': 3, 'Senior': 4}
        if 'Age_Group' in data.columns:
            data['Age_Group'] = data['Age_Group'].map(age_group_mapping)
            data['Age_Group'].fillna(3, inplace=True)  # 默认为Adult
        else:
            data['Age_Group'] = 3
        
        # 票价组编码
        fare_group_mapping = {'Very_Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4}
        if 'Fare_Group' in data.columns:
            data['Fare_Group'] = data['Fare_Group'].map(fare_group_mapping)
            data['Fare_Group'].fillna(2, inplace=True)  # 默认为Medium
        else:
            data['Fare_Group'] = 2
        
        # 家庭类型编码
        family_type_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
        if 'Family_Type' in data.columns:
            data['Family_Type'] = data['Family_Type'].map(family_type_mapping)
        else:
            data['Family_Type'] = 0
        
        # 船舱类型编码
        cabin_type_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'Missing': 8}
        if 'Cabin_Type' in data.columns:
            data['Cabin_Type'] = data['Cabin_Type'].map(cabin_type_mapping)
        else:
            data['Cabin_Type'] = 8
        
        # 群体模式编码（如果存在）
        if 'Group_Pattern' in data.columns:
            pattern_mapping = {'All_Survived': 2, 'All_Died': 0, 'Mixed': 1, 'Unknown': 1}
            data['Group_Pattern'] = data['Group_Pattern'].map(pattern_mapping)
            data['Group_Pattern'].fillna(1, inplace=True)
        else:
            data['Group_Pattern'] = 1  # 默认值
        
        # 社会地位编码
        status_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        if 'Social_Status' in data.columns:
            data['Social_Status'] = data['Social_Status'].map(status_mapping)
        else:
            data['Social_Status'] = 0
        
        # 保存处理后的数据
        self.all_data_processed = data
        
    def _select_final_features(self, data):
        """
        选择最终的特征列
        """
        # 定义所有可能的特征
        potential_features = [
            # 基础特征
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            
            # 船舱特征
            'Has_Cabin', 'Cabin_Type', 'Cabin_Count',
            
            # 家庭特征
            'Family_Size', 'Is_Alone', 'Family_Type',
            
            # 称谓特征
            'Title_Group', 'Name_Length',
            
            # 票号特征
            'Ticket_Freq', 'Fare_Per_Person', 'Group_Size', 'Is_Group_Travel',
            
            # 群体生存模式特征
            'Group_Survival_Rate', 'Group_Pattern',
            'Man_Group_Survival', 'Woman_Group_Survival', 'Boy_Group_Survival',
            
            # 分组特征
            'Age_Group', 'Fare_Group', 'Social_Status',
            
            # 其他特征
            'Age_Missing'
        ]
        
        # 只选择实际存在的特征
        self.feature_columns = [col for col in potential_features if col in data.columns]
        
        print(f"实际可用特征: {len(self.feature_columns)}个")
        print(f"缺失特征: {set(potential_features) - set(self.feature_columns)}")
        
        # 分离训练和测试数据
        train_len = len(self.train_data)
        self.train_processed = data[:train_len][self.feature_columns + ['Survived']]
        self.test_processed = data[train_len:][self.feature_columns]
        
    def train_optimized_models(self):
        """
        训练优化的机器学习模型
        """
        print("\n=== 优化模型训练阶段 ===")
        
        # 准备训练数据
        X = self.train_processed.drop('Survived', axis=1)
        y = self.train_processed['Survived']
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 定义优化的模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_split=4,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, 
                subsample=0.8, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=0.1, random_state=42, max_iter=1000
            ),
            'SVM': SVC(
                C=10, kernel='rbf', gamma='scale', probability=True, random_state=42
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
        
        # 创建优化的集成模型
        print("\n创建优化集成模型...")
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', models['RandomForest']),
                ('xgb', models['XGBoost']),
                ('gb', models['GradientBoosting']),
                ('svm', models['SVM'])
            ],
            voting='soft'
        )
        
        # 训练集成模型
        voting_clf.fit(X_train, y_train)
        
        # 评估集成模型
        ensemble_cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')
        ensemble_val_pred = voting_clf.predict(X_val)
        ensemble_val_accuracy = accuracy_score(y_val, ensemble_val_pred)
        
        print(f"优化集成模型交叉验证准确率: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
        print(f"优化集成模型验证集准确率: {ensemble_val_accuracy:.4f}")
        
        # 保存最佳模型
        self.models['Optimized_Ensemble'] = voting_clf
        model_scores['Optimized_Ensemble'] = ensemble_cv_scores.mean()
        
        # 选择最佳模型
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name} (准确率: {model_scores[best_model_name]:.4f})")
        
        return model_scores
    
    def hyperparameter_optimization(self):
        """
        超参数优化
        """
        print("\n=== 超参数优化阶段 ===")
        
        # 准备数据
        X = self.train_processed.drop('Survived', axis=1)
        y = self.train_processed['Survived']
        X_scaled = self.scaler.fit_transform(X)
        
        # XGBoost超参数优化
        print("优化XGBoost超参数...")
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42),
            xgb_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        xgb_grid.fit(X_scaled, y)
        
        print(f"XGBoost最佳参数: {xgb_grid.best_params_}")
        print(f"XGBoost最佳得分: {xgb_grid.best_score_:.4f}")
        
        # 保存优化后的模型
        self.models['XGBoost_Optimized'] = xgb_grid.best_estimator_
        
        return xgb_grid.best_estimator_
    
    def make_predictions(self):
        """
        使用最佳模型进行预测
        """
        print("\n=== 预测阶段 ===")
        
        # 准备测试数据
        X_test = self.test_processed
        X_test_scaled = self.scaler.transform(X_test)
        
        # 使用最佳模型进行预测
        predictions = self.best_model.predict(X_test_scaled)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'PassengerId': self.test_data['PassengerId'],
            'Survived': predictions.astype(int)
        })
        
        # 保存预测结果
        submission.to_csv('titanic_submission_optimized.csv', index=False)
        print("预测完成，结果已保存到 titanic_submission_optimized.csv")
        
        # 显示预测统计
        survival_rate = predictions.mean()
        print(f"预测生存率: {survival_rate:.2%}")
        print(f"预测生存人数: {predictions.sum()}/{len(predictions)}")
        
        return submission
    
    def feature_importance_analysis(self):
        """
        分析特征重要性
        """
        print("\n=== 特征重要性分析 ===")
        
        # 获取XGBoost模型的特征重要性
        if 'XGBoost' in self.models:
            xgb_model = self.models['XGBoost']
            feature_names = self.feature_columns
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                '特征': feature_names,
                '重要性': xgb_model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            print("\n特征重要性排序 (XGBoost):")
            for idx, row in importance_df.head(15).iterrows():
                print(f"{row['特征']}: {row['重要性']:.4f}")
            
            # 可视化特征重要性
            plt.figure(figsize=(12, 10))
            sns.barplot(data=importance_df.head(15), x='重要性', y='特征')
            plt.title('Top 15 特征重要性 (优化版本)')
            plt.xlabel('重要性分数')
            plt.tight_layout()
            plt.savefig('feature_importance_optimized.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def run_complete_optimized_pipeline(self, train_path, test_path):
        """
        运行完整的优化机器学习流水线
        """
        print("🚢 泰坦尼克号生存预测 - 优化版本流水线启动 🚢")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data(train_path, test_path)
        
        # 2. 探索性数据分析
        self.exploratory_data_analysis()
        
        # 3. 高级特征工程
        self.advanced_feature_engineering()
        
        # 4. 模型训练
        model_scores = self.train_optimized_models()
        
        # 5. 超参数优化
        try:
            optimized_model = self.hyperparameter_optimization()
            # 更新最佳模型
            if 'XGBoost_Optimized' in self.models:
                cv_scores = cross_val_score(optimized_model, 
                                          self.scaler.transform(self.train_processed.drop('Survived', axis=1)), 
                                          self.train_processed['Survived'], 
                                          cv=5, scoring='accuracy')
                model_scores['XGBoost_Optimized'] = cv_scores.mean()
                if cv_scores.mean() > max([score for name, score in model_scores.items() if name != 'XGBoost_Optimized']):
                    self.best_model = optimized_model
        except Exception as e:
            print(f"超参数优化跳过: {e}")
        
        # 6. 特征重要性分析
        self.feature_importance_analysis()
        
        # 7. 预测
        submission = self.make_predictions()
        
        print("\n" + "=" * 60)
        print("🎉 优化版本流水线执行完成！")
        print("📊 模型性能总结:")
        for model_name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model_name}: {score:.4f}")
        print("📁 输出文件:")
        print("   - titanic_submission_optimized.csv (优化预测结果)")
        print("   - titanic_eda_optimized.png (数据分析图表)")
        print("   - feature_importance_optimized.png (特征重要性图表)")
        
        return submission

# 主程序执行
if __name__ == "__main__":
    # 创建优化预测器实例
    predictor = TitanicSurvivalPredictorOptimized()
    
    # 运行完整优化流水线
    submission = predictor.run_complete_optimized_pipeline('train.csv', 'test.csv')
    
    print("\n🏆 优化版本预测完成！期待更高的Kaggle分数！")