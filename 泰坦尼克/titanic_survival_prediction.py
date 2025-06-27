# -*- coding: utf-8 -*-
"""
泰坦尼克号生存预测模型
作者：Google顶级工程师
目标：通过机器学习预测乘客生存情况，争取获得高分
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class TitanicSurvivalPredictor:
    """
    泰坦尼克号生存预测器
    使用多种机器学习算法进行集成学习，提高预测准确率
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
        
        Args:
            train_path (str): 训练数据路径
            test_path (str): 测试数据路径
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
        分析数据分布、缺失值、特征相关性等
        """
        print("\n=== 探索性数据分析 ===")
        
        # 1. 生存率统计
        survival_rate = self.train_data['Survived'].mean()
        print(f"总体生存率: {survival_rate:.2%}")
        
        # 2. 缺失值分析
        print("\n训练数据缺失值统计:")
        missing_train = self.train_data.isnull().sum()
        missing_train_pct = (missing_train / len(self.train_data)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing_train,
            '缺失比例(%)': missing_train_pct
        })
        print(missing_df[missing_df['缺失数量'] > 0])
        
        print("\n测试数据缺失值统计:")
        missing_test = self.test_data.isnull().sum()
        missing_test_pct = (missing_test / len(self.test_data)) * 100
        missing_test_df = pd.DataFrame({
            '缺失数量': missing_test,
            '缺失比例(%)': missing_test_pct
        })
        print(missing_test_df[missing_test_df['缺失数量'] > 0])
        
        # 3. 创建可视化图表
        self._create_visualizations()
        
    def _create_visualizations(self):
        """
        创建数据可视化图表
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('泰坦尼克号数据探索性分析', fontsize=16, fontweight='bold')
        
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
        plt.savefig('titanic_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_engineering(self):
        """
        特征工程
        处理缺失值、创建新特征、编码分类变量
        """
        print("\n=== 特征工程阶段 ===")
        
        # 合并训练和测试数据进行统一处理
        all_data = pd.concat([self.train_data, self.test_data], ignore_index=True)
        
        # 1. 处理缺失值
        print("处理缺失值...")
        
        # Age: 使用中位数填充，按性别和船舱等级分组
        all_data['Age'] = all_data.groupby(['Sex', 'Pclass'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Embarked: 使用众数填充
        all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
        
        # Fare: 使用中位数填充，按船舱等级分组
        all_data['Fare'] = all_data.groupby('Pclass')['Fare'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Cabin: 创建是否有船舱信息的特征
        all_data['Has_Cabin'] = all_data['Cabin'].notna().astype(int)
        
        # 2. 创建新特征
        print("创建新特征...")
        
        # 家庭规模
        all_data['Family_Size'] = all_data['SibSp'] + all_data['Parch'] + 1
        
        # 是否独自一人
        all_data['Is_Alone'] = (all_data['Family_Size'] == 1).astype(int)
        
        # 从姓名中提取称谓
        all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # 合并稀少的称谓
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        all_data['Title'] = all_data['Title'].map(title_mapping)
        all_data['Title'].fillna('Rare', inplace=True)
        
        # 年龄分组
        all_data['Age_Group'] = pd.cut(all_data['Age'], 
                                     bins=[0, 12, 18, 35, 60, 100], 
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # 票价分组
        all_data['Fare_Group'] = pd.qcut(all_data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
        
        # 3. 编码分类变量
        print("编码分类变量...")
        
        # 性别编码
        all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})
        
        # 登船港口编码
        embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
        all_data['Embarked'] = all_data['Embarked'].map(embarked_mapping)
        
        # 称谓编码
        title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
        all_data['Title'] = all_data['Title'].map(title_mapping)
        
        # 年龄组编码
        age_group_mapping = {'Child': 0, 'Teen': 1, 'Adult': 2, 'Middle': 3, 'Senior': 4}
        all_data['Age_Group'] = all_data['Age_Group'].map(age_group_mapping)
        
        # 票价组编码
        fare_group_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3}
        all_data['Fare_Group'] = all_data['Fare_Group'].map(fare_group_mapping)
        
        # 4. 选择最终特征
        feature_columns = [
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Has_Cabin', 'Family_Size', 'Is_Alone', 'Title', 'Age_Group', 'Fare_Group'
        ]
        
        # 分离训练和测试数据
        train_len = len(self.train_data)
        self.train_processed = all_data[:train_len][feature_columns + ['Survived']]
        self.test_processed = all_data[train_len:][feature_columns]
        
        print(f"特征工程完成，最终特征数量: {len(feature_columns)}")
        print(f"特征列表: {feature_columns}")
        
    def train_models(self):
        """
        训练多个机器学习模型
        使用交叉验证评估模型性能
        """
        print("\n=== 模型训练阶段 ===")
        
        # 准备训练数据
        X = self.train_processed.drop('Survived', axis=1)
        y = self.train_processed['Survived']
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 定义模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=7, min_samples_split=6,
                min_samples_leaf=3, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0, random_state=42, max_iter=1000
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', probability=True, random_state=42
            )
        }
        
        # 训练和评估每个模型
        model_scores = {}
        
        for name, model in models.items():
            print(f"\n训练 {name} 模型...")
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
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
        
        # 创建集成模型
        print("\n创建集成模型...")
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', models['RandomForest']),
                ('gb', models['GradientBoosting']),
                ('lr', models['LogisticRegression'])
            ],
            voting='soft'
        )
        
        # 训练集成模型
        voting_clf.fit(X_train, y_train)
        
        # 评估集成模型
        ensemble_cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
        ensemble_val_pred = voting_clf.predict(X_val)
        ensemble_val_accuracy = accuracy_score(y_val, ensemble_val_pred)
        
        print(f"集成模型交叉验证准确率: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
        print(f"集成模型验证集准确率: {ensemble_val_accuracy:.4f}")
        
        # 保存最佳模型
        self.models['Ensemble'] = voting_clf
        model_scores['Ensemble'] = ensemble_cv_scores.mean()
        
        # 选择最佳模型
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name} (准确率: {model_scores[best_model_name]:.4f})")
        
        return model_scores
    
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
            'Survived': predictions.astype(int)  # 确保预测值为整数
        })
        
        # 保存预测结果
        submission.to_csv('titanic_submission.csv', index=False)
        print("预测完成，结果已保存到 titanic_submission.csv")
        
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
        
        # 获取随机森林模型的特征重要性
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']
            feature_names = self.train_processed.drop('Survived', axis=1).columns
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                '特征': feature_names,
                '重要性': rf_model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            print("\n特征重要性排序:")
            for idx, row in importance_df.iterrows():
                print(f"{row['特征']}: {row['重要性']:.4f}")
            
            # 可视化特征重要性
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(10), x='重要性', y='特征')
            plt.title('Top 10 特征重要性')
            plt.xlabel('重要性分数')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def run_complete_pipeline(self, train_path, test_path):
        """
        运行完整的机器学习流水线
        
        Args:
            train_path (str): 训练数据路径
            test_path (str): 测试数据路径
        """
        print("🚢 泰坦尼克号生存预测 - 完整流水线启动 🚢")
        print("=" * 50)
        
        # 1. 加载数据
        self.load_data(train_path, test_path)
        
        # 2. 探索性数据分析
        self.exploratory_data_analysis()
        
        # 3. 特征工程
        self.feature_engineering()
        
        # 4. 模型训练
        model_scores = self.train_models()
        
        # 5. 特征重要性分析
        self.feature_importance_analysis()
        
        # 6. 预测
        submission = self.make_predictions()
        
        print("\n=" * 50)
        print("🎉 流水线执行完成！")
        print("📊 模型性能总结:")
        for model_name, score in model_scores.items():
            print(f"   {model_name}: {score:.4f}")
        print("📁 输出文件:")
        print("   - titanic_submission.csv (预测结果)")
        print("   - titanic_eda.png (数据分析图表)")
        print("   - feature_importance.png (特征重要性图表)")
        
        return submission

# 主程序执行
if __name__ == "__main__":
    # 创建预测器实例
    predictor = TitanicSurvivalPredictor()
    
    # 运行完整流水线
    submission = predictor.run_complete_pipeline('train.csv', 'test.csv')
    
    print("\n🏆 预测完成！祝您在Kaggle竞赛中取得好成绩！")