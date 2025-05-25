import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免Qt相关警告
import matplotlib.pyplot as plt
import logging
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd  # 添加pandas导入
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kalman_dl.log'),
        logging.StreamHandler()
    ]
)

class Config:
    """配置管理类"""
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.default_config = {
            'state_dim': 4,
            'measurement_dim': 2,
            'hidden_dims': [64, 128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'random_seed': 42,
            'model_save_path': 'models',
            'use_gpu': torch.cuda.is_available()
        }
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """加载配置文件，如果不存在则创建默认配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            with open(self.config_path, 'w') as f:
                json.dump(self.default_config, f, indent=4)
            return self.default_config
    
    def save_config(self):
        """保存当前配置到文件"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

class KalmanFilter:
    """扩展的卡尔曼滤波器类"""
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # 初始化状态估计
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        
        # 系统模型
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)
        
        # 历史记录
        self.history = {
            'predictions': [],
            'updates': [],
            'covariances': []
        }
        
    def predict(self) -> np.ndarray:
        """预测步骤"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # 记录历史
        self.history['predictions'].append(self.x.copy())
        self.history['covariances'].append(self.P.copy())
        
        return self.x.flatten()
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """更新步骤"""
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        
        # 记录历史
        self.history['updates'].append(self.x.copy())
        
        return self.x.flatten()
    
    def reset(self):
        """重置滤波器状态"""
        self.x = np.zeros((self.dim_x, 1))
        self.P = np.eye(self.dim_x)
        self.history = {
            'predictions': [],
            'updates': [],
            'covariances': []
        }

class DeepLearningModel(nn.Module):
    """扩展的深度学习模型类"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.2):
        super(DeepLearningModel, self).__init__()
        
        # 构建多层网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class KalmanDeepLearning:
    """扩展的卡尔曼滤波+深度学习组合模型类"""
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if config.config['use_gpu'] else 'cpu')
        
        # 初始化组件
        self.kf = KalmanFilter(config.config['state_dim'], config.config['measurement_dim'])
        self.model = DeepLearningModel(
            config.config['state_dim'] + config.config['measurement_dim'],
            config.config['hidden_dims'],
            config.config['state_dim'],
            config.config['dropout_rate']
        ).to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.config['learning_rate']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # 数据预处理
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        logging.info(f"Model initialized on {self.device}")
        
    def preprocess_data(self, measurements: np.ndarray, true_states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """数据预处理"""
        # 标准化
        measurements_scaled = self.scaler_x.fit_transform(measurements)
        true_states_scaled = self.scaler_y.fit_transform(true_states)
        
        # 转换为张量
        measurements_tensor = torch.tensor(measurements_scaled, dtype=torch.float32)
        true_states_tensor = torch.tensor(true_states_scaled, dtype=torch.float32)
        
        return measurements_tensor, true_states_tensor
    
    def train(self, measurements: np.ndarray, true_states: np.ndarray):
        """训练模型"""
        # 数据预处理
        measurements_tensor, true_states_tensor = self.preprocess_data(measurements, true_states)
        
        # 划分训练集和验证集
        train_measurements, val_measurements, train_states, val_states = train_test_split(
            measurements_tensor, true_states_tensor,
            test_size=self.config.config['validation_split'],
            random_state=self.config.config['random_seed']
        )
        
        # 创建数据加载器
        train_dataset = KalmanDataset(train_measurements, train_states)
        val_dataset = KalmanDataset(val_measurements, val_states)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.config['batch_size'],
            shuffle=False
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_measurements, batch_true_states in train_loader:
                batch_measurements = batch_measurements.to(self.device)
                batch_true_states = batch_true_states.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 卡尔曼滤波预测
                kf_predictions = []
                for measurement in batch_measurements:
                    measurement_np = measurement.cpu().numpy().reshape(-1, 1)
                    kf_pred = self.kf.predict()
                    kf_predictions.append(kf_pred)
                    self.kf.update(measurement_np)
                
                kf_predictions = np.stack(kf_predictions, axis=0)
                kf_predictions = torch.tensor(kf_predictions, dtype=torch.float32).to(self.device)
                
                # 深度学习预测
                combined_input = torch.cat([kf_predictions, batch_measurements], dim=1)
                predictions = self.model(combined_input)
                
                loss = self.criterion(predictions, batch_true_states)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_measurements, batch_true_states in val_loader:
                    batch_measurements = batch_measurements.to(self.device)
                    batch_true_states = batch_true_states.to(self.device)
                    
                    kf_predictions = []
                    for measurement in batch_measurements:
                        measurement_np = measurement.cpu().numpy().reshape(-1, 1)
                        kf_pred = self.kf.predict()
                        kf_predictions.append(kf_pred)
                        self.kf.update(measurement_np)
                    
                    kf_predictions = np.stack(kf_predictions, axis=0)
                    kf_predictions = torch.tensor(kf_predictions, dtype=torch.float32).to(self.device)
                    
                    combined_input = torch.cat([kf_predictions, batch_measurements], dim=1)
                    predictions = self.model(combined_input)
                    
                    loss = self.criterion(predictions, batch_true_states)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # 更新学习率
            self.scheduler.step(avg_val_loss)
            
            # 记录历史
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config.config['early_stopping_patience']:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{self.config.config['epochs']}], "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, "
                           f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def predict(self, measurement: np.ndarray) -> np.ndarray:
        """预测单个样本"""
        self.model.eval()
        with torch.no_grad():
            # 标准化输入
            measurement_scaled = self.scaler_x.transform(measurement.reshape(1, -1))
            
            # 卡尔曼滤波预测
            kf_pred = self.kf.predict()
            self.kf.update(measurement.reshape(-1, 1))
            
            # 准备输入
            kf_pred_tensor = torch.tensor(kf_pred, dtype=torch.float32).to(self.device)
            measurement_tensor = torch.tensor(measurement_scaled, dtype=torch.float32).to(self.device)
            
            combined_input = torch.cat([kf_pred_tensor, measurement_tensor.flatten()], dim=0)
            combined_input = combined_input.unsqueeze(0)
            
            # 预测
            prediction = self.model(combined_input)
            
            # 反标准化
            prediction = prediction.cpu().numpy()
            prediction = self.scaler_y.inverse_transform(prediction)
            
            return prediction.flatten()
    
    def save_model(self, filename: str):
        """保存模型"""
        if not os.path.exists(self.config.config['model_save_path']):
            os.makedirs(self.config.config['model_save_path'])
        
        save_path = os.path.join(self.config.config['model_save_path'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'history': self.history
        }, save_path)
        logging.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """加载模型"""
        load_path = os.path.join(self.config.config['model_save_path'], filename)
        checkpoint = torch.load(load_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler_x = checkpoint['scaler_x']
        self.scaler_y = checkpoint['scaler_y']
        self.history = checkpoint['history']
        
        logging.info(f"Model loaded from {load_path}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def plot_prediction(self, measurements: np.ndarray, true_states: np.ndarray, num_samples: int = 100):
        """绘制预测结果"""
        # 预测
        preds = []
        for i in range(num_samples):
            pred = self.predict(measurements[i])
            preds.append(pred)
        preds = np.array(preds)
        
        # 绘制结果
        plt.figure(figsize=(15, 10))
        for i in range(true_states.shape[1]):
            plt.subplot(2, 2, i+1)
            plt.plot(true_states[:num_samples, i], label=f'True State {i+1}', linestyle='--')
            plt.plot(preds[:, i], label=f'Pred State {i+1}')
            plt.xlabel('Sample')
            plt.ylabel('State Value')
            plt.title(f'State {i+1} Prediction vs True')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('prediction_results.png')
        plt.show()
        
        # 计算并显示评估指标
        mse = np.mean((true_states[:num_samples] - preds) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_states[:num_samples] - preds), axis=0)
        
        metrics_df = pd.DataFrame({
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }, index=[f'State {i+1}' for i in range(true_states.shape[1])])
        
        print("\nPrediction Metrics:")
        print(metrics_df)
        
        # 保存指标到文件
        metrics_df.to_csv('prediction_metrics.csv')

class KalmanDataset(Dataset):
    """数据集类"""
    def __init__(self, measurements: torch.Tensor, true_states: torch.Tensor):
        self.measurements = measurements
        self.true_states = true_states
    
    def __len__(self) -> int:
        return len(self.measurements)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.measurements[idx], self.true_states[idx]

def generate_synthetic_data(n_samples: int, state_dim: int, measurement_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成数据"""
    # 生成状态序列
    true_states = np.zeros((n_samples, state_dim))
    for i in range(1, n_samples):
        true_states[i] = true_states[i-1] + np.random.randn(state_dim) * 0.1
    
    # 生成测量值
    measurements = np.zeros((n_samples, measurement_dim))
    for i in range(n_samples):
        measurements[i] = np.dot(np.random.randn(measurement_dim, state_dim), true_states[i]) + np.random.randn(measurement_dim) * 0.1
    
    return measurements, true_states

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载配置
    config = Config()
    
    # 生成数据
    measurements, true_states = generate_synthetic_data(
        n_samples=1000,
        state_dim=config.config['state_dim'],
        measurement_dim=config.config['measurement_dim']
    )
    
    # 创建模型
    model = KalmanDeepLearning(config)
    
    # 训练模型
    model.train(measurements, true_states)
    
    # 可视化训练历史
    model.plot_training_history()
    
    # 可视化预测结果
    model.plot_prediction(measurements, true_states)
    
    # 保存模型
    model.save_model('final_model.pth')
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main() 