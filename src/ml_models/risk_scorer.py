
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RiskScorer:
    """
    Machine Learning model for driver risk assessment
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.is_trained = False
        self.logger = self._setup_logging()

    def _default_config(self) -> Dict:
        """Default configuration for ML models"""
        return {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'risk_categories': {
                'low': (0, 0.33),
                'medium': (0.33, 0.67),
                'high': (0.67, 1.0)
            },
            'models_to_train': ['random_forest', 'logistic_regression', 'gradient_boosting'],
            'hyperparameter_tuning': False,  # Disable for speed
            'feature_importance_threshold': 0.01
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the model"""
        logger = logging.getLogger('RiskScorer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def prepare_features(self, features_df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features for training"""

        print(f"Original features shape: {features_df.shape}")
        print(f"Columns: {list(features_df.columns)}")

        # Handle missing values only for numeric columns
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            features_df[numeric_columns] = features_df[numeric_columns].fillna(features_df[numeric_columns].median())

        # Create synthetic risk labels if not provided
        if target_column is None or target_column not in features_df.columns:
            risk_score = self._calculate_synthetic_risk_score(features_df)
            features_df['risk_score'] = risk_score
            target_column = 'risk_score'

        # Exclude non-numeric columns and target column
        feature_cols = [col for col in features_df.columns 
                       if col not in ['driver_id', target_column] and 
                       features_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

        print(f"Selected feature columns: {feature_cols}")

        if len(feature_cols) == 0:
            raise ValueError("No numeric features found for training!")

        X = features_df[feature_cols]
        y = features_df[target_column] if target_column in features_df.columns else None

        # Store feature names
        self.feature_names = feature_cols

        # Remove any remaining NaN values
        if X.isnull().any().any():
            X = X.fillna(0)

        self.logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        print(f"Final X shape: {X.shape}, y shape: {y.shape if y is not None else 'None'}")

        return X, y

    def _calculate_synthetic_risk_score(self, features_df: pd.DataFrame) -> pd.Series:
        """Calculate synthetic risk score based on telematics features"""

        # Simplified risk calculation with fallbacks
        risk_scores = []

        for idx, row in features_df.iterrows():
            score = 0.5  # Base score

            # Add risk based on available features
            if 'harsh_braking_count' in row:
                score += min(row['harsh_braking_count'] / 50.0, 0.2)  # Max 0.2 contribution

            if 'harsh_acceleration_count' in row:
                score += min(row['harsh_acceleration_count'] / 50.0, 0.2)

            if 'over_speed_ratio' in row:
                score += min(row['over_speed_ratio'], 0.2)

            if 'phone_usage_ratio' in row:
                score += min(row['phone_usage_ratio'], 0.1)

            if 'night_driving_ratio' in row:
                score += min(row['night_driving_ratio'] * 0.1, 0.1)

            # Add some random variation
            score += np.random.normal(0, 0.1)

            # Clamp to [0, 1]
            risk_scores.append(max(0, min(score, 1)))

        return pd.Series(risk_scores)

    def create_risk_categories(self, risk_scores: pd.Series) -> pd.Series:
        """Convert continuous risk scores to categories"""
        categories = []

        for score in risk_scores:
            if score <= self.config['risk_categories']['low'][1]:
                categories.append('low')
            elif score <= self.config['risk_categories']['medium'][1]:
                categories.append('medium')
            else:
                categories.append('high')

        return pd.Series(categories)

    def train_models(self, features_df: pd.DataFrame, target_column: str = None) -> Dict[str, Dict]:
        """Train multiple ML models for risk assessment"""

        try:
            X, y = self.prepare_features(features_df, target_column)
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return {}

        if len(X) < 5:
            self.logger.error("Not enough samples for training")
            return {}

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], 
                random_state=self.config['random_state']
            )
        except Exception as e:
            self.logger.error(f"Data splitting failed: {e}")
            return {}

        # Create both regression and classification targets
        y_categories = self.create_risk_categories(y)
        try:
            y_train_cat, y_test_cat = train_test_split(
                y_categories, test_size=self.config['test_size'], 
                random_state=self.config['random_state']
            )
        except Exception as e:
            self.logger.error(f"Category splitting failed: {e}")
            return {}

        results = {}

        for model_name in self.config['models_to_train']:
            self.logger.info(f"Training {model_name} model...")

            try:
                if model_name == 'random_forest':
                    results[model_name] = self._train_random_forest(X_train, X_test, y_train, y_test,
                                                                   y_train_cat, y_test_cat)
                elif model_name == 'logistic_regression':
                    results[model_name] = self._train_logistic_regression(X_train, X_test, y_train, y_test,
                                                                        y_train_cat, y_test_cat)
                elif model_name == 'gradient_boosting':
                    results[model_name] = self._train_gradient_boosting(X_train, X_test, y_train, y_test,
                                                                      y_train_cat, y_test_cat)

                print(f"{model_name} trained successfully")

            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                print(f"{model_name} training failed: {str(e)}")
                continue

        if results:
            self.is_trained = True
            self.logger.info("Model training completed")
        else:
            self.logger.error("All model training failed")

        return results

    def _train_random_forest(self, X_train, X_test, y_train, y_test, y_train_cat, y_test_cat):
        """Train Random Forest models"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['random_forest'] = scaler

        # Regression model
        rf_reg = RandomForestRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            random_state=self.config['random_state'],
            n_jobs=-1
        )

        rf_reg.fit(X_train_scaled, y_train)

        # Classification model
        rf_clf = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            random_state=self.config['random_state'],
            n_jobs=-1
        )

        rf_clf.fit(X_train_scaled, y_train_cat)

        # Store models
        self.models['random_forest'] = {
            'regressor': rf_reg,
            'classifier': rf_clf
        }

        # Evaluate
        reg_pred = rf_reg.predict(X_test_scaled)
        clf_pred = rf_clf.predict(X_test_scaled)
        clf_pred_proba = rf_clf.predict_proba(X_test_scaled)

        return {
            'regression_mse': mean_squared_error(y_test, reg_pred),
            'regression_r2': r2_score(y_test, reg_pred),
            'classification_auc': roc_auc_score(y_test_cat, clf_pred_proba, multi_class='ovr'),
            'feature_importance': dict(zip(self.feature_names, rf_reg.feature_importances_))
        }

    def _train_logistic_regression(self, X_train, X_test, y_train, y_test, y_train_cat, y_test_cat):
        """Train Logistic Regression model"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['logistic_regression'] = scaler

        # Linear regression for continuous scores
        lr_reg = LinearRegression()
        lr_reg.fit(X_train_scaled, y_train)

        # Logistic regression for categories
        lr_clf = LogisticRegression(
            random_state=self.config['random_state'],
            max_iter=1000,
            multi_class='ovr'
        )

        lr_clf.fit(X_train_scaled, y_train_cat)

        # Store models
        self.models['logistic_regression'] = {
            'regressor': lr_reg,
            'classifier': lr_clf
        }

        # Evaluate
        reg_pred = lr_reg.predict(X_test_scaled)
        clf_pred = lr_clf.predict(X_test_scaled)
        clf_pred_proba = lr_clf.predict_proba(X_test_scaled)

        # Feature importance from coefficients
        if hasattr(lr_clf, 'coef_') and lr_clf.coef_.shape[0] > 0:
            feature_importance = dict(zip(self.feature_names, np.abs(lr_clf.coef_[0])))
        else:
            feature_importance = {}

        return {
            'regression_mse': mean_squared_error(y_test, reg_pred),
            'regression_r2': r2_score(y_test, reg_pred),
            'classification_auc': roc_auc_score(y_test_cat, clf_pred_proba, multi_class='ovr'),
            'feature_importance': feature_importance
        }

    def _train_gradient_boosting(self, X_train, X_test, y_train, y_test, y_train_cat, y_test_cat):
        """Train Gradient Boosting models"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['gradient_boosting'] = scaler

        # Regression model
        gb_reg = GradientBoostingRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config['random_state']
        )

        gb_reg.fit(X_train_scaled, y_train)

        # Classification model
        gb_clf = GradientBoostingClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config['random_state']
        )

        gb_clf.fit(X_train_scaled, y_train_cat)

        # Store models
        self.models['gradient_boosting'] = {
            'regressor': gb_reg,
            'classifier': gb_clf
        }

        # Evaluate
        reg_pred = gb_reg.predict(X_test_scaled)
        clf_pred = gb_clf.predict(X_test_scaled)
        clf_pred_proba = gb_clf.predict_proba(X_test_scaled)

        return {
            'regression_mse': mean_squared_error(y_test, reg_pred),
            'regression_r2': r2_score(y_test, reg_pred),
            'classification_auc': roc_auc_score(y_test_cat, clf_pred_proba, multi_class='ovr'),
            'feature_importance': dict(zip(self.feature_names, gb_reg.feature_importances_))
        }

    def predict_risk_score(self, features: pd.DataFrame, model_name: str = 'random_forest') -> np.ndarray:
        """Predict risk scores for new data"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")

        # Use first available model if requested model not found
        if model_name not in self.models:
            available_models = list(self.models.keys())
            if not available_models:
                raise ValueError("No trained models available")
            model_name = available_models[0]
            print(f"Model {model_name} not found, using {available_models[0]}")

        # Prepare features
        feature_cols = [col for col in features.columns if col in self.feature_names]
        X = features[feature_cols]

        # Handle missing features
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0  # Default value

        # Reorder to match training
        X = X[self.feature_names]
        X = X.fillna(0)  # Handle any NaN values

        # Scale features
        X_scaled = self.scalers[model_name].transform(X)

        # Predict
        risk_scores = self.models[model_name]['regressor'].predict(X_scaled)

        return np.clip(risk_scores, 0, 1)  # Ensure scores are in [0, 1]

    def predict_risk_category(self, features: pd.DataFrame, model_name: str = 'random_forest') -> Tuple[np.ndarray, np.ndarray]:
        """Predict risk categories and probabilities"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")

        if model_name not in self.models:
            available_models = list(self.models.keys())
            if not available_models:
                raise ValueError("No trained models available")
            model_name = available_models[0]

        # Prepare features (same as above)
        feature_cols = [col for col in features.columns if col in self.feature_names]
        X = features[feature_cols]

        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0

        X = X[self.feature_names]
        X = X.fillna(0)
        X_scaled = self.scalers[model_name].transform(X)

        # Predict
        categories = self.models[model_name]['classifier'].predict(X_scaled)
        probabilities = self.models[model_name]['classifier'].predict_proba(X_scaled)

        return categories, probabilities

    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return {}

        if model_name not in self.models:
            available_models = list(self.models.keys())
            if not available_models:
                return {}
            model_name = available_models[0]

        if hasattr(self.models[model_name]['regressor'], 'feature_importances_'):
            return dict(zip(self.feature_names, 
                          self.models[model_name]['regressor'].feature_importances_))
        else:
            return {}

    def save_models(self, filepath: str):
        """Save trained models to disk"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Models saved to {filepath}")

    def load_models(self, filepath: str):
        """Load trained models from disk"""
        model_data = joblib.load(filepath)

        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']

        self.logger.info(f"Models loaded from {filepath}")

class ModelEvaluator:
    """
    Evaluate and compare different ML models
    """

    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare model performance"""
        if not results:
            print("⚠️  No models were successfully trained")
            return pd.DataFrame()

        comparison_data = []

        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Regression MSE': metrics.get('regression_mse', np.nan),
                'Regression R²': metrics.get('regression_r2', np.nan),
                'Classification AUC': metrics.get('classification_auc', np.nan)
            })

        return pd.DataFrame(comparison_data)

    @staticmethod
    def print_model_comparison(results: Dict[str, Dict]):
        """Print formatted model comparison"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)

        if not results:
            print("⚠️  No models were successfully trained")
            print("Check the error messages above for troubleshooting guidance")
            return

        df = ModelEvaluator.compare_models(results)
        print(df.to_string(index=False))

        # Find best models only if we have data
        if not df.empty and 'Regression R²' in df.columns and 'Classification AUC' in df.columns:
            try:
                best_regression = df.loc[df['Regression R²'].idxmax(), 'Model']
                best_classification = df.loc[df['Classification AUC'].idxmax(), 'Model']

                print(f"\nBest Regression Model: {best_regression}")
                print(f"Best Classification Model: {best_classification}")
            except Exception as e:
                print(f"\nCould not determine best models: {e}")

        print("\nModel training completed!")
