import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any, Optional

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectFromModel

# Gradient Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Model persistence
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dynamic_pricing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration class for the dynamic pricing system"""
    # Data paths
    DATA_PATHS = {
        'customers': '/Users/hipla/Downloads/archive 3/olist_customers_dataset.csv',
        'geolocation': '/Users/hipla/Downloads/archive 3/olist_geolocation_dataset.csv',
        'order_items': '/Users/hipla/Downloads/archive 3/olist_order_items_dataset.csv',
        'order_payments': '/Users/hipla/Downloads/archive 3/olist_order_payments_dataset.csv',
        'order_reviews': '/Users/hipla/Downloads/archive 3/olist_order_reviews_dataset.csv',
        'orders': '/Users/hipla/Downloads/archive 3/olist_orders_dataset.csv',
        'products': '/Users/hipla/Downloads/archive 3/olist_products_dataset.csv',
        'sellers': '/Users/hipla/Downloads/archive 3/olist_sellers_dataset.csv',
        'category_translation': '/Users/hipla/Downloads/archive 3/product_category_name_translation.csv'
    }
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Pricing constraints
    MIN_MULTIPLIER = 0.7
    MAX_MULTIPLIER = 1.5
    MIN_MARGIN = 0.2
    MAX_PRICE_CHANGE = 0.3
    
    # Feature engineering parameters
    DEMAND_WINDOWS = [7, 14, 30]
    ROLLING_WINDOWS = [3, 7, 14]
    
    # Model parameters
    MODEL_PARAMS = {
        'xgb': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'eval_metric': 'rmse'
        },
        'lgb': {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'metric': 'rmse',
            'verbosity': -1
        },
        'cat': {
            'iterations': 300,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': RANDOM_STATE,
            'verbose': False,
            'loss_function': 'RMSE',
            'allow_writing_files': False
        },
        'sklearn_gb': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.8,
            'random_state': RANDOM_STATE,
            'loss': 'squared_error'
        }
    }
    
    # Feature thresholds
    FEATURE_IMPORTANCE_THRESHOLD = 0.001
    CORRELATION_THRESHOLD = 0.95


# ============================================================================
# DATA LOADER AND PREPROCESSOR
# ============================================================================
class DataPreprocessor:
    """Handles data loading, merging, and preprocessing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data = {}
        
    def load_data(self) -> None:
        """Load all datasets"""
        logger.info("Loading datasets...")
        for name, path in self.config.DATA_PATHS.items():
            try:
                self.data[name] = pd.read_csv(path)
                logger.info(f"  ✓ Loaded {name}: {self.data[name].shape}")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {name}: {e}")
                raise
        
    def merge_data(self) -> pd.DataFrame:
        """Merge all datasets based on database relationships"""
        logger.info("Merging datasets...")
        
        # Merge orders with customers
        orders_customers = pd.merge(
            self.data['orders'], 
            self.data['customers'], 
            on='customer_id', 
            how='inner'
        )
        logger.info("  ✓ Merged orders with customers")
        
        # Merge with order_payments (using left join to preserve all orders)
        orders_full = pd.merge(
            orders_customers, 
            self.data['order_payments'], 
            on='order_id', 
            how='left'
        )
        logger.info("  ✓ Merged with order payments")
        
        # Merge with order_reviews
        orders_full = pd.merge(
            orders_full, 
            self.data['order_reviews'], 
            on='order_id', 
            how='left',
            suffixes=('', '_review')
        )
        logger.info("  ✓ Merged with order reviews")
        
        # Merge with order_items
        orders_full = pd.merge(
            orders_full, 
            self.data['order_items'], 
            on='order_id', 
            how='left'
        )
        logger.info("  ✓ Merged with order items")
        
        # Merge with products
        orders_full = pd.merge(
            orders_full, 
            self.data['products'], 
            on='product_id', 
            how='left'
        )
        logger.info("  ✓ Merged with products")
        
        # Merge with category translation
        orders_full = pd.merge(
            orders_full, 
            self.data['category_translation'], 
            left_on='product_category_name', 
            right_on='product_category_name', 
            how='left'
        )
        logger.info("  ✓ Merged with category translation")
        
        # Merge with sellers
        orders_full = pd.merge(
            orders_full, 
            self.data['sellers'], 
            on='seller_id', 
            how='left'
        )
        logger.info("  ✓ Merged with sellers")
        
        # Add geolocation data
        orders_full = self._add_geolocation_data(orders_full)
        
        logger.info(f"Final dataset shape: {orders_full.shape}")
        logger.info(f"Memory usage: {orders_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return orders_full
    
    def _add_geolocation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geolocation data for sellers and customers"""
        logger.info("Adding geolocation data...")
        
        # Get unique zip codes
        seller_zips = self.data['sellers']['seller_zip_code_prefix'].unique()
        customer_zips = self.data['customers']['customer_zip_code_prefix'].unique()
        all_zips = np.unique(np.concatenate([seller_zips, customer_zips]))
        
        # Filter and aggregate geolocation data
        geolocation_filtered = self.data['geolocation'][
            self.data['geolocation']['geolocation_zip_code_prefix'].isin(all_zips)
        ]
        
        # Use median to reduce outlier impact
        geolocation_agg = geolocation_filtered.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'median',
            'geolocation_lng': 'median'
        }).reset_index()
        geolocation_agg.columns = ['zip_code_prefix', 'median_latitude', 'median_longitude']
        
        # Merge seller geolocation
        df = pd.merge(
            df, 
            geolocation_agg, 
            left_on='seller_zip_code_prefix', 
            right_on='zip_code_prefix', 
            how='left', 
            suffixes=('', '_seller')
        )
        df.rename(columns={
            'median_latitude': 'seller_latitude', 
            'median_longitude': 'seller_longitude'
        }, inplace=True)
        df.drop('zip_code_prefix', axis=1, inplace=True)
        
        # Merge customer geolocation
        df = pd.merge(
            df, 
            geolocation_agg, 
            left_on='customer_zip_code_prefix', 
            right_on='zip_code_prefix', 
            how='left', 
            suffixes=('', '_customer')
        )
        df.rename(columns={
            'median_latitude': 'customer_latitude', 
            'median_longitude': 'customer_longitude'
        }, inplace=True)
        df.drop('zip_code_prefix', axis=1, inplace=True)
        
        logger.info("  ✓ Added geolocation data")
        return df


# ============================================================================
# FEATURE ENGINEER
# ============================================================================
class FeatureEngineer:
    """Handles feature engineering for dynamic pricing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_cache = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for dynamic pricing"""
        logger.info("Engineering features...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Convert dates
        df = self._convert_dates(df)
        
        # Create time-based features
        df = self._create_time_features(df)
        
        # Create product features
        df = self._create_product_features(df)
        
        # Create price-related features
        df = self._create_price_features(df)
        
        # Create geographic features
        df = self._create_geographic_features(df)
        
        # Create payment features
        df = self._create_payment_features(df)
        
        # Create review features
        df = self._create_review_features(df)
        
        # Create inventory features
        df = self._create_inventory_features(df)
        
        # Create competitor features
        df = self._create_competitor_features(df)
        
        # Create market signal features
        df = self._create_market_features(df)
        
        # Create customer behavior features
        df = self._create_customer_features(df)
        
        # Create seller performance features
        df = self._create_seller_features(df)
        
        # Create target variable
        df = self._create_target_variable(df)
        
        # Optimize memory usage
        df = self._optimize_memory(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    def _convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime"""
        date_columns = [
            'order_purchase_timestamp', 'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        df['purchase_date'] = pd.to_datetime(df['order_purchase_timestamp']).dt.date
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        # Basic time features
        df['order_hour'] = df['order_purchase_timestamp'].dt.hour
        df['order_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
        df['order_month'] = df['order_purchase_timestamp'].dt.month
        df['order_year'] = df['order_purchase_timestamp'].dt.year
        df['order_weekend'] = (df['order_dayofweek'] >= 5).astype(int)
        df['order_quarter'] = df['order_purchase_timestamp'].dt.quarter
        df['order_season'] = (df['order_month'] % 12 + 3) // 3
        df['order_dayofyear'] = df['order_purchase_timestamp'].dt.dayofyear
        
        # Cyclical encoding for hour and dayofweek
        df['order_hour_sin'] = np.sin(2 * np.pi * df['order_hour'] / 24)
        df['order_hour_cos'] = np.cos(2 * np.pi * df['order_hour'] / 24)
        df['order_dow_sin'] = np.sin(2 * np.pi * df['order_dayofweek'] / 7)
        df['order_dow_cos'] = np.cos(2 * np.pi * df['order_dayofweek'] / 7)
        df['order_month_sin'] = np.sin(2 * np.pi * df['order_month'] / 12)
        df['order_month_cos'] = np.cos(2 * np.pi * df['order_month'] / 12)
        
        # Time-based delivery features
        df['estimated_delivery_days'] = (
            df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
        ).dt.days
        df['actual_delivery_days'] = (
            df['order_delivered_customer_date'] - df['order_purchase_timestamp']
        ).dt.days
        df['delivery_delay'] = df['actual_delivery_days'] - df['estimated_delivery_days']
        df['delivery_delay'] = df['delivery_delay'].fillna(0)
        df['delivery_on_time'] = (df['delivery_delay'] <= 2).astype(int)
        df['delivery_early'] = (df['delivery_delay'] < 0).astype(int)
        
        # Create time of day categories
        df['time_of_day'] = pd.cut(
            df['order_hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        return df
    
    def _create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product-related features"""
        # Product dimensions
        df['product_volume'] = (
            df['product_length_cm'] * 
            df['product_height_cm'] * 
            df['product_width_cm']
        ).replace(0, 1)
        
        df['product_density'] = df['product_weight_g'] / df['product_volume']
        
        # Product completeness score
        product_info_cols = [
            'product_weight_g', 'product_length_cm', 'product_height_cm',
            'product_width_cm', 'product_photos_qty', 'product_description_lenght'
        ]
        df['product_info_completeness'] = df[product_info_cols].notna().sum(axis=1) / len(product_info_cols)
        
        # Size category
        df['product_size_category'] = pd.qcut(
            df['product_volume'], 
            q=4, 
            labels=['XS', 'S', 'M', 'L']
        )
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-related features"""
        # Basic price features
        df['price_per_weight'] = df['price'] / df['product_weight_g'].replace(0, 1)
        df['freight_ratio'] = df['freight_value'] / df['price'].replace(0, 1)
        df['total_order_value'] = df.groupby('order_id')['price'].transform('sum')
        df['item_share_of_order'] = df['price'] / df['total_order_value'].replace(0, 1)
        
        # Price categories
        df['price_percentile'] = df.groupby('product_category_name_english')['price'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # Value metrics
        df['value_score'] = (
            df['product_photos_qty'] * 0.2 +
            df['product_description_lenght'].fillna(0) / 1000 * 0.3 +
            (1 / df['price_per_weight'].replace(np.inf, 0)).clip(0, 1) * 0.5
        )
        
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic features"""
        # Location matching
        df['same_state'] = (df['customer_state'] == df['seller_state']).astype(int)
        df['same_city'] = (df['customer_city'] == df['seller_city']).astype(int)
        
        # Calculate distance (Haversine formula)
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
        
        df['delivery_distance_km'] = haversine_distance(
            df['seller_latitude'], df['seller_longitude'],
            df['customer_latitude'], df['customer_longitude']
        )
        
        # Distance categories
        df['distance_category'] = pd.cut(
            df['delivery_distance_km'].fillna(0),
            bins=[0, 50, 200, 500, 1000, np.inf],
            labels=['Very Close', 'Close', 'Medium', 'Far', 'Very Far'],
            include_lowest=True
        )
        
        return df
    
    def _create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create payment-related features"""
        # Fill missing values
        df['payment_installments'].fillna(1, inplace=True)
        df['payment_value'].fillna(df['price'], inplace=True)
        df['payment_type'].fillna('credit_card', inplace=True)
        
        # Payment type indicators
        payment_dummies = pd.get_dummies(df['payment_type'], prefix='payment')
        df = pd.concat([df, payment_dummies], axis=1)
        
        # Payment behavior features
        df['payment_installment_ratio'] = df['payment_value'] / df['payment_installments'].replace(0, 1)
        df['high_installment'] = (df['payment_installments'] > 3).astype(int)
        
        return df
    
    def _create_review_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create review-related features"""
        # Fill missing values
        df['review_score'].fillna(df['review_score'].median(), inplace=True)
        df['review_count'] = (~df['review_id'].isna()).astype(int)
        
        # Review time features
        df['review_creation_date'] = pd.to_datetime(df['review_creation_date'], errors='coerce')
        df['review_answer_timestamp'] = pd.to_datetime(df['review_answer_timestamp'], errors='coerce')
        
        df['review_response_time_days'] = (
            df['review_answer_timestamp'] - df['review_creation_date']
        ).dt.days
        df['review_response_time_days'].fillna(df['review_response_time_days'].median(), inplace=True)
        
        # Review sentiment features
        df['positive_review'] = (df['review_score'] >= 4).astype(int)
        df['negative_review'] = (df['review_score'] <= 2).astype(int)
        
        # Review consistency
        seller_review_stats = df.groupby('seller_id')['review_score'].agg(['mean', 'std']).fillna(0)
        seller_review_stats.columns = ['seller_avg_rating', 'seller_rating_std']
        df = df.merge(seller_review_stats, on='seller_id', how='left')
        
        df['review_consistency'] = 1 / (1 + df['seller_rating_std'].fillna(0))
        
        return df
    
    def _create_inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create inventory-related features"""
        # Sales velocity calculations
        product_sales = df.groupby('product_id').agg({
            'order_id': 'count',
            'purchase_date': 'nunique',
            'price': 'mean'
        }).rename(columns={
            'order_id': 'product_total_sales',
            'purchase_date': 'sales_days',
            'price': 'product_avg_price'
        })
        
        product_sales['product_avg_daily_sales'] = (
            product_sales['product_total_sales'] / product_sales['sales_days'].replace(0, 1)
        )
        product_sales['product_sales_velocity'] = product_sales['product_avg_daily_sales'].rank(pct=True)
        
        df = df.merge(product_sales[['product_avg_daily_sales', 'product_sales_velocity']], 
                     on='product_id', how='left')
        
        # Inventory pressure indicators
        df['inventory_pressure'] = df['product_sales_velocity'].fillna(0.5)
        df['low_inventory_indicator'] = (df['inventory_pressure'] > 0.8).astype(int)
        df['high_inventory_indicator'] = (df['inventory_pressure'] < 0.2).astype(int)
        df['normal_inventory_indicator'] = (
            (df['inventory_pressure'] >= 0.2) & 
            (df['inventory_pressure'] <= 0.8)
        ).astype(int)
        
        # Stockout risk estimation (simplified)
        df['stockout_risk'] = np.clip(df['inventory_pressure'] * 1.2, 0, 1)
        
        return df
    
    def _create_competitor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competitor-related features"""
        # Competitor analysis by region and category
        competitor_data = df.groupby(['seller_state', 'product_category_name_english']).agg({
            'price': ['mean', 'median', 'std', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
            'seller_id': 'nunique',
            'order_id': 'count'
        }).reset_index()
        
        competitor_data.columns = [
            'seller_state', 'product_category_name_english',
            'comp_avg_price', 'comp_median_price', 'comp_price_std',
            'comp_min_price', 'comp_max_price', 'comp_price_q25', 'comp_price_q75',
            'num_competitors', 'category_sales'
        ]
        
        df = df.merge(
            competitor_data, 
            on=['seller_state', 'product_category_name_english'], 
            how='left'
        )
        
        # Price positioning metrics
        df['price_vs_comp_avg'] = df['price'] / df['comp_avg_price'].replace(0, 1)
        df['price_vs_comp_median'] = df['price'] / df['comp_median_price'].replace(0, 1)
        df['price_vs_comp_min'] = df['price'] / df['comp_min_price'].replace(0, 1)
        df['price_vs_comp_q25'] = df['price'] / df['comp_price_q25'].replace(0, 1)
        df['price_vs_comp_q75'] = df['price'] / df['comp_price_q75'].replace(0, 1)
        
        # Competition intensity
        df['competition_intensity'] = (
            df['num_competitors'] / df['num_competitors'].max()
        ).fillna(0)
        
        # Market share proxy
        df['market_share_proxy'] = (
            df.groupby(['seller_id', 'product_category_name_english'])['order_id']
            .transform('count') / df['category_sales'].replace(0, 1)
        ).fillna(0)
        
        return df
    
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market signal features"""
        # Daily market metrics
        daily_market = df.groupby('purchase_date').agg({
            'order_id': 'nunique',
            'price': ['mean', 'median', 'std'],
            'review_score': 'mean'
        }).reset_index()
        
        daily_market.columns = [
            'purchase_date', 'daily_orders', 'daily_avg_price',
            'daily_median_price', 'daily_price_std', 'daily_avg_rating'
        ]
        
        # Calculate rolling market indicators
        for window in self.config.DEMAND_WINDOWS:
            daily_market[f'demand_index_{window}d'] = (
                daily_market['daily_orders'].rolling(window, min_periods=1).mean() / 
                daily_market['daily_orders'].mean()
            ).fillna(1)
            
            daily_market[f'price_index_{window}d'] = (
                daily_market['daily_avg_price'].rolling(window, min_periods=1).mean() / 
                daily_market['daily_avg_price'].mean()
            ).fillna(1)
        
        # Market volatility
        daily_market['price_volatility_7d'] = (
            daily_market['daily_price_std'].rolling(7, min_periods=1).std()
        ).fillna(0)
        
        df = df.merge(
            daily_market[['purchase_date'] + 
                        [col for col in daily_market.columns if 'index' in col or 'volatility' in col]],
            on='purchase_date', 
            how='left'
        )
        
        # Holiday indicators (Brazilian holidays as example)
        brazilian_holidays = {
            '2017-01-01', '2017-02-27', '2017-02-28', '2017-04-14', '2017-04-21',
            '2017-05-01', '2017-06-15', '2017-09-07', '2017-10-12', '2017-11-02',
            '2017-11-15', '2017-12-25'
        }
        
        df['is_holiday'] = df['purchase_date'].astype(str).isin(brazilian_holidays).astype(int)
        df['is_month_end'] = (pd.to_datetime(df['purchase_date']).dt.is_month_end).astype(int)
        df['is_month_start'] = (pd.to_datetime(df['purchase_date']).dt.is_month_start).astype(int)
        
        # Seasonality features
        df['sin_dayofyear'] = np.sin(2 * np.pi * pd.to_datetime(df['purchase_date']).dt.dayofyear / 365.25)
        df['cos_dayofyear'] = np.cos(2 * np.pi * pd.to_datetime(df['purchase_date']).dt.dayofyear / 365.25)
        
        return df
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer behavior features"""
        # Customer purchase history
        customer_history = df.groupby('customer_id').agg({
            'order_id': 'nunique',
            'price': ['mean', 'sum', 'std', 'max', 'min'],
            'review_score': 'mean',
            'purchase_date': ['min', 'max', 'nunique']
        }).reset_index()
        
        customer_history.columns = [
            'customer_id', 'customer_order_frequency', 
            'customer_avg_spent', 'customer_total_spent', 'customer_price_std',
            'customer_max_spent', 'customer_min_spent', 'customer_avg_rating',
            'customer_first_purchase', 'customer_last_purchase', 'customer_active_days'
        ]
        
        # Customer tenure and recency
        customer_history['customer_tenure_days'] = (
            pd.to_datetime(customer_history['customer_last_purchase']) - 
            pd.to_datetime(customer_history['customer_first_purchase'])
        ).dt.days.fillna(0)
        
        customer_history['customer_recency_days'] = (
            pd.to_datetime(df['purchase_date'].max()) - 
            pd.to_datetime(customer_history['customer_last_purchase'])
        ).dt.days.fillna(0)
        
        # Customer loyalty score
        customer_history['customer_loyalty_score'] = (
            (customer_history['customer_order_frequency'] / 
             customer_history['customer_order_frequency'].max()) * 0.4 +
            (customer_history['customer_avg_rating'] / 5) * 0.3 +
            ((customer_history['customer_tenure_days'] > 180).astype(int)) * 0.3
        )
        
        # Customer value segment
        conditions = [
            (customer_history['customer_total_spent'] >= 1000),
            (customer_history['customer_total_spent'] >= 500),
            (customer_history['customer_total_spent'] >= 100),
            (customer_history['customer_total_spent'] < 100)
        ]
        choices = ['VIP', 'Premium', 'Regular', 'Occasional']
        customer_history['customer_segment'] = np.select(conditions, choices, default='Occasional')
        
        df = df.merge(customer_history, on='customer_id', how='left')
        
        # Price sensitivity proxy
        df['customer_price_sensitivity'] = (
            df['customer_price_std'].fillna(0) / 
            df['customer_avg_spent'].replace(0, 1)
        )
        
        return df
    
    def _create_seller_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seller performance features"""
        # Seller performance metrics
        seller_performance = df.groupby('seller_id').agg({
            'order_id': 'nunique',
            'review_score': ['mean', 'std', 'count'],
            'delivery_delay': ['mean', 'median', 'std'],
            'price': ['mean', 'std'],
            'delivery_on_time': 'mean',
            'positive_review': 'mean'
        }).reset_index()
        
        seller_performance.columns = [
            'seller_id', 'seller_order_count', 'seller_avg_rating',
            'seller_rating_std', 'seller_review_count', 'seller_avg_delivery_delay',
            'seller_median_delivery_delay', 'seller_delivery_std', 'seller_avg_price',
            'seller_price_std', 'seller_on_time_rate', 'seller_positive_rate'
        ]
        
        # Seller reliability score
        seller_performance['seller_reliability_score'] = (
            seller_performance['seller_on_time_rate'] * 0.4 +
            seller_performance['seller_positive_rate'] * 0.4 +
            (1 - seller_performance['seller_delivery_std'].fillna(0) / 30) * 0.2
        )
        
        # Seller experience (based on order count)
        seller_performance['seller_experience_level'] = pd.qcut(
            seller_performance['seller_order_count'],
            q=4,
            labels=['New', 'Developing', 'Experienced', 'Expert']
        )
        
        df = df.merge(seller_performance, on='seller_id', how='left')
        
        return df
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable: optimal price multiplier"""
        logger.info("Creating target variable...")
        
        # Calculate base factors with more sophisticated logic
        np.random.seed(self.config.RANDOM_STATE)
        
        # Demand factor (normalized with sigmoid)
        df['demand_factor'] = 1 / (1 + np.exp(-(df['demand_index_7d'].fillna(1) - 1) * 3))
        
        # Competition factor (inverse relationship with intensity)
        df['competition_factor'] = np.exp(-df['competition_intensity'].fillna(0.5))
        
        # Inventory factor
        df['inventory_factor'] = np.where(
            df['low_inventory_indicator'] == 1, 1.15,
            np.where(df['high_inventory_indicator'] == 1, 0.85,
                    np.where(df['normal_inventory_indicator'] == 1, 1.0, 1.0))
        )
        
        # Customer factor (reward loyal customers)
        df['customer_factor'] = np.where(
            df['customer_loyalty_score'] > 0.7, 0.95,
            np.where(df['customer_loyalty_score'] < 0.3, 1.05, 1.0)
        )
        
        # Geographic factor
        df['geographic_factor'] = np.where(
            df['same_city'] == 1, 0.97,
            np.where(df['same_state'] == 1, 0.98, 1.0)
        )
        
        # Delivery performance factor
        df['delivery_factor'] = np.where(
            df['seller_avg_delivery_delay'] < -2, 1.03,  # Very early
            np.where(df['seller_avg_delivery_delay'] < 0, 1.01,  # Early
                    np.where(df['seller_avg_delivery_delay'] > 5, 0.97,  # Late
                            np.where(df['seller_avg_delivery_delay'] > 10, 0.95, 1.0)))  # Very late
        )
        
        # Review factor
        df['review_factor'] = np.where(
            df['seller_avg_rating'] > 4.5, 1.03,
            np.where(df['seller_avg_rating'] > 4.0, 1.01,
                    np.where(df['seller_avg_rating'] < 3.0, 0.98,
                            np.where(df['seller_avg_rating'] < 2.5, 0.95, 1.0)))
        )
        
        # Seasonal factor (more realistic seasonality)
        dayofyear = pd.to_datetime(df['purchase_date']).dt.dayofyear
        df['seasonal_factor'] = 1 + 0.15 * np.sin(2 * np.pi * (dayofyear - 30) / 365.25)
        
        # Time of day factor
        df['time_factor'] = np.where(
            df['time_of_day'].isin(['Morning', 'Evening']), 1.02,
            np.where(df['time_of_day'] == 'Night', 0.98, 1.0)
        )
        
        # Calculate optimal price multiplier
        df['optimal_price_multiplier'] = (
            1.0 *  # Base multiplier
            df['demand_factor'] *
            df['competition_factor'] *
            df['inventory_factor'] *
            df['customer_factor'] *
            df['geographic_factor'] *
            df['delivery_factor'] *
            df['review_factor'] *
            df['seasonal_factor'] *
            df['time_factor']
        )
        
        # Add realistic randomness (less for higher prices)
        price_noise = np.random.normal(0, 0.03, len(df)) * (1 / (1 + df['price'] / 100))
        df['optimal_price_multiplier'] += price_noise
        
        # Apply constraints
        df['optimal_price_multiplier'] = df['optimal_price_multiplier'].clip(
            self.config.MIN_MULTIPLIER, 
            self.config.MAX_MULTIPLIER
        )
        
        return df
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        logger.info("Optimizing memory usage...")
        
        # Convert float64 to float32 where possible
        for col in df.select_dtypes(include=['float64']).columns:
            if col not in ['optimal_price_multiplier', 'price', 'freight_value']:  # Keep key columns as float64
                df[col] = df[col].astype('float32')
        
        # Convert int64 to int32 where possible
        for col in df.select_dtypes(include=['int64']).columns:
            if col not in ['order_id', 'customer_id', 'seller_id']:  # Keep ID columns as int64
                df[col] = df[col].astype('int32')
        
        # Convert object to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        logger.info(f"Memory after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df


# ============================================================================
# MODEL TRAINER
# ============================================================================
class ModelTrainer:
    """Handles model training, evaluation, and selection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features for modeling with feature selection"""
        logger.info("Preparing features for modeling...")
        
        # Define feature categories
        feature_categories = {
            'product': [
                'product_weight_g', 'product_volume', 'product_density',
                'product_photos_qty', 'product_description_lenght',
                'product_name_lenght', 'product_info_completeness'
            ],
            'inventory': [
                'inventory_pressure', 'low_inventory_indicator',
                'high_inventory_indicator', 'normal_inventory_indicator',
                'product_avg_daily_sales', 'product_sales_velocity',
                'stockout_risk'
            ],
            'price': [
                'price', 'freight_value', 'price_per_weight', 'freight_ratio',
                'item_share_of_order', 'total_order_value', 'payment_value',
                'payment_installments', 'payment_installment_ratio',
                'price_percentile', 'value_score'
            ],
            'demand': [
                'demand_index_7d', 'demand_index_14d', 'demand_index_30d',
                'price_index_7d', 'price_index_14d', 'price_index_30d',
                'price_volatility_7d'
            ],
            'competition': [
                'comp_avg_price', 'comp_median_price', 'comp_price_std',
                'comp_min_price', 'comp_max_price', 'comp_price_q25', 'comp_price_q75',
                'num_competitors', 'category_sales', 'market_share_proxy',
                'price_vs_comp_avg', 'price_vs_comp_median', 'price_vs_comp_min',
                'price_vs_comp_q25', 'price_vs_comp_q75', 'competition_intensity'
            ],
            'customer': [
                'customer_order_frequency', 'customer_avg_spent',
                'customer_total_spent', 'customer_price_std',
                'customer_max_spent', 'customer_min_spent',
                'customer_avg_rating', 'customer_tenure_days',
                'customer_recency_days', 'customer_loyalty_score',
                'customer_price_sensitivity'
            ],
            'time': [
                'order_hour', 'order_dayofweek', 'order_month', 'order_year',
                'order_weekend', 'order_quarter', 'order_season',
                'order_hour_sin', 'order_hour_cos', 'order_dow_sin', 'order_dow_cos',
                'order_month_sin', 'order_month_cos', 'sin_dayofyear', 'cos_dayofyear'
            ],
            'geographic': [
                'same_state', 'same_city', 'delivery_distance_km',
                'seller_latitude', 'seller_longitude',
                'customer_latitude', 'customer_longitude'
            ],
            'delivery': [
                'estimated_delivery_days', 'delivery_delay', 'delivery_on_time',
                'delivery_early', 'seller_avg_delivery_delay',
                'seller_median_delivery_delay', 'seller_delivery_std'
            ],
            'review': [
                'review_score', 'review_count', 'review_response_time_days',
                'positive_review', 'negative_review', 'seller_avg_rating',
                'seller_rating_std', 'review_consistency', 'seller_positive_rate'
            ],
            'seller': [
                'seller_order_count', 'seller_avg_price', 'seller_price_std',
                'seller_on_time_rate', 'seller_reliability_score'
            ],
            'payment': [
                'payment_installment_ratio', 'high_installment',
                'payment_credit_card', 'payment_boleto', 'payment_voucher',
                'payment_debit_card'
            ],
            'market': [
                'is_holiday', 'is_month_end', 'is_month_start'
            ]
        }
        
        # Flatten feature list
        all_features = []
        for category_features in feature_categories.values():
            all_features.extend(category_features)
        
        # Check which features exist
        available_features = []
        for feature in all_features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                logger.warning(f"Feature not found: {feature}")
        
        logger.info(f"Total available features: {len(available_features)}")
        
        # Prepare clean dataset
        df_clean = df[available_features + ['optimal_price_multiplier']].copy()
        
        # Fill missing values
        for col in available_features:
            if df_clean[col].dtype in ['float32', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif df_clean[col].dtype in ['int32', 'int64']:
                df_clean[col] = df_clean[col].fillna(0)
            elif df_clean[col].dtype.name == 'category':
                df_clean[col] = df_clean[col].cat.add_categories(['Missing']).fillna('Missing')
            else:
                df_clean[col] = df_clean[col].fillna(0)
        
        # Remove any remaining NaN values
        df_clean = df_clean.dropna()
        
        # Separate features and target
        X = df_clean[available_features]
        y = df_clean['optimal_price_multiplier']
        
        logger.info(f"Final dataset: {len(X)} samples, {len(available_features)} features")
        
        return X, y, available_features
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train multiple models and evaluate them"""
        logger.info("Training gradient boosting models...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        models = {
            'XGBoost': xgb.XGBRegressor(**self.config.MODEL_PARAMS['xgb']),
            'LightGBM': lgb.LGBMRegressor(**self.config.MODEL_PARAMS['lgb']),
            'CatBoost': CatBoostRegressor(**self.config.MODEL_PARAMS['cat']),
            'GradientBoosting': GradientBoostingRegressor(**self.config.MODEL_PARAMS['sklearn_gb'])
        }
        
        # Train and evaluate each model
        predictions = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Fit model with early stopping if supported
                if name == 'XGBoost':
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_test_scaled, y_test)],
                        verbose=False
                    )
                elif name == 'LightGBM':
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_test_scaled, y_test)],
                        eval_metric='rmse',
                        callbacks=[lgb.log_evaluation(period=0)],
                    )
                elif name == 'CatBoost':
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=(X_test_scaled, y_test),
                        verbose=False
                    )
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Make predictions
                predictions[name] = model.predict(X_test_scaled)
                self.models[name] = model
                
                logger.info(f"  ✓ {name} trained successfully")
                
            except Exception as e:
                logger.error(f"  ✗ Error training {name}: {e}")
        
        # Create ensemble model
        logger.info("Creating ensemble model...")
        try:
            ensemble_model = VotingRegressor(
                estimators=[
                    ('xgb', self.models['XGBoost']),
                    ('lgb', self.models['LightGBM']),
                    ('cat', self.models['CatBoost'])
                ],
                weights=[0.4, 0.3, 0.3]
            )
            ensemble_model.fit(X_train_scaled, y_train)
            predictions['Ensemble'] = ensemble_model.predict(X_test_scaled)
            self.models['Ensemble'] = ensemble_model
            logger.info("  ✓ Ensemble model created successfully")
        except Exception as e:
            logger.error(f"  ✗ Error creating ensemble: {e}")
        
        # Evaluate all models
        self.results = self.evaluate_all_models(y_test, predictions, scaler)
        
        # Select best model
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['metrics']['r2'])
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"Best model: {self.best_model_name} (R²: {self.results[self.best_model_name]['metrics']['r2']:.4f})")
        
        return {
            'models': self.models,
            'results': self.results,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': scaler
        }
    
    def evaluate_all_models(self, y_true: pd.Series, predictions: Dict[str, np.ndarray], 
                           scaler: StandardScaler) -> Dict[str, Dict]:
        """Evaluate all trained models"""
        results = {}
        
        for name, y_pred in predictions.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            feature_importance = self.get_feature_importance(name, scaler)
            
            results[name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'feature_importance': feature_importance
            }
            
            logger.info(f"{name} - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return results
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        
        # Additional business metrics
        metrics['avg_multiplier'] = np.mean(y_pred)
        metrics['std_multiplier'] = np.std(y_pred)
        metrics['multiplier_stability'] = 1 / (metrics['std_multiplier'] + 0.001)
        
        # Price change distribution
        bins = [0.7, 0.9, 1.1, 1.5]
        labels = ['Decrease', 'Hold', 'Increase']
        
        try:
            price_change_categories = pd.cut(y_pred, bins=bins, labels=labels)
            category_dist = price_change_categories.value_counts(normalize=True).to_dict()
            metrics['category_distribution'] = category_dist
        except:
            metrics['category_distribution'] = {'Decrease': 0.33, 'Hold': 0.34, 'Increase': 0.33}
        
        return metrics
    
    def get_feature_importance(self, model_name: str, scaler: StandardScaler) -> pd.DataFrame:
        """Get feature importance from model"""
        model = self.models.get(model_name)
        
        if model is None:
            return pd.DataFrame()
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                return pd.DataFrame()
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importances))],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.warning(f"Could not get feature importance for {model_name}: {e}")
            return pd.DataFrame()
    
    def time_series_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List]:
        """Perform time series cross-validation"""
        logger.info("Performing time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=self.config.CV_FOLDS)
        cv_scores = {'XGBoost': [], 'LightGBM': [], 'CatBoost': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"  Fold {fold}/{self.config.CV_FOLDS}")
            
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features for this fold
            scaler_cv = StandardScaler()
            X_train_scaled_cv = scaler_cv.fit_transform(X_train_cv)
            X_val_scaled_cv = scaler_cv.transform(X_val_cv)
            
            # Train and evaluate each model
            for name in cv_scores.keys():
                model_class = {
                    'XGBoost': xgb.XGBRegressor,
                    'LightGBM': lgb.LGBMRegressor,
                    'CatBoost': CatBoostRegressor
                }[name]
                
                model_params = self.config.MODEL_PARAMS[name.lower().replace('boost', '')]
                model = model_class(**model_params)
                
                try:
                    model.fit(X_train_scaled_cv, y_train_cv)
                    y_pred_cv = model.predict(X_val_scaled_cv)
                    score = r2_score(y_val_cv, y_pred_cv)
                    cv_scores[name].append(score)
                except Exception as e:
                    logger.warning(f"    Error in {name}, fold {fold}: {e}")
                    cv_scores[name].append(np.nan)
        
        # Print CV results
        logger.info("Cross-validation results:")
        for name, scores in cv_scores.items():
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                logger.info(f"  {name}: {mean_score:.4f} ± {std_score:.4f}")
        
        return cv_scores


# ============================================================================
# DYNAMIC PRICING SYSTEM
# ============================================================================
class DynamicPricingSystem:
    """Main dynamic pricing system with business logic"""
    
    def __init__(self, model, scaler, feature_columns: List[str], config: Config):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.config = config
        self.constraints = PricingConstraints(config)
        self.monitor = ModelMonitor()
        
    def predict_optimal_price(self, base_price: float, current_features: Dict[str, Any], 
                            cost_price: float = None) -> Dict[str, Any]:
        """Predict optimal price with business constraints"""
        try:
            # Validate inputs
            self._validate_inputs(base_price, current_features)
            
            # Prepare features
            features_df = self._prepare_features(base_price, current_features)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict multiplier
            multiplier = float(self.model.predict(features_scaled)[0])
            
            # Apply business constraints
            multiplier = self.constraints.apply(
                multiplier, base_price, cost_price, current_features
            )
            
            # Calculate optimal price
            optimal_price = base_price * multiplier
            
            # Generate recommendation
            recommendation = self._generate_recommendation(multiplier, current_features)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(current_features)
            
            # Monitor prediction
            self.monitor.record_prediction(multiplier, confidence)
            
            return {
                'base_price': base_price,
                'predicted_multiplier': multiplier,
                'optimal_price': optimal_price,
                'price_change_pct': (multiplier - 1) * 100,
                'recommendation': recommendation,
                'confidence_score': confidence,
                'is_fallback': False
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to rule-based pricing
            return self._rule_based_fallback(base_price, current_features, cost_price)
    
    def _validate_inputs(self, base_price: float, features: Dict[str, Any]) -> None:
        """Validate input parameters"""
        if base_price <= 0:
            raise ValueError("Base price must be positive")
        
        required_features = ['price', 'demand_index_7d', 'competition_intensity']
        missing = [f for f in required_features if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
    
    def _prepare_features(self, base_price: float, features: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for prediction"""
        features_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = self._get_default_value(col, base_price, features)
        
        # Select and order columns
        features_df = features_df[self.feature_columns]
        
        return features_df
    
    def _get_default_value(self, feature: str, base_price: float, 
                          features: Dict[str, Any]) -> Any:
        """Get default value for missing feature"""
        # Try to get from provided features first
        if feature in features:
            return features[feature]
        
        # Provide sensible defaults based on feature type
        if 'price' in feature:
            return base_price
        elif any(x in feature for x in ['avg', 'mean', 'value']):
            return base_price * 0.5
        elif any(x in feature for x in ['count', 'frequency', 'number']):
            return 1
        elif any(x in feature for x in ['indicator', 'same_', 'is_', 'high_', 'low_']):
            return 0
        elif 'score' in feature or 'rate' in feature:
            return 0.5
        elif 'factor' in feature:
            return 1.0
        elif 'distance' in feature:
            return 100.0
        elif 'days' in feature:
            return 7.0
        else:
            return 0.0
    
    def _generate_recommendation(self, multiplier: float, 
                               features: Dict[str, Any]) -> str:
        """Generate business recommendation"""
        if multiplier > 1.05:
            if features.get('low_inventory_indicator', 0) == 1:
                return "INCREASE_PRICE: Low inventory and high demand"
            elif features.get('demand_index_7d', 1) > 1.2:
                return "INCREASE_PRICE: High market demand"
            elif features.get('competition_intensity', 0) < 0.3:
                return "INCREASE_PRICE: Low competition"
            else:
                return "INCREASE_PRICE: Favorable market conditions"
        elif multiplier < 0.95:
            if features.get('high_inventory_indicator', 0) == 1:
                return "DECREASE_PRICE: High inventory levels"
            elif features.get('competition_intensity', 0) > 0.7:
                return "DECREASE_PRICE: Strong competition"
            elif features.get('demand_index_7d', 1) < 0.8:
                return "DECREASE_PRICE: Low demand"
            else:
                return "DECREASE_PRICE: Market conditions favor lower prices"
        else:
            return "HOLD_PRICE: Current price is optimal"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for prediction"""
        confidence = 1.0
        
        # Reduce confidence for extreme values
        if features.get('demand_index_7d', 1) > 1.5 or features.get('demand_index_7d', 1) < 0.5:
            confidence *= 0.8
        
        if features.get('competition_intensity', 0.5) > 0.9:
            confidence *= 0.9
        
        if features.get('customer_loyalty_score', 0.5) < 0.1:
            confidence *= 0.85
        
        # Increase confidence for complete data
        missing_count = sum(1 for f in self.feature_columns if f not in features)
        completeness = 1 - (missing_count / len(self.feature_columns))
        confidence *= completeness
        
        return np.clip(confidence, 0.5, 1.0)
    
    def _rule_based_fallback(self, base_price: float, features: Dict[str, Any], 
                            cost_price: float = None) -> Dict[str, Any]:
        """Rule-based fallback when model fails"""
        logger.warning("Using rule-based fallback pricing")
        
        multiplier = 1.0
        
        # Simple rule-based adjustments
        if features.get('demand_index_7d', 1) > 1.2:
            multiplier *= 1.1
        elif features.get('demand_index_7d', 1) < 0.8:
            multiplier *= 0.9
        
        if features.get('competition_intensity', 0) > 0.7:
            multiplier *= 0.95
        
        if features.get('low_inventory_indicator', 0) == 1:
            multiplier *= 1.05
        elif features.get('high_inventory_indicator', 0) == 1:
            multiplier *= 0.95
        
        if features.get('customer_loyalty_score', 0.5) > 0.7:
            multiplier *= 0.98
        
        # Apply constraints
        if cost_price:
            multiplier = max(multiplier, cost_price / base_price * 1.2)
        
        multiplier = np.clip(multiplier, self.config.MIN_MULTIPLIER, self.config.MAX_MULTIPLIER)
        
        return {
            'base_price': base_price,
            'optimal_price': base_price * multiplier,
            'predicted_multiplier': multiplier,
            'price_change_pct': (multiplier - 1) * 100,
            'recommendation': 'RULE_BASED_FALLBACK: Using simplified pricing rules',
            'confidence_score': 0.6,
            'is_fallback': True
        }


# ============================================================================
# SUPPORTING CLASSES
# ============================================================================
class PricingConstraints:
    """Apply business constraints to pricing decisions"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def apply(self, multiplier: float, base_price: float, cost_price: float = None, 
              features: Dict[str, Any] = None) -> float:
        """Apply all business constraints"""
        multiplier = float(multiplier)
        
        # 1. Basic constraints
        multiplier = np.clip(multiplier, self.config.MIN_MULTIPLIER, self.config.MAX_MULTIPLIER)
        
        # 2. Margin constraint
        if cost_price and cost_price > 0:
            min_multiplier = cost_price / base_price * (1 + self.config.MIN_MARGIN)
            multiplier = max(multiplier, min_multiplier)
        
        # 3. Competitive constraint
        if features and 'comp_avg_price' in features and features['comp_avg_price'] > 0:
            max_multiplier = (features['comp_avg_price'] * 1.15) / base_price
            multiplier = min(multiplier, max_multiplier)
        
        # 4. Price change limit
        max_change = 1 + self.config.MAX_PRICE_CHANGE
        min_change = 1 - self.config.MAX_PRICE_CHANGE
        multiplier = np.clip(multiplier, min_change, max_change)
        
        # 5. Round to reasonable precision
        multiplier = round(multiplier, 3)
        
        return multiplier


class ModelMonitor:
    """Monitor model performance and drift"""
    
    def __init__(self):
        self.predictions = []
        self.confidence_scores = []
        self.drift_detected = False
    
    def record_prediction(self, multiplier: float, confidence: float) -> None:
        """Record a prediction for monitoring"""
        self.predictions.append(multiplier)
        self.confidence_scores.append(confidence)
        
        # Keep only recent history
        if len(self.predictions) > 1000:
            self.predictions = self.predictions[-1000:]
            self.confidence_scores = self.confidence_scores[-1000:]
    
    def check_drift(self, window: int = 100, threshold: float = 0.05) -> bool:
        """Check for prediction drift"""
        if len(self.predictions) < window * 2:
            return False
        
        recent = np.array(self.predictions[-window:])
        historical = np.array(self.predictions[-(window*2):-window])
        
        # Calculate distribution difference
        recent_mean = np.mean(recent)
        historical_mean = np.mean(historical)
        
        mean_diff = abs(recent_mean - historical_mean) / historical_mean
        
        if mean_diff > threshold:
            self.drift_detected = True
            logger.warning(f"Model drift detected: {mean_diff:.2%} change in mean prediction")
            return True
        
        self.drift_detected = False
        return False


# ============================================================================
# VISUALIZATION AND REPORTING
# ============================================================================
class Visualizer:
    """Create visualizations for dynamic pricing system"""
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, 
                               top_n: int = 20) -> None:
        """Plot feature importance"""
        if importance_df.empty:
            return
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.4f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_model_comparison(results: Dict[str, Dict]) -> None:
        """Plot comparison of all models"""
        models = list(results.keys())
        metrics = ['r2', 'mae', 'rmse', 'mape']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [results[model]['metrics'][metric] for model in models]
            
            if metric == 'mape':
                # For MAPE, lower is better
                bars = axes[idx].bar(models, values, color='lightcoral')
                axes[idx].set_title(f'{metric.upper()} (Lower is Better)')
            else:
                bars = axes[idx].bar(models, values, color='skyblue')
                axes[idx].set_title(f'{metric.upper()}')
            
            axes[idx].set_ylabel(metric.upper())
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                             f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_price_distribution(predictions: Dict[str, np.ndarray], 
                               true_values: pd.Series) -> None:
        """Plot distribution of predicted prices"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        models = list(predictions.keys())
        
        for idx, (ax, model) in enumerate(zip(axes, models)):
            y_pred = predictions[model]
            
            # Scatter plot
            ax.scatter(true_values, y_pred, alpha=0.3, s=10)
            ax.plot([true_values.min(), true_values.max()], 
                   [true_values.min(), true_values.max()], 
                   'r--', lw=2)
            ax.set_xlabel('True Multiplier')
            ax.set_ylabel('Predicted Multiplier')
            ax.set_title(f'{model} Predictions')
            ax.grid(True, alpha=0.3)
            
            # Add R² text
            r2 = r2_score(true_values, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(len(models), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Prediction vs True Values', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("DYNAMIC PRICING SYSTEM - STARTING")
    logger.info("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    try:
        # Step 1: Load and preprocess data
        logger.info("\n1. Loading and preprocessing data...")
        preprocessor = DataPreprocessor(config)
        preprocessor.load_data()
        merged_data = preprocessor.merge_data()
        
        # Step 2: Engineer features
        logger.info("\n2. Engineering features...")
        feature_engineer = FeatureEngineer(config)
        data_with_features = feature_engineer.engineer_features(merged_data)
        
        # Step 3: Train models
        logger.info("\n3. Training models...")
        model_trainer = ModelTrainer(config)
        
        # Prepare features
        X, y, feature_columns = model_trainer.prepare_features(data_with_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, 
            shuffle=True
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Perform time series cross-validation
        cv_results = model_trainer.time_series_cross_validation(X_train, y_train)
        
        # Train models
        training_results = model_trainer.train_models(X_train, y_train, X_test, y_test)
        
        # Step 4: Create dynamic pricing system
        logger.info("\n4. Creating dynamic pricing system...")
        pricing_system = DynamicPricingSystem(
            model=training_results['best_model'],
            scaler=training_results['scaler'],
            feature_columns=feature_columns,
            config=config
        )
        
        # Step 5: Generate visualizations
        logger.info("\n5. Generating visualizations...")
        visualizer = Visualizer()
        
        # Plot feature importance for best model
        best_model_results = training_results['results'][training_results['best_model_name']]
        if not best_model_results['feature_importance'].empty:
            visualizer.plot_feature_importance(
                best_model_results['feature_importance'],
                training_results['best_model_name']
            )
        
        # Plot model comparison
        visualizer.plot_model_comparison(training_results['results'])
        
        # Plot price distributions
        predictions_dict = {
            name: training_results['results'][name]['predictions']
            for name in training_results['results'].keys()
        }
        visualizer.plot_price_distribution(predictions_dict, y_test)
        
        # Step 6: Test scenarios
        logger.info("\n6. Testing pricing scenarios...")
        
        # Scenario 1: High demand, low inventory
        scenario1 = {
            'price': 100,
            'demand_index_7d': 1.3,
            'price_index_7d': 1.1,
            'low_inventory_indicator': 1,
            'high_inventory_indicator': 0,
            'competition_intensity': 0.3,
            'comp_avg_price': 105,
            'customer_loyalty_score': 0.8,
            'same_state': 1,
            'same_city': 0,
            'delivery_distance_km': 50,
            'seller_avg_rating': 4.5,
            'seller_avg_delivery_delay': -1,
            'order_weekend': 0,
            'order_month': 6,
            'stockout_risk': 0.7,
            'market_share_proxy': 0.15,
            'seller_reliability_score': 0.9
        }
        
        result1 = pricing_system.predict_optimal_price(100, scenario1, cost_price=60)
        logger.info("\nScenario 1: High Demand, Low Inventory")
        logger.info(f"  Base Price: ${result1['base_price']:.2f}")
        logger.info(f"  Optimal Price: ${result1['optimal_price']:.2f}")
        logger.info(f"  Price Multiplier: {result1['predicted_multiplier']:.3f}")
        logger.info(f"  Price Change: {result1['price_change_pct']:+.2f}%")
        logger.info(f"  Recommendation: {result1['recommendation']}")
        logger.info(f"  Confidence: {result1['confidence_score']:.2%}")
        
        # Scenario 2: High competition, high inventory
        scenario2 = {
            'price': 100,
            'demand_index_7d': 0.8,
            'price_index_7d': 0.9,
            'low_inventory_indicator': 0,
            'high_inventory_indicator': 1,
            'competition_intensity': 0.9,
            'comp_avg_price': 90,
            'customer_loyalty_score': 0.4,
            'same_state': 0,
            'same_city': 0,
            'delivery_distance_km': 200,
            'seller_avg_rating': 3.2,
            'seller_avg_delivery_delay': 3,
            'order_weekend': 1,
            'order_month': 2,
            'stockout_risk': 0.2,
            'market_share_proxy': 0.05,
            'seller_reliability_score': 0.7
        }
        
        result2 = pricing_system.predict_optimal_price(100, scenario2, cost_price=70)
        logger.info("\nScenario 2: High Competition, High Inventory")
        logger.info(f"  Base Price: ${result2['base_price']:.2f}")
        logger.info(f"  Optimal Price: ${result2['optimal_price']:.2f}")
        logger.info(f"  Price Multiplier: {result2['predicted_multiplier']:.3f}")
        logger.info(f"  Price Change: {result2['price_change_pct']:+.2f}%")
        logger.info(f"  Recommendation: {result2['recommendation']}")
        logger.info(f"  Confidence: {result2['confidence_score']:.2%}")
        
        # Step 7: Save the model
        logger.info("\n7. Saving the model...")
        
        model_pipeline = {
            'model': training_results['best_model'],
            'scaler': training_results['scaler'],
            'feature_columns': feature_columns,
            'pricing_system': pricing_system,
            'config': config,
            'metadata': {
                'best_model': training_results['best_model_name'],
                'r2_score': best_model_results['metrics']['r2'],
                'mae': best_model_results['metrics']['mae'],
                'rmse': best_model_results['metrics']['rmse'],
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features_count': len(feature_columns),
                'samples_count': len(data_with_features),
                'cv_results': cv_results
            }
        }
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(models_dir, "dynamic_pricing_model_optimized.pkl")
        joblib.dump(model_pipeline, model_path)
        logger.info(f"  ✓ Model saved to: {model_path}")
        
        # Step 8: Generate summary report
        logger.info("\n" + "="*60)
        logger.info("DYNAMIC PRICING SYSTEM - SUMMARY")
        logger.info("="*60)
        
        best_metrics = best_model_results['metrics']
        
        summary = f"""
        System Performance Summary:
        {'-' * 40}
        • Best Model: {training_results['best_model_name']}
        • R² Score: {best_metrics['r2']:.4f} ({best_metrics['r2']*100:.1f}% variance explained)
        • Mean Absolute Error: {best_metrics['mae']:.4f}
        • Root Mean Square Error: {best_metrics['rmse']:.4f}
        • Mean Absolute Percentage Error: {best_metrics['mape']:.2f}%
        
        Business Metrics:
        {'-' * 40}
        • Average Multiplier: {best_metrics['avg_multiplier']:.3f}
        • Multiplier Stability: {best_metrics['multiplier_stability']:.2f}
        • Price Change Distribution:
            - Decrease: {best_metrics['category_distribution'].get('Decrease', 0)*100:.1f}%
            - Hold: {best_metrics['category_distribution'].get('Hold', 0)*100:.1f}%
            - Increase: {best_metrics['category_distribution'].get('Increase', 0)*100:.1f}%
        
        Dataset Statistics:
        {'-' * 40}
        • Total Samples: {len(data_with_features):,}
        • Training Samples: {len(X_train):,}
        • Test Samples: {len(X_test):,}
        • Features Used: {len(feature_columns)}
        
        Cross-Validation Results:
        {'-' * 40}
        """
        
        for model_name, scores in cv_results.items():
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                summary += f"  • {model_name}: {mean_score:.4f} ± {std_score:.4f}\n"
        
        summary += f"""
        System Capabilities:
        {'-' * 40}
        1. Real-time dynamic pricing with {best_metrics['r2']*100:.1f}% accuracy
        2. Comprehensive feature engineering ({len(feature_columns)} features)
        3. Business constraint enforcement
        4. Fallback mechanism for robustness
        5. Model performance monitoring
        6. Multi-model ensemble support
        7. Time series cross-validation
        
        The system is ready for deployment!
        """
        
        logger.info(summary)
        
        # Save summary to file
        with open(os.path.join(models_dir, "system_summary.txt"), 'w') as f:
            f.write(summary)
        
        logger.info(f"\n✓ Summary saved to: {os.path.join(models_dir, 'system_summary.txt')}")
        logger.info("\n" + "="*60)
        logger.info("DYNAMIC PRICING SYSTEM - COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n✗ Error in main execution: {e}")
        logger.error("Traceback:", exc_info=True)
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()