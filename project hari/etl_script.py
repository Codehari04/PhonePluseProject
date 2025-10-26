"""
PhonePe Pulse - ETL Script
Extract, Transform, and Load data from GitHub repository to MySQL database
"""

import os
import json
import pandas as pd
import mysql.connector
from mysql.connector import Error
import requests
import shutil
from pathlib import Path
import logging
from datetime import datetime

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phonepe_etl.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'phonepe_pulse',
    'user': 'root',
    'password': 'your_password'
}

# GitHub Repository Details
GITHUB_REPO = "https://github.com/PhonePe/pulse.git"
CLONE_DIR = "./phonepe_pulse_data"

class PhonePeETL:
    """ETL Pipeline for PhonePe Pulse Data"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
        
    def connect_database(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
    def close_connection(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")
    
    def clone_repository(self):
        """Clone PhonePe Pulse GitHub repository"""
        try:
            if os.path.exists(CLONE_DIR):
                logger.info(f"Directory {CLONE_DIR} already exists. Removing...")
                shutil.rmtree(CLONE_DIR)
            
            logger.info(f"Cloning repository from {GITHUB_REPO}...")
            os.system(f"git clone {GITHUB_REPO} {CLONE_DIR}")
            logger.info("Repository cloned successfully")
            return True
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    
    def extract_aggregated_transaction(self):
        """Extract aggregated transaction data from JSON files"""
        logger.info("Extracting aggregated transaction data...")
        
        data_path = Path(CLONE_DIR) / "data" / "aggregated" / "transaction" / "country" / "india" / "state"
        all_records = []
        
        try:
            if not data_path.exists():
                logger.warning(f"Path does not exist: {data_path}")
                return pd.DataFrame()
            
            for state_dir in data_path.iterdir():
                if state_dir.is_dir():
                    state_name = state_dir.name.replace("-", " ").title()
                    
                    for year_dir in state_dir.iterdir():
                        if year_dir.is_dir():
                            year = year_dir.name
                            
                            for json_file in year_dir.glob("*.json"):
                                quarter = json_file.stem
                                
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                    
                                    if 'data' in data and 'transactionData' in data['data']:
                                        for transaction in data['data']['transactionData']:
                                            record = {
                                                'state': state_name,
                                                'year': int(year),
                                                'quarter': int(quarter),
                                                'transaction_type': transaction['name'],
                                                'transaction_count': transaction['paymentInstruments'][0]['count'],
                                                'transaction_amount': transaction['paymentInstruments'][0]['amount']
                                            }
                                            all_records.append(record)
            
            df = pd.DataFrame(all_records)
            logger.info(f"Extracted {len(df)} aggregated transaction records")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting aggregated transaction data: {e}")
            return pd.DataFrame()
    
    def extract_aggregated_user(self):
        """Extract aggregated user data from JSON files"""
        logger.info("Extracting aggregated user data...")
        
        data_path = Path(CLONE_DIR) / "data" / "aggregated" / "user" / "country" / "india" / "state"
        all_records = []
        
        try:
            if not data_path.exists():
                logger.warning(f"Path does not exist: {data_path}")
                return pd.DataFrame()
            
            for state_dir in data_path.iterdir():
                if state_dir.is_dir():
                    state_name = state_dir.name.replace("-", " ").title()
                    
                    for year_dir in state_dir.iterdir():
                        if year_dir.is_dir():
                            year = year_dir.name
                            
                            for json_file in year_dir.glob("*.json"):
                                quarter = json_file.stem
                                
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                    
                                    if 'data' in data:
                                        record = {
                                            'state': state_name,
                                            'year': int(year),
                                            'quarter': int(quarter),
                                            'registered_users': data['data'].get('aggregated', {}).get('registeredUsers', 0),
                                            'app_opens': data['data'].get('aggregated', {}).get('appOpens', 0)
                                        }
                                        all_records.append(record)
            
            df = pd.DataFrame(all_records)
            logger.info(f"Extracted {len(df)} aggregated user records")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting aggregated user data: {e}")
            return pd.DataFrame()
    
    def extract_aggregated_insurance(self):
        """Extract aggregated insurance data from JSON files"""
        logger.info("Extracting aggregated insurance data...")
        
        data_path = Path(CLONE_DIR) / "data" / "aggregated" / "insurance" / "country" / "india" / "state"
        all_records = []
        
        try:
            if not data_path.exists():
                logger.warning(f"Path does not exist: {data_path}")
                return pd.DataFrame()
            
            for state_dir in data_path.iterdir():
                if state_dir.is_dir():
                    state_name = state_dir.name.replace("-", " ").title()
                    
                    for year_dir in state_dir.iterdir():
                        if year_dir.is_dir():
                            year = year_dir.name
                            
                            for json_file in year_dir.glob("*.json"):
                                quarter = json_file.stem
                                
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                    
                                    if 'data' in data and 'transactionData' in data['data']:
                                        for transaction in data['data']['transactionData']:
                                            record = {
                                                'state': state_name,
                                                'year': int(year),
                                                'quarter': int(quarter),
                                                'transaction_count': transaction.get('count', 0),
                                                'transaction_amount': transaction.get('amount', 0)
                                            }
                                            all_records.append(record)
            
            df = pd.DataFrame(all_records)
            logger.info(f"Extracted {len(df)} aggregated insurance records")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting aggregated insurance data: {e}")
            return pd.DataFrame()
    
    def extract_map_transaction(self):
        """Extract map (district-level) transaction data"""
        logger.info("Extracting map transaction data...")
        
        data_path = Path(CLONE_DIR) / "data" / "map" / "transaction" / "hover" / "country" / "india" / "state"
        all_records = []
        
        try:
            if not data_path.exists():
                logger.warning(f"Path does not exist: {data_path}")
                return pd.DataFrame()
            
            for state_dir in data_path.iterdir():
                if state_dir.is_dir():
                    state_name = state_dir.name.replace("-", " ").title()
                    
                    for year_dir in state_dir.iterdir():
                        if year_dir.is_dir():
                            year = year_dir.name
                            
                            for json_file in year_dir.glob("*.json"):
                                quarter = json_file.stem
                                
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                    
                                    if 'data' in data and 'hoverDataList' in data['data']:
                                        for district_data in data['data']['hoverDataList']:
                                            record = {
                                                'state': state_name,
                                                'district': district_data['name'],
                                                'year': int(year),
                                                'quarter': int(quarter),
                                                'transaction_count': district_data['metric'][0]['count'],
                                                'transaction_amount': district_data['metric'][0]['amount']
                                            }
                                            all_records.append(record)
            
            df = pd.DataFrame(all_records)
            logger.info(f"Extracted {len(df)} map transaction records")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting map transaction data: {e}")
            return pd.DataFrame()
    
    def extract_top_transaction(self):
        """Extract top transaction data (states, districts, pincodes)"""
        logger.info("Extracting top transaction data...")
        
        data_path = Path(CLONE_DIR) / "data" / "top" / "transaction" / "country" / "india" / "state"
        all_records = []
        
        try:
            if not data_path.exists():
                logger.warning(f"Path does not exist: {data_path}")
                return pd.DataFrame()
            
            for state_dir in data_path.iterdir():
                if state_dir.is_dir():
                    state_name = state_dir.name.replace("-", " ").title()
                    
                    for year_dir in state_dir.iterdir():
                        if year_dir.is_dir():
                            year = year_dir.name
                            
                            for json_file in year_dir.glob("*.json"):
                                quarter = json_file.stem
                                
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                    
                                    if 'data' in data:
                                        # Districts
                                        if 'districts' in data['data']:
                                            for district in data['data']['districts']:
                                                record = {
                                                    'category': 'districts',
                                                    'entity_name': district['entityName'],
                                                    'year': int(year),
                                                    'quarter': int(quarter),
                                                    'transaction_count': district['metric']['count'],
                                                    'transaction_amount': district['metric']['amount']
                                                }
                                                all_records.append(record)
                                        
                                        # Pincodes
                                        if 'pincodes' in data['data']:
                                            for pincode in data['data']['pincodes']:
                                                record = {
                                                    'category': 'pincodes',
                                                    'entity_name': pincode['entityName'],
                                                    'year': int(year),
                                                    'quarter': int(quarter),
                                                    'transaction_count': pincode['metric']['count'],
                                                    'transaction_amount': pincode['metric']['amount']
                                                }
                                                all_records.append(record)
            
            df = pd.DataFrame(all_records)
            logger.info(f"Extracted {len(df)} top transaction records")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting top transaction data: {e}")
            return pd.DataFrame()
    
    def load_data_to_db(self, df, table_name):
        """Load DataFrame to MySQL database"""
        if df.empty:
            logger.warning(f"No data to load for table {table_name}")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Clear existing data
            cursor.execute(f"DELETE FROM {table_name}")
            logger.info(f"Cleared existing data from {table_name}")
            
            # Prepare insert query
            columns = ', '.join(df.columns)
            placeholders = ', '.join(['%s'] * len(df.columns))
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            
            # Insert data in batches
            batch_size = 1000
            total_rows = len(df)
            
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                data_tuples = [tuple(row) for row in batch.values]
                cursor.executemany(insert_query, data_tuples)
                self.connection.commit()
                logger.info(f"Loaded {min(i+batch_size, total_rows)}/{total_rows} rows to {table_name}")
            
            logger.info(f"Successfully loaded {total_rows} records to {table_name}")
            cursor.close()
            return True
            
        except Error as e:
            logger.error(f"Error loading data to {table_name}: {e}")
            self.connection.rollback()
            return False
    
    def generate_top_states_data(self):
        """Generate top states data from aggregated transaction"""
        logger.info("Generating top states data...")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
                SELECT 
                    state as entity_name,
                    year,
                    quarter,
                    SUM(transaction_count) as transaction_count,
                    SUM(transaction_amount) as transaction_amount
                FROM aggregated_transaction
                GROUP BY state, year, quarter
                ORDER BY year, quarter, transaction_amount DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Add category column
            for row in results:
                row['category'] = 'states'
            
            df = pd.DataFrame(results)
            cursor.close()
            
            if not df.empty:
                # Load to top_transaction table
                cursor = self.connection.cursor()
                for _, row in df.iterrows():
                    insert_query = """
                        INSERT INTO top_transaction 
                        (category, entity_name, year, quarter, transaction_count, transaction_amount)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, tuple(row[['category', 'entity_name', 'year', 
                                                              'quarter', 'transaction_count', 
                                                              'transaction_amount']]))
                self.connection.commit()
                cursor.close()
                logger.info(f"Generated and loaded {len(df)} top states records")
            
        except Error as e:
            logger.error(f"Error generating top states data: {e}")
    
    def run_etl_pipeline(self):
        """Execute complete ETL pipeline"""
        logger.info("=" * 60)
        logger.info("Starting PhonePe Pulse ETL Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Clone repository
        if not self.clone_repository():
            logger.error("Failed to clone repository. Exiting...")
            return
        
        # Step 2: Connect to database
        if not self.connect_database():
            logger.error("Failed to connect to database. Exiting...")
            return
        
        try:
            # Step 3: Extract and Load Aggregated Transaction Data
            df_agg_trans = self.extract_aggregated_transaction()
            self.load_data_to_db(df_agg_trans, 'aggregated_transaction')
            
            # Step 4: Extract and Load Aggregated User Data
            df_agg_user = self.extract_aggregated_user()
            self.load_data_to_db(df_agg_user, 'aggregated_user')
            
            # Step 5: Extract and Load Aggregated Insurance Data
            df_agg_insurance = self.extract_aggregated_insurance()
            self.load_data_to_db(df_agg_insurance, 'aggregated_insurance')
            
            # Step 6: Extract and Load Map Transaction Data
            df_map_trans = self.extract_map_transaction()
            self.load_data_to_db(df_map_trans, 'map_transaction')
            
            # Step 7: Extract and Load Top Transaction Data
            df_top_trans = self.extract_top_transaction()
            self.load_data_to_db(df_top_trans, 'top_transaction')
            
            # Step 8: Generate Top States Data
            self.generate_top_states_data()
            
            logger.info("=" * 60)
            logger.info("ETL Pipeline Completed Successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in ETL pipeline: {e}")
        finally:
            self.close_connection()

def main():
    """Main execution function"""
    etl = PhonePeETL(DB_CONFIG)
    etl.run_etl_pipeline()

if __name__ == "__main__":
    main()